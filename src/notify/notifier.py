"""
src/notify/notifier.py

Dispatches BirdObservation events to configured notification channels.

Supported channels (toggled via configs/notify.yaml):
    - log:     Always active. Appends JSON lines to the log file defined in
               configs/paths.yaml. Each line is a complete, parseable JSON object.
               This is the local equivalent of a database — every observation is
               persisted here regardless of other channel status.
    - print:   Console output — useful for development and live demos.
    - push:    Mobile push notification via Pushover (Phase 5+).
               Requires PUSHOVER_USER_KEY and PUSHOVER_APP_TOKEN in .env.
               Phase 6: attaches saved capture frame when available.
    - webhook: HTTP POST to a configurable backend URL (Phase 6+ stub).
               Designed for the future web app / mobile app backend.
               When active, posts the full BirdObservation JSON to webhook_url
               so a backend can persist observations, serve a UI, and trigger
               richer notifications (in-app alerts, live feed, species history).
    - email:   Email notification (future — not yet implemented).

Design principles:
    - The notifier is channel-agnostic. It receives a finished BirdObservation
      and delivers it. Adding a new channel means adding one method here —
      no other module changes.
    - Channel credentials (API keys, tokens) come from environment variables
      via python-dotenv. They are never stored in YAML configs or source code.
    - Each channel fails independently. An error in push does not prevent
      log or webhook from running.
    - The webhook channel is the bridge to a future web app. When a backend
      is built, webhook_url is configured and the channel is enabled. The
      BirdObservation schema is already rich enough to drive a full UI:
      species, confidence, timestamps, image/audio paths, dual-camera results,
      and Phase 6 stereo depth estimates.

Phase 5: log, print, push implemented. webhook stub defined.
Phase 6: push image attachment via multipart POST. push config block added.
Phase 6+: webhook implemented pointing at web app backend.

Notification channel config (configs/notify.yaml):
    channels:
        log: true
        print: true
        push: true        # Pushover — Phase 5 MVP
        webhook: false    # Future web app backend
        email: false

    push:
        attach_image: true           # Phase 6 — attach capture frame
        max_attachment_bytes: 2500000

Pushover credentials (.env):
    PUSHOVER_USER_KEY=your_user_key_here
    PUSHOVER_APP_TOKEN=your_app_token_here

Webhook config (configs/notify.yaml, when enabled):
    webhook:
        url: "https://your-backend.com/api/observations"
        timeout_seconds: 5
        auth_header: ""   # e.g. "Bearer your_token_here"
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path

import yaml

from src.data.schema import BirdObservation

logger = logging.getLogger(__name__)

# Load .env if python-dotenv is available.
# Credentials are read from environment variables, not from YAML.
# If python-dotenv is not installed, credentials must be set in the
# shell environment before running the agent.
try:
    from dotenv import load_dotenv  # type: ignore[import]

    load_dotenv()
except ImportError:
    logger.debug("python-dotenv not installed — reading credentials from environment directly.")


def _build_multipart(
    fields: dict[str, str],
    attachment: bytes,
    filename: str,
    mime_type: str,
) -> tuple[bytes, str]:
    """
    Build a multipart/form-data encoded request body.

    Used by Notifier._push() when attaching a capture frame to a Pushover
    notification. Pushover requires multipart/form-data when an attachment
    is included — a plain application/x-www-form-urlencoded POST is used
    for text-only notifications.

    Args:
        fields:     Form fields as string key-value pairs (token, user, title,
                    message, priority, sound). All values must be strings.
        attachment: Raw bytes of the file to attach.
        filename:   Filename to report in the Content-Disposition header.
                    Used by Pushover to determine display format.
        mime_type:  MIME type of the attachment (e.g. "image/jpeg", "image/png").

    Returns:
        Tuple of (encoded_body_bytes, content_type_header_value).
        The content_type_header_value includes the boundary parameter and
        should be passed directly as the Content-Type request header value.

    Example:
        body, content_type = _build_multipart(
            {"token": "abc", "user": "xyz", "message": "Bird!"},
            frame_bytes,
            "capture.jpg",
            "image/jpeg",
        )
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": content_type})
    """
    import uuid

    boundary = uuid.uuid4().hex
    body = io.BytesIO()

    for key, value in fields.items():
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        body.write(f"{value}\r\n".encode())

    body.write(f"--{boundary}\r\n".encode())
    body.write(
        f'Content-Disposition: form-data; name="attachment"; ' f'filename="{filename}"\r\n'.encode()
    )
    body.write(f"Content-Type: {mime_type}\r\n\r\n".encode())
    body.write(attachment)
    body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())

    content_type = f"multipart/form-data; boundary={boundary}"
    return body.getvalue(), content_type


class Notifier:
    """
    Dispatches a BirdObservation to all active notification channels.

    Channel architecture:
        Each channel is an independent method. Enabling/disabling a channel
        requires only a config change — no source code changes.

        Current channels:
            _log()      — JSONL append (always active, local persistence)
            _print()    — Console output (development/demo)
            _push()     — Pushover mobile push with optional image attachment
            _webhook()  — HTTP POST to backend (Phase 6+ stub)
            _email()    — Email (future stub)

    Usage:
        notifier = Notifier.from_config("configs/notify.yaml", "configs/paths.yaml")
        notifier.dispatch(observation)
    """

    def __init__(
        self,
        log_path: str | Path,
        enable_print: bool = True,
        enable_push: bool = False,
        enable_webhook: bool = False,
        enable_email: bool = False,
        message_template: str = (
            "🐦 {common_name} ({scientific_name}) detected! " "Confidence: {confidence:.0%}"
        ),
        webhook_url: str = "",
        webhook_timeout_seconds: float = 5.0,
        webhook_auth_header: str = "",
        push_attach_image: bool = True,
        push_max_attachment_bytes: int = 2_500_000,
        push_base_timeout_seconds: float = 10.0,
        push_per_kb_timeout_seconds: float = 0.1,
        push_max_attempts: int = 3,
        push_retry_backoff_seconds: float = 2.0,
    ) -> None:
        """
        Args:
            log_path:                    Path to the JSONL observation log file.
                                         Parent directory is created on first write.
            enable_print:                Whether to print observations to stdout.
            enable_push:                 Whether to send Pushover push notifications.
                                         Requires PUSHOVER_USER_KEY and
                                         PUSHOVER_APP_TOKEN environment variables.
            enable_webhook:              Whether to POST observations to webhook_url.
                                         Phase 6+ — set False until backend exists.
            enable_email:                Whether to send email notifications (future).
            message_template:            Format string for console/push display.
                                         Supports: {common_name}, {scientific_name},
                                         {confidence}, {species_code}, {timestamp}.
            webhook_url:                 Full URL for the webhook POST endpoint.
            webhook_timeout_seconds:     HTTP request timeout for webhook POSTs.
            webhook_auth_header:         Optional Authorization header value.
            push_attach_image:           If True, attach the saved capture frame to
                                         Pushover notifications when image_path is
                                         set on the observation and the file exists
                                         within push_max_attachment_bytes.
                                         Falls back to text-only silently if the
                                         file is missing, too large, or unreadable.
            push_max_attachment_bytes:   Maximum file size in bytes for push
                                         attachments. Files exceeding this are
                                         skipped and the notification is sent
                                         text-only. Pushover hard limit is 2,621,440.
            push_base_timeout_seconds:   Minimum timeout for any Pushover POST,
                                         regardless of payload size. Scales up
                                         from here with per_kb_timeout_seconds.
                                         Default 10.0s is enough for the round
                                         trip on any reasonable network.
            push_per_kb_timeout_seconds: Additional timeout budget per KB of
                                         request payload. Default 0.1s/KB
                                         handles upload speeds as low as
                                         ~10KB/s (typical for marginal WiFi
                                         at -70 dBm). A 500KB image gets
                                         60s total timeout; a 1KB text gets ~10s.
            push_max_attempts:           Total attempts to send a push including
                                         the first try. Default 3. Retries only
                                         on network errors (timeout, connection
                                         refused); API rejections are not retried.
            push_retry_backoff_seconds:  Base delay between retry attempts.
                                         Exponential backoff is applied:
                                         1st retry = backoff, 2nd = backoff*2,
                                         3rd = backoff*4. Default 2.0s gives
                                         2s/4s/8s spacing across 3 retries.
        """
        self.log_path = Path(log_path)
        self.enable_print = enable_print
        self.enable_push = enable_push
        self.enable_webhook = enable_webhook
        self.enable_email = enable_email
        self.message_template = message_template
        self.webhook_url = webhook_url
        self.webhook_timeout_seconds = webhook_timeout_seconds
        self.webhook_auth_header = webhook_auth_header
        self.push_attach_image = push_attach_image
        self.push_max_attachment_bytes = push_max_attachment_bytes
        self.push_base_timeout_seconds = push_base_timeout_seconds
        self.push_per_kb_timeout_seconds = push_per_kb_timeout_seconds
        self.push_max_attempts = push_max_attempts
        self.push_retry_backoff_seconds = push_retry_backoff_seconds
        self.push_enabled = enable_push

    @classmethod
    def from_config(cls, notify_config_path: str, paths_config_path: str) -> Notifier:
        """
        Construct a Notifier from config YAMLs.

        Reads channel toggles, push config, and webhook config from notify.yaml.
        Reads log_path from paths.yaml.
        Credentials (Pushover keys) are read from environment variables.

        Args:
            notify_config_path: Path to configs/notify.yaml.
            paths_config_path:  Path to configs/paths.yaml.

        Returns:
            Configured Notifier instance.

        Raises:
            FileNotFoundError: If either config file does not exist.
        """
        notify_path = Path(notify_config_path)
        paths_path = Path(paths_config_path)

        if not notify_path.exists():
            raise FileNotFoundError(f"Notify config not found: {notify_path}")
        if not paths_path.exists():
            raise FileNotFoundError(f"Paths config not found: {paths_path}")

        with open(notify_path, encoding="utf-8") as f:
            notify_cfg = yaml.safe_load(f)

        with open(paths_path, encoding="utf-8") as f:
            paths_cfg = yaml.safe_load(f)

        channels = notify_cfg.get("channels", {})
        display = notify_cfg.get("display", {})
        webhook_cfg = notify_cfg.get("webhook", {})
        push_cfg = notify_cfg.get("push", {})
        log_path = paths_cfg.get("logs", {}).get("observations", "logs/observations.jsonl")

        template = display.get(
            "message_template",
            "🐦 {common_name} ({scientific_name}) detected! Confidence: {confidence:.0%}",
        )

        logger.info(
            "Notifier loaded | log=%s print=%s push=%s webhook=%s email=%s attach_image=%s",
            log_path,
            channels.get("print", True),
            channels.get("push", False),
            channels.get("webhook", False),
            channels.get("email", False),
            push_cfg.get("attach_image", True),
        )

        network_cfg = push_cfg.get("network", {})

        return cls(
            log_path=log_path,
            enable_print=channels.get("print", True),
            enable_push=channels.get("push", False),
            enable_webhook=channels.get("webhook", False),
            enable_email=channels.get("email", False),
            message_template=template,
            webhook_url=webhook_cfg.get("url", ""),
            webhook_timeout_seconds=float(webhook_cfg.get("timeout_seconds", 5.0)),
            webhook_auth_header=webhook_cfg.get("auth_header", ""),
            push_attach_image=push_cfg.get("attach_image", True),
            push_max_attachment_bytes=int(push_cfg.get("max_attachment_bytes", 2_500_000)),
            push_base_timeout_seconds=float(network_cfg.get("base_timeout_seconds", 10.0)),
            push_per_kb_timeout_seconds=float(network_cfg.get("per_kb_timeout_seconds", 0.1)),
            push_max_attempts=int(network_cfg.get("max_attempts", 3)),
            push_retry_backoff_seconds=float(network_cfg.get("retry_backoff_seconds", 2.0)),
        )

    def dispatch(self, observation: BirdObservation) -> None:
        """
        Send the observation to all active channels.

        Log channel always runs first — local persistence is guaranteed
        regardless of other channel failures. Each subsequent channel runs
        independently; an exception in one does not prevent others from running.

        Args:
            observation: A confirmed BirdObservation from the fusion module.
        """
        # Log is always first — guaranteed local persistence
        self._log(observation)

        if self.enable_print:
            self._print(observation)

        if self.enable_push:
            self._push(observation)

        if self.enable_webhook:
            self._webhook(observation)

        if self.enable_email:
            self._email(observation)

    # ── Channel implementations ───────────────────────────────────────────────

    def _log(self, observation: BirdObservation) -> None:
        """
        Append the observation as a JSON line to the JSONL log file.

        Creates the parent directory if it does not exist.
        Each line is a complete JSON object — the file is newline-delimited JSON.
        Timestamps are serialized as ISO 8601 strings.

        This is the local persistence layer. It is the source of truth for
        all observations and the data source for a future web app backend
        until a database is introduced.

        Args:
            observation: BirdObservation to log.
        """
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            record = observation.model_dump(mode="json")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.debug(
                "Logged: %s (%.3f)",
                observation.species_code,
                observation.fused_confidence,
            )
        except OSError as exc:
            logger.error("Failed to write observation log: %s", exc)

    def _print(self, observation: BirdObservation) -> None:
        """
        Print a human-readable observation summary to stdout.

        Format controlled by message_template in notify.yaml.
        Available variables: {common_name}, {scientific_name},
        {confidence}, {species_code}, {timestamp}.

        Args:
            observation: BirdObservation to display.
        """
        try:
            message = self.message_template.format(
                common_name=observation.common_name,
                scientific_name=observation.scientific_name,
                confidence=observation.fused_confidence,
                species_code=observation.species_code,
                timestamp=observation.timestamp.isoformat(),
            )
            print(message)
            logger.debug("Printed: %s", observation.species_code)
        except (KeyError, ValueError) as exc:
            logger.error("Failed to format print message: %s", exc)

    def _push(self, observation: BirdObservation) -> None:
        """
        Send a mobile push notification via Pushover.

        Pushover API: single HTTPS POST to api.pushover.net/1/messages.json
        No SDK required — uses Python's built-in urllib.

        Attachment behaviour (Phase 6):
            If push.attach_image is true in notify.yaml and observation.image_path
            points to an existing file within push.max_attachment_bytes, the saved
            capture frame is attached using multipart/form-data encoding.
            Falls back silently to text-only (application/x-www-form-urlencoded)
            if any of the following are true:
                - push_attach_image is False
                - observation.image_path is None (audio-only detection)
                - the file does not exist on disk
                - the file exceeds push_max_attachment_bytes
                - the file cannot be read (OSError)
            Audio-only detections include a note in the message body.

        Phase 8 intent:
            Stock reference image and audio clip attachment are deferred to
            Phase 8 when the webhook backend can serve richer notifications.
            The audio_path is always persisted to observations.jsonl regardless.

        Credentials (.env):
            PUSHOVER_USER_KEY   — your Pushover user key
            PUSHOVER_APP_TOKEN  — your application API token

        Args:
            observation: BirdObservation to deliver.
        """
        import mimetypes
        import urllib.parse
        import urllib.request

        user_key = os.environ.get("PUSHOVER_USER_KEY", "")
        app_token = os.environ.get("PUSHOVER_APP_TOKEN", "")

        if not user_key or not app_token:
            logger.warning(
                "Pushover credentials missing — set PUSHOVER_USER_KEY and "
                "PUSHOVER_APP_TOKEN in .env. Push notification skipped."
            )
            return

        # ── Build message body ────────────────────────────────────────────
        audio_marker = "✓" if observation.audio_result else "–"
        visual_marker = "✓" if observation.visual_result else "–"
        dual_cam_note = " (dual cam)" if observation.has_dual_camera else ""

        message_lines = [
            f"{observation.scientific_name}",
            f"Confidence: {observation.fused_confidence:.0%}",
            f"Audio {audio_marker}  Visual {visual_marker}{dual_cam_note}",
        ]

        if not observation.visual_result and observation.audio_result:
            message_lines.append("Audio-only detection — no visual captured.")

        # Stereo depth — populated in Phase 6 when StereoEstimator is active
        if observation.has_stereo_estimate and observation.estimated_size_cm:
            message_lines.append(
                f"Est. size: {observation.estimated_size_cm:.1f} cm  "
                f"Depth: {observation.estimated_depth_cm:.0f} cm"
            )

        message_lines.append(observation.timestamp.strftime("%H:%M:%S UTC"))

        payload = {
            "token": app_token,
            "user": user_key,
            "title": f"🐦 {observation.common_name}",
            "message": "\n".join(message_lines),
            "priority": "0",
            "sound": "none",
        }

        # ── Resolve attachment ────────────────────────────────────────────
        attachment_bytes: bytes | None = None
        attachment_filename: str = "capture.jpg"
        attachment_mime: str = "image/jpeg"

        best_image = observation.image_path or observation.image_path_2
        if self.push_attach_image and best_image:
            image_file = Path(best_image)
            if not image_file.exists():
                logger.debug("Push attachment: image_path set but file not found — %s", image_file)
            elif image_file.stat().st_size > self.push_max_attachment_bytes:
                logger.warning(
                    "Push attachment: file exceeds %d bytes (%d) — sending text-only. "
                    "Reduce feeder_crop size or compress frames to enable attachments.",
                    self.push_max_attachment_bytes,
                    image_file.stat().st_size,
                )
            else:
                try:
                    attachment_bytes = image_file.read_bytes()
                    attachment_filename = image_file.name
                    mime, _ = mimetypes.guess_type(str(image_file))
                    attachment_mime = mime or "image/jpeg"
                    logger.debug(
                        "Push attachment: %s (%d bytes)",
                        attachment_filename,
                        len(attachment_bytes),
                    )
                except OSError as exc:
                    logger.error("Push attachment: failed to read image file — %s", exc)

        # ── POST to Pushover API ──────────────────────────────────────────
        if attachment_bytes is not None:
            data, content_type = _build_multipart(
                payload, attachment_bytes, attachment_filename, attachment_mime
            )
            context = (
                f"{observation.species_code} " f"({observation.fused_confidence:.0%}) with image"
            )
        else:
            data = urllib.parse.urlencode(payload).encode("utf-8")
            content_type = "application/x-www-form-urlencoded"
            context = (
                f"{observation.species_code} " f"({observation.fused_confidence:.0%}) text-only"
            )

        self._post_to_pushover(data, content_type, context=context)

    def _post_to_pushover(
        self,
        data: bytes,
        content_type: str,
        *,
        context: str,
    ) -> bool:
        """
        POST to the Pushover API with payload-scaled timeout and retry logic.

        Shared implementation for both image-attached (_push) and text-only
        (_push_text) notifications. Handles transient network failures common on
        edge deployments (weak WiFi, marginal cellular, variable residential
        broadband) by retrying with exponential backoff.

        Timeout calculation:
            Base + (payload_size_kb * per_kb_multiplier)
            Default: 10s base + 0.1s/KB → a 500KB image gets 60s;
                                           a 1KB text message gets ~10s.
            This handles upload speeds as low as ~10 KB/s, which is typical for
            marginal WiFi at -70 dBm or weak cellular connections.

        Retry strategy:
            - Retries on urllib.error.URLError and socket.timeout (network issues)
            - Does NOT retry on status != 1 Pushover responses (API-level rejection
              will not succeed on retry — likely invalid credentials or rate limit)
            - Exponential backoff between retries: backoff * (2 ** (attempt - 1))
            - Default: 3 attempts with 2s, 4s, 8s spacing

        Args:
            data:         Pre-encoded request body bytes.
            content_type: Value for the Content-Type header.
            context:      Short description for log messages (e.g.
                          "HOFI push with image", "daily summary text push").

        Returns:
            True if the push succeeded (either first try or after retry).
            False if all attempts failed or the API rejected the push.
        """
        import time
        import urllib.error
        import urllib.request

        payload_kb = len(data) / 1024.0
        timeout_seconds = self.push_base_timeout_seconds + (
            payload_kb * self.push_per_kb_timeout_seconds
        )

        req = urllib.request.Request(
            "https://api.pushover.net/1/messages.json",
            data=data,
            headers={"Content-Type": content_type},
            method="POST",
        )

        for attempt in range(1, self.push_max_attempts + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                    status = json.loads(body).get("status", 0)
                    if status == 1:
                        retry_note = (
                            f" (succeeded on attempt {attempt}/{self.push_max_attempts})"
                            if attempt > 1
                            else ""
                        )
                        logger.info(
                            "Pushover push sent: %s | %.1fKB, timeout=%.1fs%s",
                            context,
                            payload_kb,
                            timeout_seconds,
                            retry_note,
                        )
                        return True
                    else:
                        # API-level rejection — don't retry, these don't recover
                        logger.warning(
                            "Pushover API rejected push: %s | status=%s body=%s",
                            context,
                            status,
                            body,
                        )
                        return False

            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                # OSError is the parent of URLError and covers bare socket errors
                # on some platforms (e.g. ECONNRESET, EPIPE). All are transient.
                if attempt < self.push_max_attempts:
                    backoff = self.push_retry_backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Pushover push network error (attempt %d/%d): %s | %s | retrying in %.1fs",
                        attempt,
                        self.push_max_attempts,
                        context,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        "Pushover push failed after %d attempts: %s | last error: %s",
                        self.push_max_attempts,
                        context,
                        exc,
                    )

        return False

    def _push_text(self, message: str) -> None:  # noqa: ANN001
        """
        Push a plain text message via Pushover.

        Used by ExperimentOrchestrator for system-level notifications:
        - Boot / "Avis is live" message
        - A/B window summary after each mode rotation
        - Daily species summary

        Unlike _push(), this method does not format a BirdObservation.
        The caller supplies the complete message string.

        Falls back silently if credentials are missing or the request fails —
        the orchestrator loop must never crash because of a push failure.

        Args:
        message: The notification body text. Keep under ~500 chars
                 for clean display on Pushover mobile clients.
        """
        import os
        import urllib.error
        import urllib.parse
        import urllib.request

        user_key = os.getenv("PUSHOVER_USER_KEY", "")
        app_token = os.getenv("PUSHOVER_APP_TOKEN", "")

        if not user_key or not app_token:
            import logging

            logging.getLogger(__name__).warning(
                "Pushover credentials not set — skipping plain-text push."
            )
            return

        payload = urllib.parse.urlencode(
            {
                "token": app_token,
                "user": user_key,
                "message": message,
                "title": "Avis",
            }
        ).encode()

        self._post_to_pushover(
            payload,
            "application/x-www-form-urlencoded",
            context="system text push",
        )

    def _webhook(self, observation: BirdObservation) -> None:
        """
        POST the observation as JSON to a configured backend URL.

        Phase 6+ — stub. Raises NotImplementedError until a backend exists.

        When implemented, this channel bridges the Pi to a web/mobile app
        backend. The backend receives the full BirdObservation JSON and can:
            - Persist observations to a database
            - Store image/audio media (upload from paths in observation)
            - Trigger richer notifications (in-app alerts, websocket push)
            - Serve a species history UI
            - Drive a live feed viewer
            - Support multi-feeder aggregation

        Implementation sketch for Phase 6+:
            import urllib.request, urllib.parse

            headers = {"Content-Type": "application/json"}
            if self.webhook_auth_header:
                headers["Authorization"] = self.webhook_auth_header

            payload = observation.model_dump(mode="json")
            data    = json.dumps(payload, ensure_ascii=False).encode("utf-8")

            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.webhook_timeout_seconds) as resp:
                logger.info("Webhook posted: %s → %s", observation.species_code, self.webhook_url)

        Config (notify.yaml) when active:
            channels:
                webhook: true
            webhook:
                url: "https://api.yourdomain.com/observations"
                timeout_seconds: 5
                auth_header: "Bearer your_token_here"

        Args:
            observation: BirdObservation to deliver.

        Raises:
            NotImplementedError: Always in Phase 5/6. Implement in Phase 6+.
        """
        raise NotImplementedError(
            "Webhook channel is not yet implemented. "
            "Build a backend API endpoint that accepts BirdObservation JSON, "
            "implement this method, and set webhook_url in notify.yaml. "
            "This is a Phase 6+ task."
        )

    def _email(self, observation: BirdObservation) -> None:
        """
        Send an email notification (future).

        Implementation options:
            - SMTP via smtplib (no external dependency)
            - SendGrid / Mailgun API (reliable delivery, easy attachments)

        Useful for: daily digests, rare species alerts, extended absence alerts
        (e.g. "no birds detected in 24 hours — feeder may need refilling").

        Args:
            observation: BirdObservation to deliver.

        Raises:
            NotImplementedError: Always until implemented.
        """
        raise NotImplementedError(
            "Email notifications are not yet implemented. "
            "Options: smtplib (built-in) or SendGrid/Mailgun API. "
            "This is a future task."
        )
