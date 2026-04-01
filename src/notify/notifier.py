"""
src/notify/notifier.py

Dispatches BirdObservation events to configured notification channels.

Supported channels (toggled via configs/notify.yaml):
    - log:   Always active. Appends JSON lines to the log file defined in
             configs/paths.yaml. Each line is a complete, parseable JSON object.
    - print: Console output — useful for development and live demos.
    - push:  Mobile push notification (Phase 5 — requires service credentials).
    - email: Email notification (Phase 5 — requires SMTP credentials).

Design principle:
    The notifier does not know or care about classification. It receives a
    finished BirdObservation and delivers it. Adding a new channel means
    adding a new _channel() method here — no other module needs to change.

Log format:
    Each line in the log file is a JSON object with all BirdObservation fields
    serialized. The log file grows indefinitely and is never truncated by this
    module — rotation is handled externally if needed.

    Example line:
        {"species_code": "HOFI", "common_name": "House Finch", ...}

Phase 3: log and print channels implemented.
Phase 5: push and email channels will be added.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from src.data.schema import BirdObservation

logger = logging.getLogger(__name__)


class Notifier:
    """
    Dispatches a BirdObservation to all active notification channels.

    Usage:
        notifier = Notifier.from_config("configs/notify.yaml", "configs/paths.yaml")
        notifier.dispatch(observation)
    """

    def __init__(
        self,
        log_path: str | Path,
        enable_print: bool = True,
        enable_push: bool = False,
        enable_email: bool = False,
        message_template: str = (
            "🐦 {common_name} ({scientific_name}) detected! " "Confidence: {confidence:.0%}"
        ),
    ) -> None:
        """
        Args:
            log_path:         Path to the JSONL log file where observations are appended.
                              Parent directory is created on first write if absent.
            enable_print:     Whether to print observations to stdout.
            enable_push:      Whether to send push notifications (Phase 5).
            enable_email:     Whether to send email notifications (Phase 5).
            message_template: Format string for console/push display.
                              Supports: {common_name}, {scientific_name},
                              {confidence}, {species_code}, {timestamp}.
        """
        self.log_path = Path(log_path)
        self.enable_print = enable_print
        self.enable_push = enable_push
        self.enable_email = enable_email
        self.message_template = message_template

    @classmethod
    def from_config(cls, notify_config_path: str, paths_config_path: str) -> Notifier:
        """
        Construct a Notifier from config YAMLs.

        Reads channel toggles from notify.yaml and log_path from paths.yaml.

        Args:
            notify_config_path: Path to configs/notify.yaml.
            paths_config_path:  Path to configs/paths.yaml (for log_path).

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
        log_path = paths_cfg.get("logs", {}).get("observations", "logs/observations.jsonl")

        template = display.get(
            "message_template",
            "🐦 {common_name} ({scientific_name}) detected! Confidence: {confidence:.0%}",
        )

        logger.info(
            "Notifier loaded: log=%s print=%s push=%s email=%s",
            log_path,
            channels.get("print", True),
            channels.get("push", False),
            channels.get("email", False),
        )

        return cls(
            log_path=log_path,
            enable_print=channels.get("print", True),
            enable_push=channels.get("push", False),
            enable_email=channels.get("email", False),
            message_template=template,
        )

    def dispatch(self, observation: BirdObservation) -> None:
        """
        Send the observation to all active channels.

        The log channel is always called first. Print, push, and email
        channels are called only if enabled. Errors in individual channels
        are logged and do not prevent other channels from running.

        Args:
            observation: A confirmed BirdObservation from the fusion module.
        """
        self._log(observation)

        if self.enable_print:
            self._print(observation)
        if self.enable_push:
            self._push(observation)
        if self.enable_email:
            self._email(observation)

    def _log(self, observation: BirdObservation) -> None:
        """
        Append the observation as a JSON line to the log file.

        Creates the parent directory if it does not exist.
        Each line is a complete JSON object — the file is newline-delimited JSON (JSONL).
        Timestamps are serialized as ISO 8601 strings.

        Args:
            observation: BirdObservation to log.
        """
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # model_dump() serializes the Pydantic model to a dict.
            # mode="json" converts datetime objects to ISO strings automatically.
            record = observation.model_dump(mode="json")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.debug(
                "Logged observation: %s (%.3f)",
                observation.species_code,
                observation.fused_confidence,
            )
        except OSError as exc:
            logger.error("Failed to write observation log: %s", exc)

    def _print(self, observation: BirdObservation) -> None:
        """
        Print a human-readable observation summary to stdout.

        Uses message_template with the following variables:
            {common_name}     — e.g. "House Finch"
            {scientific_name} — e.g. "Haemorhous mexicanus"
            {confidence}      — float in [0, 1], use :.0% for percentage
            {species_code}    — e.g. "HOFI"
            {timestamp}       — ISO 8601 UTC timestamp string

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
            logger.debug("Printed observation: %s", observation.species_code)
        except (KeyError, ValueError) as exc:
            logger.error("Failed to format print message: %s", exc)

    def _push(self, observation: BirdObservation) -> None:
        """
        Send a mobile push notification (Phase 5).

        Args:
            observation: BirdObservation to deliver.
        """
        raise NotImplementedError("Push notifications will be implemented in Phase 5.")

    def _email(self, observation: BirdObservation) -> None:
        """
        Send an email notification (Phase 5).

        Args:
            observation: BirdObservation to deliver.
        """
        raise NotImplementedError("Email notifications will be implemented in Phase 5.")
