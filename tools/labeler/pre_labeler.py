"""Pre-labeling agent for the Avis labeling assistant.

Batch-processes deployment-captured images through Gemini 2.5 Flash to
produce suggested species labels. Designed to be run on the laptop (not
the Pi) as a dev-time tool, typically overnight or in the background.

Pipeline for a single image:
    1. Locate matching observations.jsonl record by filename timestamp
    2. Extract audio hint from observation.audio_result if available
    3. Build prompt with image + audio hint + species reference
    4. Call Gemini 2.5 Flash with structured output
    5. Validate response via Pydantic schema
    6. Append PreLabel record to pre_labels.jsonl

Design principles:
- Resumable: re-running skips already-labeled images based on pre_labels.jsonl
- Graceful: transient Gemini errors retry once, then skip the image
- Observable: per-image progress, aggregate timing, error counts
- Auditable: every record carries the prompt version and model name
- Cheap: uses Gemini 2.5 Flash (~$0.0004 per image)

Does NOT:
- Download images from the Pi — expects them already on disk
- Modify source images
- Train or deploy anything — pre-labeling only
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from .prompts import PROMPT_VERSION, build_system_prompt, build_user_message_text
from .schema import PreLabel, PreLabelResponse

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.1  # low — we want consistent classification, not creativity
DEFAULT_MAX_RETRIES = 1     # one retry on transient failure, then skip
DEFAULT_RETRY_DELAY_SECONDS = 2.0
DEFAULT_INTER_REQUEST_DELAY = 0.1  # small delay between images to avoid hammering API

# Filename timestamp pattern: "20260424_141605_420369_cam0.png" etc.
# Groups: date, time, microseconds, camera.
FILENAME_TIMESTAMP_PATTERN = re.compile(
    r"^(\d{8})_(\d{6})_(\d{6})_cam(\d)\.png$"
)

# Audio hints below this confidence are dropped — too noisy to be useful context.
MIN_AUDIO_HINT_CONFIDENCE = 0.30


# ── Filename parsing ──────────────────────────────────────────────────────────


def parse_capture_timestamp(filename: str) -> Optional[datetime]:
    """Parse UTC capture timestamp from a Pi capture filename.

    Example: "20260424_141605_420369_cam0.png" -> 2026-04-24 14:16:05.420369 UTC

    Pi writes image filenames using UTC timestamps (as does the orchestrator),
    so this is a straightforward parse. Returns None if the filename does not
    match the expected pattern.
    """
    match = FILENAME_TIMESTAMP_PATTERN.match(filename)
    if not match:
        return None
    date_str, time_str, micro_str, _cam = match.groups()
    try:
        return datetime(
            year=int(date_str[0:4]),
            month=int(date_str[4:6]),
            day=int(date_str[6:8]),
            hour=int(time_str[0:2]),
            minute=int(time_str[2:4]),
            second=int(time_str[4:6]),
            microsecond=int(micro_str),
            tzinfo=timezone.utc,
        )
    except ValueError:
        logger.warning("Could not parse timestamp from filename: %s", filename)
        return None


# ── Observation index ─────────────────────────────────────────────────────────


class ObservationIndex:
    """Fast lookup from image filename to matching observation record.

    The pre-labeler needs to know which observation (if any) was logged for
    each image, so it can extract the audio_result as a hint to pass to
    Gemini. Building a filename->record dict once up front means we don't
    re-scan observations.jsonl for every image.

    Indexing is keyed on image_path.name (basename) rather than full path,
    because the observation records store absolute Pi paths that won't match
    our laptop-side paths.
    """

    def __init__(self) -> None:
        self._by_image_filename: dict[str, dict] = {}
        self._record_count = 0
        self._match_count = 0  # records that had a resolvable image_path

    @classmethod
    def from_jsonl(cls, observations_path: Path) -> "ObservationIndex":
        """Build an index from a local copy of observations.jsonl."""
        idx = cls()
        if not observations_path.exists():
            logger.warning(
                "Observations file not found at %s — pre-labeling will proceed "
                "without audio hints.",
                observations_path,
            )
            return idx

        with observations_path.open("rb") as fh:
            for line_num, raw in enumerate(fh, start=1):
                idx._record_count += 1
                try:
                    record = json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError as exc:
                    logger.debug(
                        "Skipping malformed observation at line %d: %s",
                        line_num,
                        exc,
                    )
                    continue

                for key in ("image_path", "image_path_2"):
                    img_path = record.get(key)
                    if not img_path:
                        continue
                    # Extract basename — observations store Pi absolute paths.
                    filename = Path(img_path).name
                    if filename:
                        idx._by_image_filename[filename] = record
                        idx._match_count += 1

        logger.info(
            "ObservationIndex built: %d records scanned, %d image paths indexed",
            idx._record_count,
            idx._match_count,
        )
        return idx

    def lookup(self, image_filename: str) -> Optional[dict]:
        """Return the observation record that references this image, or None."""
        return self._by_image_filename.get(image_filename)

    def extract_audio_hint(
        self, image_filename: str
    ) -> tuple[Optional[str], Optional[float]]:
        """Pull (species_code, confidence) from the matched observation.

        Returns (None, None) if there is no matched observation, no
        audio_result, or the audio confidence is below MIN_AUDIO_HINT_CONFIDENCE.
        """
        record = self.lookup(image_filename)
        if record is None:
            return None, None

        audio_result = record.get("audio_result")
        if not audio_result or not isinstance(audio_result, dict):
            return None, None

        species_code = audio_result.get("species_code")
        confidence = audio_result.get("confidence")
        if not species_code or confidence is None:
            return None, None
        if confidence < MIN_AUDIO_HINT_CONFIDENCE:
            return None, None

        return species_code, float(confidence)


# ── Already-labeled tracking ──────────────────────────────────────────────────


def load_already_labeled(output_path: Path) -> set[str]:
    """Read pre_labels.jsonl and return the set of already-labeled filenames.

    Lets the pre-labeler resume a batch cleanly. If output_path doesn't exist
    yet, returns an empty set.
    """
    labeled: set[str] = set()
    if not output_path.exists():
        return labeled

    with output_path.open("rb") as fh:
        for raw in fh:
            try:
                record = json.loads(raw.decode("utf-8", errors="replace"))
                filename = record.get("image_filename")
                if filename:
                    labeled.add(filename)
            except json.JSONDecodeError:
                # Malformed line — skip but don't crash the whole run
                continue
    return labeled


# ── The pre-labeler itself ────────────────────────────────────────────────────


class PreLabeler:
    """Batch pre-labeler that sends images to Gemini for species classification.

    Usage:
        labeler = PreLabeler(api_key=..., model_name="gemini-2.5-flash")
        labeler.run(
            image_dir=Path("data/captures/images"),
            observations_path=Path("logs/observations.jsonl"),
            output_path=Path("data/labels/pre_labels.jsonl"),
            limit=500,
        )

    The class is thin by design — the bulk of the work is in helpers that
    can be tested in isolation (parse_capture_timestamp, ObservationIndex,
    load_already_labeled).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        inter_request_delay: float = DEFAULT_INTER_REQUEST_DELAY,
    ) -> None:
        """Initialise the pre-labeler.

        Args:
            api_key:
                Gemini API key. If None, reads GOOGLE_API_KEY or GEMINI_API_KEY
                from the environment (GOOGLE_API_KEY takes precedence, matching
                langchain-google-genai's own lookup order).
            model_name:
                Gemini model identifier. Default is the current Flash workhorse.
            temperature:
                Sampling temperature. Keep low for consistent classification.
            max_retries:
                How many times to retry a failed Gemini call. Default 1 =
                two attempts total (initial + one retry).
            retry_delay_seconds:
                Wall-clock delay before retrying after a failure.
            inter_request_delay:
                Delay between successful requests, to avoid hammering the API.
        """
        # Lazy import so unit tests can run without langchain-google-genai
        # installed (we stub the model in tests).
        from langchain_google_genai import ChatGoogleGenerativeAI

        # LangChain prefers GOOGLE_API_KEY; our .env uses GEMINI_API_KEY.
        # Mirror langchain-google-genai's own fallback logic here so callers
        # don't have to set both.
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get(
            "GEMINI_API_KEY"
        )
        if not resolved_key:
            raise RuntimeError(
                "No Gemini API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY "
                "in the environment, or pass api_key= explicitly."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.inter_request_delay = inter_request_delay

        # Build the structured-output model once — it's reused across images.
        base_model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=resolved_key,
        )
        self._structured_model = base_model.with_structured_output(PreLabelResponse)
        self._system_prompt = build_system_prompt()

        logger.info(
            "PreLabeler initialised | model=%s temperature=%.2f prompt=%s",
            model_name,
            temperature,
            PROMPT_VERSION,
        )

    # ── Single-image labeling ────────────────────────────────────────────────

    def label_image(
        self,
        image_path: Path,
        audio_hint: Optional[str] = None,
        audio_confidence: Optional[float] = None,
    ) -> PreLabel:
        """Pre-label a single image. Raises on hard failures.

        Callers that want graceful batch behaviour should use run() which
        catches and logs errors per-image.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with image_path.open("rb") as fh:
            img_b64 = base64.b64encode(fh.read()).decode("ascii")

        user_text = build_user_message_text(audio_hint, audio_confidence)

        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{img_b64}",
                    },
                ]
            ),
        ]

        last_exc: Optional[Exception] = None
        started = time.monotonic()

        for attempt in range(self.max_retries + 1):
            try:
                response = self._structured_model.invoke(messages)
                elapsed = time.monotonic() - started
                if not isinstance(response, PreLabelResponse):
                    # Belt and braces — with_structured_output returns an
                    # instance of our schema, but if a future langchain version
                    # returns a dict we coerce rather than crash.
                    response = PreLabelResponse.model_validate(response)
                return PreLabel(
                    image_path=str(image_path.resolve()),
                    image_filename=image_path.name,
                    capture_timestamp=parse_capture_timestamp(image_path.name),
                    observation_timestamp=None,  # filled in by batch loop if needed
                    audio_hint=audio_hint,
                    audio_confidence=audio_confidence,
                    llm_response=response,
                    model_name=self.model_name,
                    prompt_version=PROMPT_VERSION,
                    elapsed_seconds=elapsed,
                )
            except ValidationError:
                # Pydantic rejected the LLM output — probably a bad species
                # code. Don't retry; the model will likely return the same
                # thing. Surface the error to the caller.
                raise
            except Exception as exc:  # noqa: BLE001 — network / transient errors
                last_exc = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "Gemini call failed for %s (attempt %d/%d): %s — retrying",
                        image_path.name,
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                    )
                    time.sleep(self.retry_delay_seconds)
                    continue
                break

        # Exhausted retries
        assert last_exc is not None
        raise last_exc

    # ── Batch run ────────────────────────────────────────────────────────────

    def run(
        self,
        image_dir: Path,
        observations_path: Optional[Path],
        output_path: Path,
        limit: Optional[int] = None,
        min_capture_time: Optional[datetime] = None,
        camera_filter: Optional[str] = None,  # "cam0", "cam1", or None for both
    ) -> dict:
        """Pre-label a batch of images, appending results to output_path.

        Args:
            image_dir:
                Directory containing capture PNGs.
            observations_path:
                Path to observations.jsonl for audio hints. May be None.
            output_path:
                Where to append PreLabel records. Created if missing.
            limit:
                Max number of NEW images to label this run. Resumed runs
                skip already-labeled images before counting toward the limit.
            min_capture_time:
                Skip images captured before this UTC datetime. Useful for
                scoping to post-PR #51 images (April 21 03:17 PDT onward).
            camera_filter:
                "cam0" / "cam1" / None. None labels both cameras.

        Returns:
            A summary dict with counts: attempted, succeeded, failed, skipped.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        obs_index = (
            ObservationIndex.from_jsonl(observations_path)
            if observations_path
            else ObservationIndex()
        )
        already_labeled = load_already_labeled(output_path)
        if already_labeled:
            logger.info(
                "Resuming: %d images already pre-labeled in %s",
                len(already_labeled),
                output_path,
            )

        summary = {
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped_already_labeled": 0,
            "skipped_too_old": 0,
            "skipped_no_match": 0,
            "total_elapsed_seconds": 0.0,
        }
        run_started = time.monotonic()

        for image_path in self._iter_images(image_dir, camera_filter):
            if limit is not None and summary["succeeded"] >= limit:
                logger.info("Reached limit of %d labelled images — stopping.", limit)
                break

            filename = image_path.name

            # Skip images already in pre_labels.jsonl (resume support).
            if filename in already_labeled:
                summary["skipped_already_labeled"] += 1
                continue

            # Time cutoff — skip images captured before min_capture_time.
            capture_ts = parse_capture_timestamp(filename)
            if min_capture_time is not None:
                if capture_ts is None:
                    # Unparseable filename — conservatively skip when a
                    # minimum time is enforced rather than labeling garbage.
                    summary["skipped_too_old"] += 1
                    continue
                if capture_ts < min_capture_time:
                    summary["skipped_too_old"] += 1
                    continue

            # Fetch audio hint and observation timestamp.
            audio_hint, audio_conf = obs_index.extract_audio_hint(filename)
            observation_record = obs_index.lookup(filename)
            obs_timestamp = None
            if observation_record is not None:
                ts_str = observation_record.get("timestamp")
                if ts_str:
                    try:
                        obs_timestamp = datetime.fromisoformat(
                            ts_str.replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass

            summary["attempted"] += 1
            try:
                pre_label = self.label_image(
                    image_path,
                    audio_hint=audio_hint,
                    audio_confidence=audio_conf,
                )
                # Enrich with observation timestamp (not visible to label_image).
                pre_label = pre_label.model_copy(
                    update={"observation_timestamp": obs_timestamp}
                )
                self._append_record(output_path, pre_label)
                summary["succeeded"] += 1
                summary["total_elapsed_seconds"] += pre_label.elapsed_seconds

                logger.info(
                    "[%d] %s -> %s (%.2f) %s | %.2fs",
                    summary["succeeded"],
                    filename,
                    pre_label.llm_response.species_code,
                    pre_label.llm_response.confidence,
                    "hint=" + audio_hint if audio_hint else "no-hint",
                    pre_label.elapsed_seconds,
                )
            except Exception as exc:  # noqa: BLE001 — we catch broadly on purpose
                summary["failed"] += 1
                logger.error(
                    "Failed to pre-label %s: %s: %s",
                    filename,
                    type(exc).__name__,
                    exc,
                )

            time.sleep(self.inter_request_delay)

        summary["wall_clock_seconds"] = time.monotonic() - run_started
        logger.info(
            "PreLabeler run complete | attempted=%d succeeded=%d failed=%d "
            "skipped_already=%d skipped_too_old=%d wall=%.1fs",
            summary["attempted"],
            summary["succeeded"],
            summary["failed"],
            summary["skipped_already_labeled"],
            summary["skipped_too_old"],
            summary["wall_clock_seconds"],
        )
        return summary

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _iter_images(
        image_dir: Path, camera_filter: Optional[str]
    ) -> Iterator[Path]:
        """Yield image paths in newest-first order, optionally filtered by camera.

        Newest-first because we prioritize recent data (post-PR #51 color fix)
        per the investigation doc. Users can reverse this by re-running with
        different min_capture_time.
        """
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        all_images = sorted(
            image_dir.glob("*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,  # newest first
        )
        for path in all_images:
            if camera_filter and camera_filter not in path.name:
                continue
            yield path

    @staticmethod
    def _append_record(output_path: Path, pre_label: PreLabel) -> None:
        """Append a single PreLabel record to the JSONL output file.

        Uses model_dump(mode='json') to ensure datetime and Path objects are
        serialised to strings. Each record is written atomically as one line.
        """
        record_json = pre_label.model_dump_json()
        with output_path.open("a", encoding="utf-8") as fh:
            fh.write(record_json + "\n")