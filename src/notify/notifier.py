"""
src/notify/notifier.py

Dispatches BirdObservation events to configured notification channels.

Supported channels (toggled via .env):
    - log:   Always active. Writes JSON to the log file (configs/paths.yaml).
    - print: Console output (useful for development / demo mode).
    - push:  Mobile push notification (future phase — requires a push service).
    - email: Email notification (future phase).

Design principle:
    The notifier does not know or care about classification. It receives a
    finished BirdObservation and delivers it. Adding a new channel means
    adding a new method here — no other module needs to change.

Phase 3 will implement log and print channels.
Phase 5+ will add push/email channels.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
    ) -> None:
        """
        Args:
            log_path: Path to the JSON log file where observations are appended.
            enable_print: Whether to print observations to stdout.
            enable_push: Whether to send push notifications (requires Phase 5 setup).
            enable_email: Whether to send email notifications (requires Phase 5 setup).
        """
        self.log_path = Path(log_path)
        self.enable_print = enable_print
        self.enable_push = enable_push
        self.enable_email = enable_email

    @classmethod
    def from_config(cls, notify_config_path: str, paths_config_path: str) -> Notifier:
        """
        Construct a Notifier from config YAMLs.

        Args:
            notify_config_path: Path to configs/notify.yaml.
            paths_config_path: Path to configs/paths.yaml (for log_path).

        Returns:
            Configured Notifier instance.
        """
        raise NotImplementedError("Implement in Phase 3.")

    def dispatch(self, observation: BirdObservation) -> None:
        """
        Send the observation to all active channels.

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

        Args:
            observation: BirdObservation to log.
        """
        raise NotImplementedError("Implement in Phase 3.")

    def _print(self, observation: BirdObservation) -> None:
        """
        Print a human-readable observation summary to stdout.

        Args:
            observation: BirdObservation to display.
        """
        raise NotImplementedError("Implement in Phase 3.")

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
