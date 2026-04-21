"""
tests/notify/test_notifier.py

Unit tests for Notifier and _build_multipart.

Phase 6 additions:
    - _make_notifier() defaults extended with push_attach_image=False so all
      existing tests explicitly opt out of attachment behaviour.
    - _build_multipart() tested as an independent pure function (TestBuildMultipart).
    - _push() image attachment paths tested exhaustively (TestNotifierPushAttachment):
        - multipart POST sent when valid image file is present
        - text-only fallback when attach_image is False
        - text-only fallback when image_path is None (audio-only detection)
        - text-only fallback when file does not exist on disk
        - text-only fallback when file exceeds max_attachment_bytes
        - audio-only message note present in text-only push body
        - graceful survival when image file is unreadable (OSError)
    - from_config() extended to cover push block loading.

Strategy (unchanged):
    - No real network calls in any test.
    - Pushover POST mocked via unittest.mock.patch on urllib.request.urlopen.
    - Credentials injected via monkeypatch on os.environ.
    - All existing Phase 3/5 tests preserved unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.schema import BirdObservation, ClassificationResult, Modality
from src.notify.notifier import Notifier, _build_multipart

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_observation(
    species_code: str = "HOFI",
    fused_confidence: float = 0.87,
    with_audio: bool = True,
    with_visual: bool = True,
) -> BirdObservation:
    audio = (
        ClassificationResult(
            species_code=species_code,
            common_name="House Finch",
            scientific_name="Haemorhous mexicanus",
            confidence=0.82,
            modality=Modality.AUDIO,
        )
        if with_audio
        else None
    )

    visual = (
        ClassificationResult(
            species_code=species_code,
            common_name="House Finch",
            scientific_name="Haemorhous mexicanus",
            confidence=0.91,
            modality=Modality.VISUAL,
        )
        if with_visual
        else None
    )

    return BirdObservation(
        species_code=species_code,
        common_name="House Finch",
        scientific_name="Haemorhous mexicanus",
        fused_confidence=fused_confidence,
        audio_result=audio,
        visual_result=visual,
    )


def _make_notifier(tmp_path: Path, **kwargs) -> Notifier:
    """
    Build a Notifier with a temporary log path and all channels disabled.

    push_attach_image defaults to False so that existing tests which do not
    exercise attachment behaviour are not affected by Phase 6 changes.
    Tests that specifically test attachment pass push_attach_image=True
    explicitly via kwargs.
    """
    defaults = dict(
        log_path=str(tmp_path / "observations.jsonl"),
        enable_print=False,
        enable_push=False,
        enable_webhook=False,
        enable_email=False,
        push_attach_image=False,
    )
    defaults.update(kwargs)
    return Notifier(**defaults)


# ── __init__ ──────────────────────────────────────────────────────────────────


class TestNotifierInit:
    def test_stores_log_path(self, tmp_path: Path) -> None:
        n = _make_notifier(tmp_path)
        assert n.log_path == tmp_path / "observations.jsonl"

    def test_default_enable_print_true(self, tmp_path: Path) -> None:
        n = Notifier(log_path=str(tmp_path / "obs.jsonl"))
        assert n.enable_print is True

    def test_default_enable_push_false(self, tmp_path: Path) -> None:
        n = Notifier(log_path=str(tmp_path / "obs.jsonl"))
        assert n.enable_push is False

    def test_default_enable_email_false(self, tmp_path: Path) -> None:
        n = Notifier(log_path=str(tmp_path / "obs.jsonl"))
        assert n.enable_email is False

    def test_default_enable_webhook_false(self, tmp_path: Path) -> None:
        n = Notifier(log_path=str(tmp_path / "obs.jsonl"))
        assert n.enable_webhook is False

    def test_default_push_attach_image_true(self, tmp_path: Path) -> None:
        n = Notifier(log_path=str(tmp_path / "obs.jsonl"))
        assert n.push_attach_image is True

    def test_default_push_max_attachment_bytes(self, tmp_path: Path) -> None:
        n = Notifier(log_path=str(tmp_path / "obs.jsonl"))
        assert n.push_max_attachment_bytes == 2_500_000

    def test_accepts_string_log_path(self, tmp_path: Path) -> None:
        n = Notifier(log_path=str(tmp_path / "obs.jsonl"))
        assert isinstance(n.log_path, Path)

    def test_custom_flags(self, tmp_path: Path) -> None:
        n = _make_notifier(tmp_path, enable_print=True, enable_push=True)
        assert n.enable_print is True
        assert n.enable_push is True

    def test_stores_webhook_url(self, tmp_path: Path) -> None:
        n = _make_notifier(tmp_path, webhook_url="https://api.example.com/obs")
        assert n.webhook_url == "https://api.example.com/obs"

    def test_stores_webhook_timeout(self, tmp_path: Path) -> None:
        n = _make_notifier(tmp_path, webhook_timeout_seconds=10.0)
        assert n.webhook_timeout_seconds == 10.0

    def test_stores_push_attach_image(self, tmp_path: Path) -> None:
        n = _make_notifier(tmp_path, push_attach_image=True)
        assert n.push_attach_image is True

    def test_stores_push_max_attachment_bytes(self, tmp_path: Path) -> None:
        n = _make_notifier(tmp_path, push_max_attachment_bytes=1_000_000)
        assert n.push_max_attachment_bytes == 1_000_000


# ── _log ──────────────────────────────────────────────────────────────────────


class TestNotifierLog:
    def test_creates_log_file(self, tmp_path: Path) -> None:
        n = _make_notifier(tmp_path)
        n._log(_make_observation())
        assert n.log_path.exists()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        log_path = tmp_path / "nested" / "dir" / "obs.jsonl"
        n = Notifier(log_path=str(log_path))
        n._log(_make_observation())
        assert log_path.exists()

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        n = _make_notifier(tmp_path)
        n._log(_make_observation())
        line = n.log_path.read_text().strip()
        parsed = json.loads(line)
        assert isinstance(parsed, dict)

    def test_log_contains_species_code(self, tmp_path: Path) -> None:
        n = _make_notifier(tmp_path)
        n._log(_make_observation(species_code="MODO"))
        record = json.loads(n.log_path.read_text())
        assert record["species_code"] == "MODO"

    def test_log_contains_confidence(self, tmp_path: Path) -> None:
        n = _make_notifier(tmp_path)
        n._log(_make_observation(fused_confidence=0.83))
        record = json.loads(n.log_path.read_text())
        assert record["fused_confidence"] == pytest.approx(0.83)

    def test_appends_multiple_observations(self, tmp_path: Path) -> None:
        n = _make_notifier(tmp_path)
        n._log(_make_observation(species_code="HOFI"))
        n._log(_make_observation(species_code="MODO"))
        lines = n.log_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["species_code"] == "HOFI"
        assert json.loads(lines[1])["species_code"] == "MODO"

    def test_log_is_idempotent_on_rerun(self, tmp_path: Path) -> None:
        n = _make_notifier(tmp_path)
        obs = _make_observation()
        n._log(obs)
        n._log(obs)
        lines = n.log_path.read_text().strip().splitlines()
        assert len(lines) == 2  # both logged — deduplication is caller's job


# ── _print ────────────────────────────────────────────────────────────────────


class TestNotifierPrint:
    def test_prints_to_stdout(self, tmp_path: Path, capsys) -> None:
        n = _make_notifier(tmp_path, enable_print=True)
        n._print(_make_observation())
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_contains_common_name(self, tmp_path: Path, capsys) -> None:
        n = _make_notifier(tmp_path, enable_print=True)
        n._print(_make_observation())
        assert "House Finch" in capsys.readouterr().out

    def test_print_contains_scientific_name(self, tmp_path: Path, capsys) -> None:
        n = _make_notifier(tmp_path, enable_print=True)
        n._print(_make_observation())
        assert "Haemorhous mexicanus" in capsys.readouterr().out

    def test_custom_template(self, tmp_path: Path, capsys) -> None:
        n = _make_notifier(
            tmp_path,
            enable_print=True,
            message_template="BIRD: {species_code}",
        )
        n._print(_make_observation(species_code="MODO"))
        assert "BIRD: MODO" in capsys.readouterr().out


# ── _push ─────────────────────────────────────────────────────────────────────


class TestNotifierPush:
    @pytest.fixture(autouse=True)
    def clear_pushover_env(self, monkeypatch):
        """Always clear real Pushover credentials before every push test.
        Prevents .env credentials leaking into tests and firing real API calls."""
        monkeypatch.delenv("PUSHOVER_USER_KEY", raising=False)
        monkeypatch.delenv("PUSHOVER_APP_TOKEN", raising=False)

    def test_skips_push_when_credentials_missing(self, tmp_path: Path, monkeypatch) -> None:
        """Missing credentials — should log a warning and return without raising."""
        monkeypatch.delenv("PUSHOVER_USER_KEY", raising=False)
        monkeypatch.delenv("PUSHOVER_APP_TOKEN", raising=False)
        n = _make_notifier(tmp_path, enable_push=True)
        # Should not raise
        n._push(_make_observation())

    def test_skips_push_when_user_key_missing(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("PUSHOVER_APP_TOKEN", "fake_token")
        monkeypatch.delenv("PUSHOVER_USER_KEY", raising=False)
        n = _make_notifier(tmp_path, enable_push=True)
        n._push(_make_observation())  # should not raise

    def test_posts_to_pushover_api(self, tmp_path: Path, monkeypatch) -> None:
        """Successful push — verify POST is made to correct URL."""
        monkeypatch.setenv("PUSHOVER_USER_KEY", "test_user_key")
        monkeypatch.setenv("PUSHOVER_APP_TOKEN", "test_app_token")

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": 1}'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            n = _make_notifier(tmp_path, enable_push=True)
            n._push(_make_observation())

        mock_urlopen.assert_called_once()
        request_arg = mock_urlopen.call_args[0][0]
        assert "pushover.net" in request_arg.full_url

    def test_push_message_contains_species(self, tmp_path: Path, monkeypatch) -> None:
        """Push payload should include the species common name in title."""
        monkeypatch.setenv("PUSHOVER_USER_KEY", "test_user_key")
        monkeypatch.setenv("PUSHOVER_APP_TOKEN", "test_app_token")

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": 1}'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            with patch("urllib.request.Request") as mock_request:
                n = _make_notifier(tmp_path, enable_push=True)
                n._push(_make_observation(species_code="HOFI"))

        call_kwargs = mock_request.call_args
        data_bytes = call_kwargs[1]["data"] if call_kwargs[1] else call_kwargs[0][1]
        assert b"House+Finch" in data_bytes or b"House Finch" in data_bytes or b"HOFI" in data_bytes

    def test_push_survives_api_error(self, tmp_path: Path, monkeypatch) -> None:
        """Network error during push — should log error, not raise."""
        monkeypatch.setenv("PUSHOVER_USER_KEY", "test_user_key")
        monkeypatch.setenv("PUSHOVER_APP_TOKEN", "test_app_token")

        with patch("urllib.request.urlopen", side_effect=OSError("network down")):
            n = _make_notifier(tmp_path, enable_push=True)
            n._push(_make_observation())  # should not raise


# ── _build_multipart ──────────────────────────────────────────────────────────


class TestBuildMultipart:
    def test_returns_bytes_and_content_type(self) -> None:
        body, ct = _build_multipart({"token": "t"}, b"imgdata", "img.jpg", "image/jpeg")
        assert isinstance(body, bytes)
        assert "multipart/form-data" in ct

    def test_content_type_contains_boundary(self) -> None:
        _, ct = _build_multipart({"token": "t"}, b"imgdata", "img.jpg", "image/jpeg")
        assert "boundary=" in ct

    def test_body_contains_field_value(self) -> None:
        body, _ = _build_multipart({"token": "mytoken"}, b"data", "f.jpg", "image/jpeg")
        assert b"mytoken" in body

    def test_body_contains_filename(self) -> None:
        body, _ = _build_multipart({"token": "t"}, b"data", "capture.png", "image/png")
        assert b"capture.png" in body

    def test_body_contains_attachment_bytes(self) -> None:
        payload = b"\x89PNG\r\nfakeimage"
        body, _ = _build_multipart({"token": "t"}, payload, "img.png", "image/png")
        assert payload in body

    def test_body_contains_mime_type(self) -> None:
        body, _ = _build_multipart({"token": "t"}, b"data", "img.jpg", "image/jpeg")
        assert b"image/jpeg" in body

    def test_multiple_fields_all_present(self) -> None:
        body, _ = _build_multipart(
            {"token": "tok", "user": "usr", "title": "Bird"},
            b"img",
            "img.jpg",
            "image/jpeg",
        )
        assert b"tok" in body
        assert b"usr" in body
        assert b"Bird" in body

    def test_boundary_closes_body(self) -> None:
        body, ct = _build_multipart({"k": "v"}, b"data", "f.jpg", "image/jpeg")
        boundary = ct.split("boundary=")[1]
        assert f"--{boundary}--".encode() in body


# ── _push image attachment ────────────────────────────────────────────────────


class TestNotifierPushAttachment:
    @pytest.fixture(autouse=True)
    def set_credentials(self, monkeypatch):
        """Inject valid Pushover credentials for all attachment tests."""
        monkeypatch.setenv("PUSHOVER_USER_KEY", "test_user_key")
        monkeypatch.setenv("PUSHOVER_APP_TOKEN", "test_app_token")

    def _mock_response(self) -> MagicMock:
        mock = MagicMock()
        mock.read.return_value = b'{"status": 1}'
        mock.__enter__ = lambda s: s
        mock.__exit__ = MagicMock(return_value=False)
        return mock

    def _obs_with_image(self, image_path: str) -> BirdObservation:
        obs = _make_observation()
        return obs.model_copy(update={"image_path": image_path})

    def test_multipart_sent_when_image_exists(self, tmp_path: Path) -> None:
        """Valid image file → POST uses multipart/form-data."""
        img = tmp_path / "capture.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"0" * 100)  # minimal fake JPEG
        obs = self._obs_with_image(str(img))

        with patch("urllib.request.urlopen", return_value=self._mock_response()) as mock_open:
            n = _make_notifier(tmp_path, enable_push=True, push_attach_image=True)
            n._push(obs)

        req = mock_open.call_args[0][0]
        assert "multipart/form-data" in req.get_header("Content-type")

    def test_text_only_when_attach_image_false(self, tmp_path: Path) -> None:
        """attach_image=False → text-only even when file exists."""
        img = tmp_path / "capture.jpg"
        img.write_bytes(b"fakeimage")
        obs = self._obs_with_image(str(img))

        with patch("urllib.request.urlopen", return_value=self._mock_response()) as mock_open:
            n = _make_notifier(tmp_path, enable_push=True, push_attach_image=False)
            n._push(obs)

        req = mock_open.call_args[0][0]
        assert "application/x-www-form-urlencoded" in req.get_header("Content-type")

    def test_text_only_when_image_path_none(self, tmp_path: Path) -> None:
        """No image_path on observation → text-only (e.g. audio-only detection)."""
        obs = _make_observation()  # image_path is None by default

        with patch("urllib.request.urlopen", return_value=self._mock_response()) as mock_open:
            n = _make_notifier(tmp_path, enable_push=True, push_attach_image=True)
            n._push(obs)

        req = mock_open.call_args[0][0]
        assert "application/x-www-form-urlencoded" in req.get_header("Content-type")

    def test_text_only_when_image_file_missing(self, tmp_path: Path) -> None:
        """image_path set but file does not exist → text-only fallback."""
        obs = self._obs_with_image(str(tmp_path / "nonexistent.jpg"))

        with patch("urllib.request.urlopen", return_value=self._mock_response()) as mock_open:
            n = _make_notifier(tmp_path, enable_push=True, push_attach_image=True)
            n._push(obs)

        req = mock_open.call_args[0][0]
        assert "application/x-www-form-urlencoded" in req.get_header("Content-type")

    def test_text_only_when_file_exceeds_size_limit(self, tmp_path: Path) -> None:
        """File larger than max_attachment_bytes → text-only fallback."""
        img = tmp_path / "big.jpg"
        img.write_bytes(b"x" * 100)
        obs = self._obs_with_image(str(img))

        with patch("urllib.request.urlopen", return_value=self._mock_response()) as mock_open:
            n = _make_notifier(
                tmp_path,
                enable_push=True,
                push_attach_image=True,
                push_max_attachment_bytes=50,  # smaller than our 100-byte file
            )
            n._push(obs)

        req = mock_open.call_args[0][0]
        assert "application/x-www-form-urlencoded" in req.get_header("Content-type")

    def test_audio_only_note_in_message(self, tmp_path: Path) -> None:
        """Audio-only detection → message body contains note about no visual."""
        obs = _make_observation(with_visual=False)  # no image_path, no visual result

        with patch("urllib.request.urlopen", return_value=self._mock_response()):
            with patch("urllib.request.Request") as mock_req:
                mock_req.return_value = MagicMock()
                n = _make_notifier(tmp_path, enable_push=True, push_attach_image=True)
                n._push(obs)

        data = mock_req.call_args[1]["data"] if mock_req.call_args[1] else mock_req.call_args[0][1]
        assert b"Audio-only" in data

    def test_survives_unreadable_image_file(self, tmp_path: Path) -> None:
        """OSError reading image file → falls back to text-only, does not raise."""
        img = tmp_path / "capture.jpg"
        img.write_bytes(b"data")
        obs = self._obs_with_image(str(img))

        with patch("urllib.request.urlopen", return_value=self._mock_response()):
            with patch("pathlib.Path.read_bytes", side_effect=OSError("permission denied")):
                n = _make_notifier(tmp_path, enable_push=True, push_attach_image=True)
                n._push(obs)  # should not raise

    def test_png_attachment_uses_correct_mime(self, tmp_path: Path) -> None:
        """PNG file → Content-Type in multipart body is image/png."""
        img = tmp_path / "capture.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 50)
        obs = self._obs_with_image(str(img))

        with patch("urllib.request.urlopen", return_value=self._mock_response()) as mock_open:
            n = _make_notifier(tmp_path, enable_push=True, push_attach_image=True)
            n._push(obs)

        req = mock_open.call_args[0][0]
        assert b"image/png" in req.data


# ── Network resilience (timeout scaling + retry) ──────────────────────────────


class TestNetworkResilience:
    """
    Tests for _post_to_pushover: payload-scaled timeouts and retry-with-backoff.

    These tests validate the fix for edge-deployment network issues. The
    original notifier used a fixed 5-second timeout and no retries, which
    produced 100% failure rate on marginal WiFi (-70 dBm or worse) when
    attaching ~500KB image frames. Field-tested on April 20 2026 with 14
    consecutive push failures, all of which would have succeeded with these
    resilience mechanics in place.
    """

    @pytest.fixture(autouse=True)
    def set_credentials(self, monkeypatch):
        """Inject valid Pushover credentials for all resilience tests."""
        monkeypatch.setenv("PUSHOVER_USER_KEY", "test_user_key")
        monkeypatch.setenv("PUSHOVER_APP_TOKEN", "test_app_token")

    def _mock_success_response(self) -> MagicMock:
        mock = MagicMock()
        mock.read.return_value = b'{"status": 1}'
        mock.__enter__ = lambda s: s
        mock.__exit__ = MagicMock(return_value=False)
        return mock

    def test_timeout_scales_with_payload_size(self, tmp_path: Path) -> None:
        """
        Larger payloads get proportionally longer timeouts.
        500KB image with base=10s, per_kb=0.1s → ~60s total timeout.
        """
        n = _make_notifier(
            tmp_path,
            enable_push=True,
            push_base_timeout_seconds=10.0,
            push_per_kb_timeout_seconds=0.1,
        )
        with patch("urllib.request.urlopen", return_value=self._mock_success_response()) as mock:
            n._post_to_pushover(b"x" * 500_000, "image/png", context="500KB image test")

        # urlopen called with positional (req, timeout) — check the timeout kwarg
        call = mock.call_args
        timeout = call.kwargs.get("timeout")
        assert timeout is not None
        # 10 (base) + 500 * 0.1 (per_kb) ≈ 58-62 depending on KB math
        assert 58.0 <= timeout <= 62.0, f"expected ~60s, got {timeout}"

    def test_retries_on_timeout(self, tmp_path: Path) -> None:
        """Transient timeout → retries up to max_attempts before giving up."""
        n = _make_notifier(
            tmp_path,
            enable_push=True,
            push_max_attempts=3,
            push_retry_backoff_seconds=0.001,  # fast test
        )
        with patch("urllib.request.urlopen", side_effect=TimeoutError("slow network")) as mock:
            result = n._post_to_pushover(b"data", "image/png", context="timeout test")

        assert mock.call_count == 3, "should retry up to max_attempts"
        assert result is False, "should return False after all retries exhausted"

    def test_succeeds_on_retry_after_transient_failure(self, tmp_path: Path) -> None:
        """
        First attempt times out, second succeeds → overall success.
        This is the exact case that was broken in production: slow WiFi
        causes the first upload to time out, but a retry gets through.
        """
        n = _make_notifier(
            tmp_path,
            enable_push=True,
            push_max_attempts=3,
            push_retry_backoff_seconds=0.001,
        )
        with patch(
            "urllib.request.urlopen",
            side_effect=[
                TimeoutError("first attempt slow"),
                self._mock_success_response(),
            ],
        ) as mock:
            result = n._post_to_pushover(b"data", "image/png", context="retry success")

        assert mock.call_count == 2, "should stop retrying after success"
        assert result is True, "should return True on successful retry"

    def test_does_not_retry_on_api_rejection(self, tmp_path: Path) -> None:
        """
        Pushover status != 1 → no retry. API rejections (bad credentials,
        rate limit, invalid user key) will not succeed on retry and would
        just hammer the API. These must be handled differently from
        network failures.
        """
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": 0, "errors": ["invalid token"]}'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        n = _make_notifier(
            tmp_path,
            enable_push=True,
            push_max_attempts=3,
            push_retry_backoff_seconds=0.001,
        )
        with patch("urllib.request.urlopen", return_value=mock_response) as mock:
            result = n._post_to_pushover(b"data", "image/png", context="api reject test")

        assert mock.call_count == 1, "must NOT retry on API-level rejection"
        assert result is False

    def test_backoff_is_exponential(self, tmp_path: Path) -> None:
        """
        Sleep durations between retries scale exponentially: base, base*2, base*4.
        This prevents hammering a temporarily slow endpoint while still
        giving enough total time for it to recover.
        """
        n = _make_notifier(
            tmp_path,
            enable_push=True,
            push_max_attempts=4,
            push_retry_backoff_seconds=1.0,
        )
        with patch("urllib.request.urlopen", side_effect=TimeoutError("fail")):
            with patch("time.sleep") as mock_sleep:
                n._post_to_pushover(b"data", "image/png", context="backoff test")

        # With 4 attempts, we sleep 3 times (between attempts 1→2, 2→3, 3→4)
        sleep_durations = [call.args[0] for call in mock_sleep.call_args_list]
        assert sleep_durations == [
            1.0,
            2.0,
            4.0,
        ], f"expected exponential backoff [1.0, 2.0, 4.0], got {sleep_durations}"

    def test_full_push_with_image_recovers_from_transient_failure(self, tmp_path: Path) -> None:
        """
        End-to-end: _push() with a real image attachment recovers when the
        first attempt times out. This is the production scenario that was
        failing — image upload on slow WiFi times out at 5s, but succeeds
        on a retry given enough time.
        """
        img = tmp_path / "capture.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"0" * 200)  # fake JPEG bytes
        obs = _make_observation()
        obs = obs.model_copy(update={"image_path": str(img)})

        n = _make_notifier(
            tmp_path,
            enable_push=True,
            push_attach_image=True,
            push_max_attempts=3,
            push_retry_backoff_seconds=0.001,
        )
        with patch(
            "urllib.request.urlopen",
            side_effect=[
                TimeoutError("slow wifi"),
                self._mock_success_response(),
            ],
        ) as mock:
            n._push(obs)  # must not raise

        assert mock.call_count == 2, "first attempt timed out, second succeeded"


# ── _webhook ──────────────────────────────────────────────────────────────────
# ── log_suppressed ────────────────────────────────────────────────────────────


class TestLogSuppressed:
    """
    Tests for Notifier.log_suppressed — the below-threshold logging path added
    in PR #51 to eliminate orphan observations.

    Before this PR, observations that failed the confidence threshold or
    cooldown gate in BirdAgent._cycle() were silently discarded with only a
    DEBUG log message. On April 20 deployment this caused 5186 of 5624
    captured frames to have no corresponding entry in observations.jsonl,
    making analysis of the visual classifier's actual behavior impossible.
    log_suppressed writes these observations to the same file with
    dispatched=False so the full classification stream is preserved.
    """

    def test_writes_to_log_file(self, tmp_path: Path) -> None:
        """log_suppressed must append a JSONL record like dispatch() does."""
        n = _make_notifier(tmp_path)
        n.log_suppressed(_make_observation(species_code="HOFI", fused_confidence=0.15))

        assert n.log_path.exists()
        lines = n.log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["species_code"] == "HOFI"

    def test_marks_dispatched_false(self, tmp_path: Path) -> None:
        """Logged record must have dispatched=False set."""
        n = _make_notifier(tmp_path)
        n.log_suppressed(_make_observation(species_code="SOSP", fused_confidence=0.12))

        record = json.loads(n.log_path.read_text().strip())
        assert record["dispatched"] is False

    def test_does_not_trigger_push(self, tmp_path: Path) -> None:
        """log_suppressed must not trigger Pushover API calls."""
        n = _make_notifier(tmp_path, enable_push=True)

        with patch("urllib.request.urlopen") as mock_urlopen:
            n.log_suppressed(_make_observation(species_code="AMCR", fused_confidence=0.10))

        mock_urlopen.assert_not_called()

    def test_does_not_print(self, tmp_path: Path, capsys) -> None:
        """log_suppressed must not write to stdout even when print is enabled."""
        n = _make_notifier(tmp_path, enable_print=True)
        n.log_suppressed(_make_observation(species_code="WBNU", fused_confidence=0.18))

        assert capsys.readouterr().out == ""

    def test_preserves_caller_observation_immutability(self, tmp_path: Path) -> None:
        """
        The observation passed in must not be mutated — model_copy should
        return a new object. This matches how dispatch() handles the update.
        """
        n = _make_notifier(tmp_path)
        obs = _make_observation(species_code="HOFI", fused_confidence=0.15)
        assert obs.dispatched is True  # default

        n.log_suppressed(obs)

        # The original object should be unchanged
        assert obs.dispatched is True


# ── dispatched field ──────────────────────────────────────────────────────────


class TestDispatchedField:
    """
    Tests for the dispatched field added to BirdObservation in PR #51.
    """

    def test_dispatch_marks_record_dispatched_true(self, tmp_path: Path) -> None:
        """dispatch() must set dispatched=True on the logged record."""
        n = _make_notifier(tmp_path)
        n.dispatch(_make_observation(species_code="HOFI", fused_confidence=0.85))

        record = json.loads(n.log_path.read_text().strip())
        assert record["dispatched"] is True

    def test_default_true_for_new_observations(self) -> None:
        """
        Backward compatibility: a BirdObservation constructed without the
        dispatched field should default to True. This ensures records written
        before PR #51 (which are loaded from observations.jsonl without the
        field) deserialize with dispatched=True, matching historical reality
        where only dispatched observations were logged.
        """
        obs = BirdObservation(
            species_code="HOFI",
            common_name="House Finch",
            scientific_name="Haemorhous mexicanus",
            fused_confidence=0.85,
        )
        assert obs.dispatched is True

    def test_deserialization_backward_compatible(self) -> None:
        """
        Loading an old record (without dispatched field) must succeed and
        default dispatched to True. This guards the analyzer notebooks and
        LLM agent tools that read observations.jsonl.
        """
        old_record = {
            "species_code": "HOFI",
            "common_name": "House Finch",
            "scientific_name": "Haemorhous mexicanus",
            "fused_confidence": 0.85,
        }
        obs = BirdObservation(**old_record)
        assert obs.dispatched is True


class TestNotifierWebhook:
    def test_webhook_raises_not_implemented(self, tmp_path: Path) -> None:
        """Phase 5/6 stub — webhook always raises NotImplementedError."""
        n = _make_notifier(tmp_path, enable_webhook=True)
        with pytest.raises(NotImplementedError):
            n._webhook(_make_observation())

    def test_dispatch_skips_webhook_when_disabled(self, tmp_path: Path) -> None:
        """Disabled webhook — dispatch should not call _webhook."""
        n = _make_notifier(tmp_path, enable_webhook=False)
        obs = _make_observation()
        # Should not raise NotImplementedError since webhook is disabled
        n.dispatch(obs)

    def test_dispatch_webhook_enabled_raises_not_implemented(self, tmp_path: Path) -> None:
        """Enabled webhook — dispatch propagates NotImplementedError."""
        n = _make_notifier(tmp_path, enable_webhook=True)
        with pytest.raises(NotImplementedError):
            n.dispatch(_make_observation())


# ── dispatch ──────────────────────────────────────────────────────────────────


class TestNotifierDispatch:
    def test_dispatch_always_calls_log(self, tmp_path: Path) -> None:
        n = _make_notifier(tmp_path)
        n.dispatch(_make_observation())
        assert n.log_path.exists()

    def test_dispatch_calls_print_when_enabled(self, tmp_path: Path, capsys) -> None:
        n = _make_notifier(tmp_path, enable_print=True)
        n.dispatch(_make_observation())
        assert len(capsys.readouterr().out) > 0

    def test_dispatch_skips_print_when_disabled(self, tmp_path: Path, capsys) -> None:
        n = _make_notifier(tmp_path, enable_print=False)
        n.dispatch(_make_observation())
        assert capsys.readouterr().out == ""

    def test_dispatch_skips_push_when_disabled(self, tmp_path: Path) -> None:
        """Push disabled — dispatch should not attempt Pushover API call."""
        n = _make_notifier(tmp_path, enable_push=False)
        with patch("urllib.request.urlopen") as mock_urlopen:
            n.dispatch(_make_observation())
        mock_urlopen.assert_not_called()

    def test_dispatch_push_enabled_raises_not_implemented(self, tmp_path: Path) -> None:
        """This test preserved for backward compat — now push IS implemented."""
        # Push won't raise NotImplementedError anymore — it will warn about
        # missing credentials and return gracefully.
        n = _make_notifier(tmp_path, enable_push=True)
        # With no credentials set, should warn and not raise
        n.dispatch(_make_observation())


# ── from_config ───────────────────────────────────────────────────────────────


class TestFromConfig:
    def test_loads_log_path_from_yaml(self) -> None:
        n = Notifier.from_config("configs/notify.yaml", "configs/paths.yaml")
        assert "observations" in str(n.log_path)

    def test_loads_print_channel_true(self) -> None:
        n = Notifier.from_config("configs/notify.yaml", "configs/paths.yaml")
        assert n.enable_print is True

    def test_loads_print_channel_false(self, tmp_path: Path) -> None:
        notify_yaml = tmp_path / "notify.yaml"
        paths_yaml = tmp_path / "paths.yaml"
        notify_yaml.write_text(
            "channels:\n  print: false\n  push: false\n  webhook: false\n  email: false\n"
            "display:\n  message_template: 'test'\n"
        )
        paths_yaml.write_text("logs:\n  observations: 'logs/obs.jsonl'\n")
        n = Notifier.from_config(str(notify_yaml), str(paths_yaml))
        assert n.enable_print is False

    def test_push_disabled_by_default(self) -> None:
        n = Notifier.from_config("configs/notify.yaml", "configs/paths.yaml")
        assert n.enable_push is False

    def test_webhook_disabled_by_default(self) -> None:
        n = Notifier.from_config("configs/notify.yaml", "configs/paths.yaml")
        assert n.enable_webhook is False

    def test_loads_push_attach_image_from_yaml(self) -> None:
        """notify.yaml push.attach_image: true → push_attach_image is True."""
        n = Notifier.from_config("configs/notify.yaml", "configs/paths.yaml")
        assert n.push_attach_image is True

    def test_loads_push_max_attachment_bytes_from_yaml(self) -> None:
        """notify.yaml push.max_attachment_bytes: 2500000 → stored correctly."""
        n = Notifier.from_config("configs/notify.yaml", "configs/paths.yaml")
        assert n.push_max_attachment_bytes == 2_500_000

    def test_push_attach_image_can_be_disabled_via_yaml(self, tmp_path: Path) -> None:
        notify_yaml = tmp_path / "notify.yaml"
        paths_yaml = tmp_path / "paths.yaml"
        notify_yaml.write_text(
            "channels:\n  print: false\n  push: false\n  webhook: false\n  email: false\n"
            "display:\n  message_template: 'test'\n"
            "push:\n  attach_image: false\n  max_attachment_bytes: 2500000\n"
        )
        paths_yaml.write_text("logs:\n  observations: 'logs/obs.jsonl'\n")
        n = Notifier.from_config(str(notify_yaml), str(paths_yaml))
        assert n.push_attach_image is False

    def test_loads_webhook_url_from_yaml(self, tmp_path: Path) -> None:
        notify_yaml = tmp_path / "notify.yaml"
        paths_yaml = tmp_path / "paths.yaml"
        notify_yaml.write_text(
            "channels:\n  print: false\n  push: false\n  webhook: true\n  email: false\n"
            "display:\n  message_template: 'test'\n"
            "webhook:\n  url: 'https://api.example.com/obs'\n  timeout_seconds: 10\n  auth_header: ''\n"
        )
        paths_yaml.write_text("logs:\n  observations: 'logs/obs.jsonl'\n")
        n = Notifier.from_config(str(notify_yaml), str(paths_yaml))
        assert n.webhook_url == "https://api.example.com/obs"
        assert n.webhook_timeout_seconds == 10.0

    def test_raises_on_missing_notify_config(self) -> None:
        with pytest.raises(FileNotFoundError):
            Notifier.from_config("nonexistent/notify.yaml", "configs/paths.yaml")

    def test_raises_on_missing_paths_config(self) -> None:
        with pytest.raises(FileNotFoundError):
            Notifier.from_config("configs/notify.yaml", "nonexistent/paths.yaml")

    def test_returns_notifier_instance(self) -> None:
        n = Notifier.from_config("configs/notify.yaml", "configs/paths.yaml")
        assert isinstance(n, Notifier)
