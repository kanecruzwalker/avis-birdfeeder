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


# ── _webhook ──────────────────────────────────────────────────────────────────


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
