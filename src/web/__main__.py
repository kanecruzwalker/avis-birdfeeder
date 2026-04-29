"""Command-line entry point for the Avis web dashboard.

Run with::

    python -m src.web

By default binds to ``127.0.0.1:8000`` — localhost only. To make the
dashboard reachable from a phone or laptop over Tailscale, pass
``--host 0.0.0.0``. Never bind to a public interface without further
hardening; ``AVIS_WEB_TOKEN`` is the only auth layer.

The CLI reads ``.env`` in the current working directory, then expects::

    AVIS_WEB_TOKEN=<at least 16 chars>

If the token is missing or too short the server refuses to start
rather than silently accepting unauthenticated traffic.

The path to ``observations.jsonl`` is resolved in this order:

1. ``--observations-path`` if given on the command line
2. ``configs/paths.yaml`` -> ``logs.observations`` if the file exists
3. ``logs/observations.jsonl`` (the factory default)

This module runs independently of ``src.agent`` and ships as its
own systemd unit (``scripts/avis-web.service``). Stopping the
dashboard doesn't touch the agent, and vice versa.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from dotenv import load_dotenv

from .app import create_app
from .auth import AuthConfigError, get_configured_token
from .box_cache import BoxCache
from .stream_buffer import StreamBuffer

logger = logging.getLogger(__name__)


_CONFIGS_DIR = Path("configs")


def _maybe_build_analyst() -> Any | None:
    """Instantiate :class:`BirdAnalystAgent` if GEMINI_API_KEY is set.

    Returns ``None`` when no key is configured, when ``configs/`` is
    absent, or when import / construction raises (the LLM stack is
    optional — the dashboard must keep booting either way).
    The import is deferred so missing langchain/google-genai installs
    don't break ``python -m src.web --help``.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    if not _CONFIGS_DIR.exists():
        logger.warning("GEMINI_API_KEY set but %s missing — chat disabled.", _CONFIGS_DIR)
        return None
    try:
        from src.agent.bird_analyst_agent import BirdAnalystAgent
    except Exception as exc:  # noqa: BLE001 — any import error means LLM stack missing
        logger.warning("Could not import BirdAnalystAgent — chat disabled: %s", exc)
        return None
    try:
        return BirdAnalystAgent.from_config(_CONFIGS_DIR)
    except Exception as exc:  # noqa: BLE001 — config or LLM init failure shouldn't block boot
        logger.warning("BirdAnalystAgent init failed — chat disabled: %s", exc)
        return None


# ── observations.jsonl resolution ────────────────────────────────────────────

_DEFAULT_OBSERVATIONS_PATH = Path("logs/observations.jsonl")
_PATHS_YAML = Path("configs/paths.yaml")


def _resolve_observations_path(cli_path: Path | None) -> Path:
    """Pick the observations.jsonl location.

    Order: explicit ``--observations-path`` flag, then
    ``configs/paths.yaml``'s ``logs.observations`` key, then
    ``logs/observations.jsonl`` as a last resort. The file doesn't
    have to exist — the store handles the missing case (returns
    empty list, /api/status reports ``stale``).
    """
    if cli_path is not None:
        return cli_path

    if _PATHS_YAML.exists():
        try:
            data = yaml.safe_load(_PATHS_YAML.read_text(encoding="utf-8")) or {}
            configured = (data.get("logs") or {}).get("observations")
            if configured:
                return Path(configured)
        except (yaml.YAMLError, OSError) as exc:
            # Don't refuse to start if paths.yaml is broken — fall back
            # to the default and log loudly so the operator notices.
            logger.warning("could not read %s: %s", _PATHS_YAML, exc)

    return _DEFAULT_OBSERVATIONS_PATH


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.web",
        description=(
            "Avis web dashboard. Token-authenticated FastAPI app with "
            "live MJPEG, observation history, and chat proxy to the "
            "BirdAnalystAgent. Reads .env from the cwd for "
            "AVIS_WEB_TOKEN."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Interface to bind. Default 127.0.0.1 (localhost only). "
            "Use 0.0.0.0 to expose over LAN/Tailscale."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on. Default 8000.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file to load. Default: .env in cwd.",
    )
    parser.add_argument(
        "--observations-path",
        type=Path,
        default=None,
        help=(
            "Override path to observations.jsonl. Default: read from "
            "configs/paths.yaml, falling back to logs/observations.jsonl."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Uvicorn log level.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help=(
            "Enable auto-reload (dev only — not recommended on the Pi "
            "where it adds inotify overhead and conflicts with systemd)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # 1. Load .env (silent if missing — not required for tests).
    #    override=False matches the prior behavior: existing env wins.
    load_dotenv(args.env_file, override=False)

    # 2. Check the token before binding the socket — fail with a clear
    #    message at startup instead of returning 500 on the first
    #    request.
    try:
        token = get_configured_token()
    except AuthConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # 3. Pick the observations.jsonl location
    obs_path = _resolve_observations_path(args.observations_path)

    # 4. Show a startup banner so the operator can grab the URL.
    #    The token is NEVER printed in full — banner output goes to
    #    stdout (and on the Pi, into the systemd journal), where it
    #    can be screen-shared, copy-pasted, or shoulder-surfed. The
    #    operator who started the process already has the token from
    #    ``.env``; we only show a fingerprint here so they can confirm
    #    the right one was loaded. See the risk table in
    #    ``docs/investigations/web-dashboard-2026-04-28.md``
    #    ("Token leaks in chat history or logs").
    masked = token[:4] + "…" + token[-4:] if len(token) > 8 else "***"
    print()
    print("=" * 64)
    print("  Avis web dashboard")
    print("=" * 64)
    print(f"  bind:    http://{args.host}:{args.port}")
    print(f"  token:   {masked}  (set AVIS_WEB_TOKEN; full value not shown)")
    print(f"  obs:     {obs_path}  (exists={obs_path.exists()})")
    if args.host == "127.0.0.1":
        print(f"  url:     http://localhost:{args.port}/?token=<your AVIS_WEB_TOKEN>")
    else:
        print(
            f"  url:     http://<this-host>:{args.port}/?token=<your AVIS_WEB_TOKEN>\n"
            f"           (Tailscale/LAN mode — anyone with the URL "
            f"and token can access)"
        )
    print(
        f"  stream:  http://{args.host}:{args.port}/api/stream  (503 until a publisher is wired in)"
    )
    print(f"  health:  http://{args.host}:{args.port}/health  (no token required)")
    print("=" * 64)
    print()

    # 5. Build the app. Stream buffer + box cache are allocated
    #    unconditionally so the API contract is stable: routes 503
    #    cleanly when no in-process publisher exists. Production
    #    standalone-dashboard mode stays in that state until the
    #    cross-process bridge lands. The analyst is opt-in via
    #    GEMINI_API_KEY — when absent, /api/ask returns 503.
    stream_buffer = StreamBuffer()
    box_cache = BoxCache()
    analyst = _maybe_build_analyst()
    if analyst is None:
        print("  chat:    /api/ask returns 503  (set GEMINI_API_KEY to enable)")
    else:
        print("  chat:    /api/ask enabled  (BirdAnalystAgent loaded)")
    print()
    try:
        app = create_app(
            observations_path=obs_path,
            stream_buffer=stream_buffer,
            box_cache=box_cache,
            analyst=analyst,
        )
    except Exception as exc:  # noqa: BLE001 — surface any startup error cleanly
        print(f"ERROR: failed to build app: {exc}", file=sys.stderr)
        return 3

    # 6. Run uvicorn. Pass the app object directly so the reloader
    #    doesn't try to re-import this __main__ module (which would
    #    re-execute argv parsing).
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
