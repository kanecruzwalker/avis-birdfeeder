"""Command-line entry point for the labeling-assistant review UI.

Run with:
    python -m tools.labeler.ui

By default binds to 127.0.0.1:8765 — localhost only. To make the UI
reachable from a phone over Tailscale (or other trusted private
network), pass `--host 0.0.0.0`. Never bind to a public interface
without further hardening; the AVIS_WEB_TOKEN is the only auth layer.

The CLI reads .env in the current working directory, then expects:
    AVIS_WEB_TOKEN=<at least 16 chars>

If the token is missing or too short, the server refuses to start
rather than silently accepting unauthenticated traffic.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn

from .auth import AuthConfigError, get_configured_token
from .server import (
    DEFAULT_IMAGES_DIR,
    DEFAULT_PRE_LABELS_PATH,
    DEFAULT_VERIFIED_LABELS_PATH,
    create_app,
)

logger = logging.getLogger(__name__)


# ── .env loading ──────────────────────────────────────────────────────────────


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from .env into os.environ.

    We don't pull in python-dotenv as a dependency for this. The format
    we accept is the simple subset already used by the rest of the
    project: lines of `KEY=VALUE`, ignore blank lines and `#` comments.
    Quoted values have surrounding quotes stripped. Existing environment
    variables take precedence — `.env` only fills gaps.
    """
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip matching surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            # Existing env wins
            if key not in os.environ:
                os.environ[key] = value


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.labeler.ui",
        description=(
            "Review UI for the Avis labeling assistant (Layer 2). Reads "
            "pre_labels.jsonl, writes verified_labels.jsonl, serves a "
            "responsive web UI on localhost."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Interface to bind. Default 127.0.0.1 (localhost only). "
            "Use 0.0.0.0 to expose over LAN/Tailscale for phone access."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on. Default 8765.",
    )
    parser.add_argument(
        "--pre-labels",
        type=Path,
        default=DEFAULT_PRE_LABELS_PATH,
        help=f"Path to pre_labels.jsonl. Default: {DEFAULT_PRE_LABELS_PATH}",
    )
    parser.add_argument(
        "--verified-labels",
        type=Path,
        default=DEFAULT_VERIFIED_LABELS_PATH,
        help=(
            f"Path to verified_labels.jsonl (created on first verify). "
            f"Default: {DEFAULT_VERIFIED_LABELS_PATH}"
        ),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help=f"Directory of capture PNGs. Default: {DEFAULT_IMAGES_DIR}",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file to load. Default: .env in cwd.",
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
        help="Enable auto-reload (dev only — not recommended for review sessions).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # 1. Load .env (silent if missing — not required for tests)
    _load_dotenv(args.env_file)

    # 2. Validate token presence before binding the socket
    try:
        token = get_configured_token()
    except AuthConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # 3. Show a banner so the operator can grab the URL quickly
    masked = token[:4] + "…" + token[-4:] if len(token) > 8 else "***"
    print()
    print("=" * 64)
    print("  Avis labeling-assistant review UI")
    print("=" * 64)
    print(f"  pre-labels:       {args.pre_labels}")
    print(f"  verified-labels:  {args.verified_labels}")
    print(f"  images-dir:       {args.images_dir}")
    print(f"  bind:             http://{args.host}:{args.port}")
    print(f"  token:            {masked}  (set AVIS_WEB_TOKEN)")
    if args.host == "127.0.0.1":
        print(f"  url:              http://localhost:{args.port}/?token={token}")
    else:
        print(
            f"  url:              http://<this-host>:{args.port}/?token={token}\n"
            f"                    (Tailscale/LAN mode — anyone with the URL "
            f"and token can access)"
        )
    print("=" * 64)
    print()

    # 4. Build the app. Construction validates that pre_labels.jsonl
    #    exists and parses; bail early if not.
    try:
        app = create_app(
            pre_labels_path=args.pre_labels,
            verified_labels_path=args.verified_labels,
            images_dir=args.images_dir,
            autoload=True,
        )
    except Exception as exc:  # noqa: BLE001 — surface any startup error cleanly
        print(f"ERROR: failed to build app: {exc}", file=sys.stderr)
        return 3

    # 5. Run uvicorn. We pass the app object directly rather than a
    #    "module:attr" string so reloader doesn't try to re-import this
    #    __main__ module (which would re-execute argv parsing).
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
