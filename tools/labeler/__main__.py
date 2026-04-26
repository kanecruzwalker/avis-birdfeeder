"""Command-line entry point for the Avis labeling assistant pre-labeler.

Usage:
    python -m tools.labeler --help
    python -m tools.labeler --limit 10                    # smoke test on 10 images
    python -m tools.labeler --limit 500                   # realistic batch
    python -m tools.labeler --limit 500 --camera cam0     # only primary camera
    python -m tools.labeler --since 2026-04-21T10:17:00Z  # post PR #51 color fix

Environment variables:
    GOOGLE_API_KEY or GEMINI_API_KEY — Gemini API key (GOOGLE_API_KEY wins)

Default paths are set to the standard repo layout. Override with --image-dir,
--observations, and --output flags if running from a non-standard location.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from .pre_labeler import DEFAULT_MODEL_NAME, PreLabeler

# ── Defaults for Avis repo layout ─────────────────────────────────────────────

DEFAULT_IMAGE_DIR = Path("data/captures/images")
DEFAULT_OBSERVATIONS_PATH = Path("logs/observations.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/labels/pre_labels.jsonl")

# Anchor for "post-PR #51 color fix" — BGR/RGB swap was fixed here, so earlier
# images have inverted colors and should NOT be used as training data.
# UTC equivalent of April 21 2026 03:17 PDT.
PR51_MERGE_TIME_UTC = datetime(2026, 4, 21, 10, 17, 0, tzinfo=UTC)


# ── Argument parsing ──────────────────────────────────────────────────────────


def _parse_iso_datetime(s: str) -> datetime:
    """Parse an ISO-8601 datetime from the command line.

    Accepts both "2026-04-21T10:17:00Z" and "2026-04-21T10:17:00+00:00".
    """
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid ISO-8601 datetime: {s!r}. "
            f"Expected format like '2026-04-21T10:17:00Z'."
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools.labeler",
        description=(
            "Pre-label Avis capture images with Gemini 2.5 Flash. "
            "Produces suggested species labels for human review."
        ),
        epilog=(
            "Example runs:\n"
            "  # Smoke test on 10 recent images\n"
            "  python -m tools.labeler --limit 10\n"
            "\n"
            "  # Realistic batch for a labeling session\n"
            "  python -m tools.labeler --limit 500 --since 2026-04-21T10:17:00Z\n"
            "\n"
            "  # Just primary camera, latest 200 images\n"
            "  python -m tools.labeler --limit 200 --camera cam0\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Paths
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Directory with capture PNGs (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=DEFAULT_OBSERVATIONS_PATH,
        help=(
            f"Path to observations.jsonl for audio hints "
            f"(default: {DEFAULT_OBSERVATIONS_PATH}). "
            "Use --no-observations to skip audio hints entirely."
        ),
    )
    parser.add_argument(
        "--no-observations",
        action="store_true",
        help="Skip audio hint lookup — pre-label images with no context.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to append PreLabel records (default: {DEFAULT_OUTPUT_PATH})",
    )

    # Scoping
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of NEW images to label this run. "
            "Already-labeled images are always skipped. Default: no limit."
        ),
    )
    parser.add_argument(
        "--since",
        type=_parse_iso_datetime,
        default=None,
        help=(
            "Only label images captured after this UTC datetime (ISO-8601). "
            "Example: '2026-04-21T10:17:00Z'. "
            "Use --post-pr51 as a shortcut for the known-good cutoff."
        ),
    )
    parser.add_argument(
        "--post-pr51",
        action="store_true",
        help=(
            "Shortcut for --since 2026-04-21T10:17:00Z. "
            "PR #51 fixed the BGR/RGB color swap — earlier images have "
            "inverted colors and are unsuitable for training."
        ),
    )
    parser.add_argument(
        "--camera",
        choices=["cam0", "cam1"],
        default=None,
        help="Restrict to a single camera (default: both).",
    )

    # Model
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Gemini model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature. Low = consistent classification. Default: 0.1",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging (shows every Gemini call detail).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only log warnings and errors.",
    )

    return parser


# ── Cost estimation (pre-flight preview) ──────────────────────────────────────


def _estimate_cost(limit: int) -> float:
    """Rough upper bound on Gemini cost for a given image count.

    Based on April 2026 pricing: $0.30/M input, $2.50/M output.
    Per image: ~258 tokens image + ~2000 tokens system prompt (cached after
    first call) + ~200 tokens user message = ~2500 input, ~100 output.
    Ignoring caching (conservative), that's roughly $0.001 per image.
    """
    return limit * 0.001


# ── Entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns shell exit code."""
    load_dotenv()  # pick up GEMINI_API_KEY from .env

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Logging setup
    if args.verbose and args.quiet:
        parser.error("Pass either --verbose or --quiet, not both.")
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve --post-pr51 shortcut vs explicit --since
    if args.post_pr51 and args.since is not None:
        parser.error("Pass either --post-pr51 or --since, not both.")
    min_capture_time = (
        PR51_MERGE_TIME_UTC if args.post_pr51 else args.since
    )

    # Observations path resolution
    observations_path: Path | None
    if args.no_observations:
        observations_path = None
    else:
        observations_path = args.observations
        if not observations_path.exists():
            logging.warning(
                "Observations file not found at %s — proceeding without audio hints.",
                observations_path,
            )
            observations_path = None

    # Sanity-check image dir
    if not args.image_dir.exists():
        print(
            f"ERROR: Image directory not found: {args.image_dir}",
            file=sys.stderr,
        )
        return 2

    # Pre-flight summary so the user can Ctrl-C before spending API credits.
    print("=" * 60)
    print("Pre-label run configuration")
    print("=" * 60)
    print(f"  Image dir:        {args.image_dir}")
    print(f"  Observations:     {observations_path or '(none)'}")
    print(f"  Output:           {args.output}")
    print(f"  Model:            {args.model}")
    print(f"  Temperature:      {args.temperature}")
    print(f"  Camera filter:    {args.camera or '(both)'}")
    print(f"  Min capture time: {min_capture_time or '(no cutoff)'}")
    print(f"  Limit:            {args.limit or '(no limit)'}")
    if args.limit:
        print(f"  Estimated cost:   ~${_estimate_cost(args.limit):.2f}")
    print("=" * 60)

    if args.limit is None:
        print(
            "\nWARNING: No --limit set. This will process every matching image, "
            "which may be thousands.\n"
            "Consider --limit 10 for a smoke test first.\n"
        )

    try:
        labeler = PreLabeler(
            model_name=args.model,
            temperature=args.temperature,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    try:
        summary = labeler.run(
            image_dir=args.image_dir,
            observations_path=observations_path,
            output_path=args.output,
            limit=args.limit,
            min_capture_time=min_capture_time,
            camera_filter=args.camera,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress saved to", args.output, file=sys.stderr)
        return 130

    # Human-readable summary
    print("\n" + "=" * 60)
    print("Run summary")
    print("=" * 60)
    print(f"  Attempted:              {summary['attempted']}")
    print(f"  Succeeded:              {summary['succeeded']}")
    print(f"  Failed:                 {summary['failed']}")
    print(f"  Skipped (already done): {summary['skipped_already_labeled']}")
    print(f"  Skipped (too old):      {summary['skipped_too_old']}")
    print(f"  Wall clock:             {summary['wall_clock_seconds']:.1f}s")
    if summary["succeeded"] > 0:
        avg = summary["total_elapsed_seconds"] / summary["succeeded"]
        print(f"  Avg per image:          {avg:.2f}s")
    print("=" * 60)
    print(f"\nPre-labels written to: {args.output}")

    # Exit code: 0 if anything succeeded or there was nothing to do, 1 if
    # we attempted images but all failed (likely auth / network issue).
    if summary["attempted"] > 0 and summary["succeeded"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
