#!/usr/bin/env bash
#
# lint-local.sh — run the same lint checks CI runs, locally.
#
# Why this exists
# ---------------
# Two CI gotchas have caught us before:
#
# 1. Scope drift. CI lints `src/ tests/` (not just the subset I happened to
#    be touching). Manual `python -m ruff format --check src/web/ tests/web/`
#    misses files outside that scope — e.g. PR 10 unformatted
#    `src/agent/experiment_orchestrator.py` and CI rejected the merge of #65.
#
# 2. Ruff version mismatch. CI uses the version pinned in
#    `requirements-dev.txt` (currently `ruff==0.4.4`). A locally-installed
#    newer ruff (e.g. `0.15.x`) disagrees with CI on what counts as
#    "formatted" — local check passes, CI fails, or vice versa.
#
# This script verifies the local ruff matches the pin before running the
# checks. If it doesn't, it prints the install command and exits.
#
# Usage (from anywhere inside the repo):
#
#     bash scripts/lint-local.sh
#
# Run before opening a PR. Exits non-zero on any check failure.

set -euo pipefail

# Resolve repo root so this script works whether you run it from repo root,
# scripts/, or a subdirectory.
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# ── Verify ruff version matches CI's pin ─────────────────────────────────────

EXPECTED=$(grep -oE '^ruff==[0-9.]+' requirements-dev.txt | head -1 | cut -d= -f3)
if [ -z "$EXPECTED" ]; then
    echo "ERROR: could not read pinned ruff version from requirements-dev.txt." >&2
    echo "       Look for a line like 'ruff==X.Y.Z'." >&2
    exit 2
fi

ACTUAL=$(python -m ruff --version 2>/dev/null | awk '{print $2}')
if [ -z "$ACTUAL" ]; then
    echo "ERROR: ruff is not installed in the active Python environment." >&2
    echo "       Install with: python -m pip install ruff==$EXPECTED" >&2
    exit 2
fi

if [ "$EXPECTED" != "$ACTUAL" ]; then
    echo "ERROR: ruff version mismatch — local ruff disagrees with CI." >&2
    echo "       expected (per requirements-dev.txt): $EXPECTED" >&2
    echo "       actual:                              $ACTUAL" >&2
    echo
    echo "       Install the pinned version with:" >&2
    echo "           python -m pip install ruff==$EXPECTED" >&2
    echo
    echo "       Re-running this script with the wrong version would either" >&2
    echo "       miss real CI failures or report false positives — fail loud" >&2
    echo "       rather than silently lie about lint status." >&2
    exit 2
fi

echo "ruff $ACTUAL — matches CI pin"
echo

# ── Run the same commands CI runs ────────────────────────────────────────────
#
# Both invocations match `.github/workflows/ci.yml` exactly. If you change
# the CI scope or steps, change this script too — the contract is "running
# this script tells you whether CI will pass."

echo "→ ruff format --check src/ tests/"
python -m ruff format --check src/ tests/
echo

echo "→ ruff check src/ tests/"
python -m ruff check src/ tests/
echo

echo "All lint checks passed (matches CI scope: src/ tests/, ruff $ACTUAL)."
