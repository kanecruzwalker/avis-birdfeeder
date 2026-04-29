#!/usr/bin/env bash
#
# avis-web-ngrok.sh — open an ngrok tunnel to the Avis web dashboard.
#
# Used for ad-hoc class demos where Tailscale isn't an option (audience
# laptops aren't on the tailnet). Tears down on Ctrl-C and never
# auto-starts — see scripts/avis-web.service for the always-on path.
#
# Usage (from project root on the Pi, with avis-web.service running):
#
#     export AVIS_WEB_TOKEN=<the token>      # or source .env
#     bash scripts/avis-web-ngrok.sh
#
# What it does:
#   1. Refuses to start if AVIS_WEB_TOKEN is unset (safety — never publish
#      an unauthenticated URL).
#   2. Starts `ngrok http 8000` in the background.
#   3. Polls ngrok's local API (http://127.0.0.1:4040/api/tunnels) for the
#      assigned public URL.
#   4. Prints the demo URL with ?token= already filled in so the operator
#      can paste it into chat / projector once.
#   5. Waits on the tunnel; Ctrl-C kills ngrok and exits cleanly.
#
# Requirements:
#   - ngrok in PATH (curl https://ngrok.com/install for one-line install)
#   - ngrok auth-token configured (`ngrok config add-authtoken <token>`)
#   - python3 (used to parse ngrok's JSON — avoids a jq dependency)
#   - The dashboard already running on localhost:8000 (the script does
#     not start uvicorn — that's avis-web.service's job)
#
# What it does NOT do:
#   - Does not survive shell logout. For a backgrounded long-running
#     tunnel, run inside `screen` / `tmux` / `nohup`. The intended
#     flow is: open laptop, start tunnel, do the demo, Ctrl-C.
#   - Does not configure ngrok. First-time install uses `ngrok config
#     add-authtoken <your-token>`; this script assumes that's done.

set -euo pipefail

# ── Pre-flight ───────────────────────────────────────────────────────────────

if [ -z "${AVIS_WEB_TOKEN:-}" ]; then
    # Try .env in cwd as a convenience — same file the dashboard reads.
    if [ -f .env ]; then
        # shellcheck disable=SC1091
        set -a; . ./.env; set +a
    fi
fi

if [ -z "${AVIS_WEB_TOKEN:-}" ]; then
    echo "ERROR: AVIS_WEB_TOKEN is not set." >&2
    echo "       Source .env or export the token before running this script." >&2
    echo "       Refusing to start — an ngrok URL without auth would expose" >&2
    echo "       the dashboard to anyone who guesses the URL." >&2
    exit 2
fi

if ! command -v ngrok >/dev/null 2>&1; then
    echo "ERROR: ngrok not found in PATH." >&2
    echo "       Install: curl -sSL https://ngrok.com/install | sh" >&2
    echo "       Then: ngrok config add-authtoken <your-ngrok-token>" >&2
    exit 3
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found in PATH (used to parse ngrok's API output)." >&2
    exit 3
fi

# Sanity: the dashboard should already be running. Tunnel without a
# backend is just a confusing 502.
if ! curl -fsS -o /dev/null --max-time 2 http://127.0.0.1:8000/health; then
    echo "WARNING: dashboard not responding at http://127.0.0.1:8000/health." >&2
    echo "         Start avis-web.service first:" >&2
    echo "             sudo systemctl start avis-web" >&2
    echo "         Continuing anyway — the tunnel will work once the service comes up." >&2
fi

# ── Launch ngrok ─────────────────────────────────────────────────────────────

NGROK_LOG="$(mktemp -t avis-ngrok-XXXXXX.log)"
trap 'rm -f "$NGROK_LOG"' EXIT

ngrok http 8000 --log=stdout >"$NGROK_LOG" 2>&1 &
NGROK_PID=$!

cleanup() {
    echo
    echo "Tearing down ngrok tunnel (pid=$NGROK_PID)..."
    kill "$NGROK_PID" 2>/dev/null || true
    wait "$NGROK_PID" 2>/dev/null || true
    echo "Done."
}
trap cleanup INT TERM

# ── Wait for the public URL ──────────────────────────────────────────────────
#
# ngrok's local API at 127.0.0.1:4040 reports the tunnel once it's up.
# Poll for ~10 s; abort if we never see a https:// URL.

PUBLIC_URL=""
for _ in $(seq 1 20); do
    sleep 0.5
    JSON=$(curl -fsS http://127.0.0.1:4040/api/tunnels 2>/dev/null || true)
    if [ -z "$JSON" ]; then
        continue
    fi
    PUBLIC_URL=$(echo "$JSON" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for t in data.get("tunnels", []):
    if t.get("proto") == "https" and t.get("public_url"):
        print(t["public_url"])
        break
')
    if [ -n "$PUBLIC_URL" ]; then
        break
    fi
done

if [ -z "$PUBLIC_URL" ]; then
    echo "ERROR: ngrok did not report a public URL within 10 seconds." >&2
    echo "       Last log lines:" >&2
    tail -n 20 "$NGROK_LOG" >&2 || true
    cleanup
    exit 4
fi

# ── Print the demo URL ───────────────────────────────────────────────────────

DEMO_URL="$PUBLIC_URL/?token=$AVIS_WEB_TOKEN"

echo
echo "================================================================"
echo "  Avis web dashboard — ngrok demo URL"
echo "================================================================"
echo
echo "  $DEMO_URL"
echo
echo "  Paste once into chat / projector. The SPA strips the ?token="
echo "  on first load, so the URL in the address bar will be clean."
echo
echo "  Tunnel:    $PUBLIC_URL"
echo "  Inspector: http://127.0.0.1:4040  (request log, request replay)"
echo
echo "  Press Ctrl-C to tear down the tunnel."
echo "================================================================"
echo

# ── Wait on ngrok ────────────────────────────────────────────────────────────

wait "$NGROK_PID"
