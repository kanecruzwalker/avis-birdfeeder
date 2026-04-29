# Web dashboard — deploy and operate

> Single-page dashboard served from the Pi. Live MJPEG stream,
> observation history, image variants, and a chat proxy to the LLM
> analyst. Token-authenticated, served over Tailscale by default,
> ngrok for class demos.

This doc is the operator guide: install, expose to friends, rotate
the token, demo to a class, troubleshoot. Architecture and design
trade-offs live in
[`docs/investigations/web-dashboard-2026-04-28.md`](investigations/web-dashboard-2026-04-28.md).

---

## What it is

The dashboard is a separate FastAPI app (`src/web/`) that runs as its
own systemd unit (`avis-web.service`), independent of the agent
(`avis.service`). Stopping the dashboard never touches the agent. The
dashboard is read-only with respect to production state — it reads
`logs/observations.jsonl` and the agent's in-memory stream buffer, and
writes nothing.

Five views, all hash-routed in a vanilla-JS SPA:

| Hash | View |
|------|------|
| `#/live` | MJPEG preview + agent status sidebar |
| `#/recent` | Card list of recent observations, dispatched-only by default |
| `#/timeline` | SVG scrub bar with one marker per observation, filterable by window + species |
| `#/gallery` | Auto-fill grid of cropped thumbnails with overlay metadata |
| `#/detail/<id>` | Single observation: cropped / annotated / full image tabs + metadata |
| `#/chat` | Ask the LLM analyst about feeder activity |

---

## Auth model

The dashboard's only auth layer is `AVIS_WEB_TOKEN` (≥16 chars). Every
`/api/*` route, every image variant, and the MJPEG stream requires it.
`/health` and `/static/*` are public — the bundle has no secrets and
the SPA's first boot needs to render before it can authenticate any
API call.

**Two ways to send the token:**

- `X-Avis-Token: <token>` header (the SPA's default for fetch calls)
- `?token=<token>` query string (the only option for `<img src=…>`
  and the SPA's first-boot URL handoff)

The SPA's first-load flow:

1. Operator opens `https://<host>/?token=<token>` once.
2. SPA reads the param, persists to `localStorage`, replaces the URL
   with the bare path via `history.replaceState`. The token never
   sits in the address bar after that.
3. Subsequent fetches send the header. MJPEG `<img>` tags use a
   query-string variant generated from cached state.

**Rotation:** change `AVIS_WEB_TOKEN` in `.env` and `sudo systemctl
restart avis-web`. Old token instantly invalid; old browser sessions
401 on the next call and clear their cache automatically.

---

## First-time install on the Pi

Assumes the agent (`avis.service`) is already deployed per
[`PI_DEPLOYMENT.md`](PI_DEPLOYMENT.md). The web stack reuses the same
venv (`/mnt/data/avis-venv`) and `.env` file.

### 1. Generate a token

Any random string ≥16 characters. From the laptop:

```bash
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

### 2. Add it to the Pi's `.env`

The Pi already has `/mnt/data/avis-birdfeeder/.env` from the agent's
setup (Pushover keys, Gemini API key). Append:

```
AVIS_WEB_TOKEN=<paste your token here>
```

The dashboard refuses to start if the token is missing or shorter
than 16 chars — fail loud rather than silently accept unauthenticated
traffic.

### 3. Install the systemd unit

From the project root on the Pi:

```bash
sudo cp scripts/avis-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable avis-web
sudo systemctl start avis-web
```

The unit file is documented inline — see `scripts/avis-web.service`
for what each setting does and why.

### 4. Verify

```bash
sudo systemctl status avis-web        # one-line health check
sudo journalctl -u avis-web --since today | tail -20
curl http://127.0.0.1:8000/health     # public — no token needed
```

Expected: `{"status":"ok","service":"avis-web","version":"0.1.0"}`.

### 5. Open from your laptop

If you're on the tailnet:

```
http://birdfeeder01:8000/?token=<your-token>
```

(Replace `birdfeeder01` with whatever Tailscale name the Pi has.)

The SPA strips the `?token=` after first load. Bookmark the bare URL.

---

## Tailscale: invite a friend

Tailscale-primary, ngrok-ephemeral. Friends on the tailnet can
bookmark a stable URL; their device key is revocable per-friend.

### Invite

In the Tailscale admin console (https://login.tailscale.com/admin/users):

1. **Users → Invite users.** Enter the friend's email.
2. They install Tailscale, sign in, and accept.
3. Send them the URL:
   ```
   http://birdfeeder01:8000/?token=<the same token>
   ```

The token is shared — there's no per-user token in this design (out
of scope for the dashboard's MVP, see investigation doc). Trust is
"anyone on the tailnet with the token."

### Revoke

Admin console → **Devices** → find the friend's device → **Disable**.
Their Tailscale connection drops; without it they can't reach
`birdfeeder01:8000` even with the token. Token rotation is only
needed if you suspect the token itself leaked (e.g. screenshare
caught the URL pre-strip).

### Watch out

- **Don't port-forward 8000 from your home router.** Tailscale is the
  only reason the dashboard is safe to run on `0.0.0.0:8000` — it
  keeps the port off the public internet. Port-forwarding bypasses
  that.
- **Mobile hotspot:** works fine over Tailscale. Bandwidth budget is
  ~150 KB/s per active MJPEG viewer; cap is 5 simultaneous viewers
  per the stream buffer's subscriber limit.

---

## ngrok: ephemeral demo URL

For a class demo where audience laptops aren't on the tailnet, run
the included script:

```bash
bash scripts/avis-web-ngrok.sh
```

The script:

1. Refuses to run if `AVIS_WEB_TOKEN` is unset (an unauthenticated
   ngrok URL is a footgun).
2. Starts `ngrok http 8000`.
3. Polls ngrok's local API for the assigned public URL.
4. Prints the full demo URL with `?token=…` already filled in:
   ```
   https://abcd1234.ngrok-free.app/?token=<token>
   ```
5. Waits on the tunnel; **Ctrl-C tears it down.**

**One-time setup** (laptop or Pi, wherever you'll run the demo):

```bash
curl -sSL https://ngrok.com/install | sh
ngrok config add-authtoken <your-ngrok-authtoken>
```

The free tier is fine for a one-off classroom demo. URLs are random
each run; if you want a stable demo URL, pay for a reserved domain.

**Demo discipline:**

- Tear down with Ctrl-C as soon as the demo ends.
- Rotate `AVIS_WEB_TOKEN` after a public demo if the URL or
  screenshare might have caught the token in the address bar.
- Never paste the URL into a chat that gets auto-archived.

---

## Operating the dashboard

### Daily SSH commands

```bash
sudo systemctl status avis-web                     # is it up?
sudo systemctl restart avis-web                    # after .env edit
sudo journalctl -u avis-web -f                     # live log tail
sudo journalctl -u avis-web --since "1 hour ago"   # bounded scrollback
```

### What the agent chip means

The status chip in the topbar reads from `/api/status`, which
heuristically reports agent freshness based on `observations.jsonl`
mtime:

| State | Meaning |
|-------|---------|
| `live` | mtime within 60 s — agent is writing |
| `idle` | mtime within 10 min — slow period or empty feeder |
| `stale` | older than 10 min, or file missing — likely crashed |

Heuristic, not a contract. An empty feeder looks identical to a
crashed agent. Use `sudo systemctl status avis` for ground truth.

### What the live stream falls back to

In production today, the agent and dashboard run as separate systemd
units with no shared memory, so `/api/stream` returns 503. The
investigation doc's "cross-process bridge" (TODO, post-PR-9) wires
this up. Detail / Recent / Timeline / Gallery / Chat all work without
the bridge.

### Chat endpoint

`POST /api/ask` proxies to `BirdAnalystAgent.answer()`. Returns 503 if
`GEMINI_API_KEY` is not in the dashboard's environment — the analyst
is opt-in. Expect 5–30 s round-trip per question; the LLM is
synchronous, not streamed.

---

## Troubleshooting

### `ERROR: AVIS_WEB_TOKEN must be at least 16 characters`

The dashboard refuses to start with a weak token. Pick a longer one.
Don't disable this check; the token is the only auth boundary.

### `/api/stream` returns 503

Expected in the production split-process layout — see the "live
stream falls back to" section above. Working as intended until the
bridge ships. To preview locally, run the agent and dashboard in the
same Python process so they share the `StreamBuffer` instance.

### Dashboard reachable from the Pi (`curl localhost:8000`) but not from a laptop

- **Tailscale:** is the laptop's Tailscale client connected? `tailscale
  status` should list the Pi.
- **Hostname:** Magic DNS (`birdfeeder01`) only works inside the
  tailnet. From outside, use the Pi's tailnet IP (visible in the
  Tailscale admin console).
- **Bind address:** `scripts/avis-web.service` runs uvicorn on
  `0.0.0.0:8000`. Localhost-only (`127.0.0.1`) means a laptop can
  never reach it; the unit file ships with `0.0.0.0` for this reason.

### `401 unauthorized` after rotating the token

The browser cached the old token in `localStorage`. Open the URL
with the new `?token=…` once; the SPA overwrites the cached value.

### `503: Analyst not configured` on chat

`GEMINI_API_KEY` is not in the dashboard's environment. Add it to
`/mnt/data/avis-birdfeeder/.env` and `sudo systemctl restart
avis-web`. The chat view shows the same hint when this happens.

### Stream pegs the Pi's CPU

JPEG encoding at 5fps on the Pi 5 is normally well under 5% of one
core. If you're seeing spikes, check:

- Multiple clients streaming simultaneously (each one is a separate
  decode on the client, but the server's encode is shared via the
  ring buffer — total CPU should stay flat with viewer count).
- Annotation in `frame_annotator.py` doing a re-encode per viewer
  (it shouldn't — annotation happens once at publish time).

If genuinely overloaded, drop the publish rate in
`VisionCapture._maybe_publish_preview` from 5fps to 3fps.

### `Failed to load observations` toast on every view

The dashboard reads from `logs/observations.jsonl` relative to its
working directory. The systemd unit sets
`WorkingDirectory=/mnt/data/avis-birdfeeder`; if you're running by
hand from a different cwd, pass `--observations-path` or set the
`logs.observations` key in `configs/paths.yaml`.

---

## File layout reference

```
src/web/
├── app.py                 # FastAPI factory, app.state wiring
├── __main__.py            # CLI entry: python -m src.web --host …
├── auth.py                # Token middleware (re-exports src.util.web_auth)
├── observation_store.py   # mtime-cached read-only view of observations.jsonl
├── stream_buffer.py       # Thread-safe MJPEG ring with subscriber cap
├── box_cache.py           # YOLO box cache for live-preview annotation
├── routes/
│   ├── pages.py           # GET / + StaticFiles mount
│   ├── status.py          # /health, /api/status
│   ├── observations.py    # /api/observations*
│   ├── images.py          # /api/observations/{id}/image/{variant}
│   ├── stream.py          # /api/stream, /api/frame
│   └── chat.py            # /api/ask
└── static/                # SPA bundle
    ├── index.html
    ├── styles.css
    ├── app.js             # Token bootstrap, router, theme, status poll
    └── views/             # live, recent, timeline, gallery, detail, chat

scripts/
├── avis-web.service       # Systemd unit
└── avis-web-ngrok.sh      # Demo tunnel helper

docs/
├── WEB_DASHBOARD.md       # This file
└── investigations/web-dashboard-2026-04-28.md
                            # Design doc, API contract, decisions
```
