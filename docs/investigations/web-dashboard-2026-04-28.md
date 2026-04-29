# Investigation: Web Dashboard (Phase 8C)

**Date:** 2026-04-28
**Authors:** Kane Cruz-Walker, Daniel Wen
**Branch:** `feat/web-dashboard` (to be created)
**Status:** Planning

---

## Goal

Build a Pi-hosted FastAPI web application that gives Kane, Dan, and invited
friends a single page to:

1. Watch a live preview of the feeder
2. See classification boxes overlaid in near-real-time when birds are detected
3. View detection history with a filterable timeline, recent-list, and gallery view
4. Inspect each detection with three image variants (cropped, annotated, full frame)
5. Ask the BirdAnalystAgent natural-language questions about observed activity

Access is via Tailscale primary, ngrok ephemeral for class demos. Application-layer
token (`AVIS_WEB_TOKEN`) protects all endpoints except `/health`.

---

## Hypothesis

A web dashboard built directly on top of the existing pipeline (motion-gated
capture loop, observations.jsonl, BirdAnalystAgent) requires no changes to the
production agent. By exposing existing artifacts via HTTP and re-broadcasting
preview frames from the existing capture loop, we add a viewing surface without
introducing new resource contention.

The dashboard will be the primary demo artifact for the IEEE report and slide
deck (Phase 8D).

---

## Architecture decisions

### Live stream source

**Decision: re-use the existing motion-gate capture loop.**

`VisionCapture._cycle()` already pulls a frame from each camera every ~200ms
to compute the motion gate. We will modify it to also publish a downsized
JPEG (640×360, quality 75) into a thread-safe ring buffer. The agent's
existing high-resolution path is untouched.

- ~5fps preview stream, no additional camera/NPU contention.
- Single source of truth for frames.
- No parallel picamera2 consumer.

Rejected alternative: dedicated capture worker thread at 10fps. Risks watchdog
timeouts on the already-loaded Pi and a second picamera2 consumer is fragile.

Rejected alternative: re-broadcast only the dispatched frames. Idle feeder = 
frozen page, fails the "live" expectation.

### Classification box overlay strategy

**Decision: last-known box with fade.**

When the agent's YOLO inference produces a bounding box, we cache it in a
shared `BoxCache` along with the species, confidence, and a timestamp.
Every streamed frame is annotated with any cached box younger than 3 seconds,
fading to transparent over the last second.

- Zero additional NPU calls.
- ~microsecond CPU overhead per frame (PIL rectangle + text).
- Box "blinks on" during active detections, clean stream otherwise.

Rejected alternative: continuous YOLO on every streamed frame. NPU contention
risk and PR #48 territory. Save the budget for real classifications.

### Three image variants

Every dispatched observation already saves the cropped frame to
`data/captures/images/`. We will additionally:

- Persist the YOLO-annotated version (drawn on the full frame) when YOLO mode
  is active and produced a box.
- Add `image_path_full` to `BirdObservation` schema (backward-compatible,
  optional, default None).

Three endpoints serve them by observation ID:

- `GET /observations/{id}/image/cropped` — what the classifier saw
- `GET /observations/{id}/image/annotated` — YOLO box + label overlay
- `GET /observations/{id}/image/full` — full 2304×1296 frame

The detail view in the UI shows all three with a tabbed switcher.

### Multi-viewer support via Tailscale

**Decision: Tailscale-primary, ngrok-ephemeral.**

The MJPEG ring buffer broadcasts the same frame to all connected subscribers
from a single capture source. CPU stays flat with viewer count; only outbound
network scales linearly.

Practical limits:

- ~30KB JPEG × 5fps = 150KB/s per viewer
- 2 viewers comfortable on home WiFi
- 5+ viewers may saturate hotspot upload — viewer count limit will be
  enforced in the stream endpoint (configurable, default 5).

Friends can be invited individually to Kane's tailnet (Tailscale handles per-
device key revocation). Public class demos use ngrok with a printed URL,
torn down after the demo.

---

## API contract

All endpoints except `/health` require `X-Avis-Token` header or `?token=` query.

```
GET /                              — HTML dashboard (single page)
GET /health                        — { "status": "ok" } (unauthenticated)

GET /api/status                    — uptime, total detections, current mode,
                                     last detection timestamp, agent status

GET /api/stream                    — multipart/x-mixed-replace MJPEG, ~5fps
GET /api/frame                     — single most recent annotated JPEG

GET /api/observations              — paginated list, query params:
                                     from, to (ISO timestamps)
                                     species (4-letter code)
                                     dispatched (bool, default true)
                                     limit (default 50, max 500)
                                     cursor (opaque, for pagination)

GET /api/observations/{id}                — single observation as JSON
GET /api/observations/{id}/image/cropped  — JPEG
GET /api/observations/{id}/image/annotated — JPEG
GET /api/observations/{id}/image/full     — JPEG

POST /api/ask                      — body: { "question": str }
                                     returns: { "answer": str, "tools_used": [...] }
                                     proxies to BirdAnalystAgent.answer()
```

JSON shapes mirror `BirdObservation` from `src/data/schema.py` exactly. No
transformation layer.

---

## Frontend component breakdown

Single-page vanilla JS SPA, mirroring the pattern from `tools/labeler/ui/`.
No build step, no React, no framework. CSS variables for theming.

```
src/web/templates/index.html      — root layout
src/web/static/styles.css         — reuse labeler UI's six-theme system
src/web/static/app.js             — entry, view router

src/web/static/views/
├── live.js          — MJPEG <img src="/api/stream"> + status bar
├── timeline.js      — horizontal scrub with detection markers, jump-to-detail
├── recent.js        — vertical infinite-scroll list of cards
├── gallery.js       — grid of cropped thumbnails
├── detail.js        — single observation, three image tabs, metadata
└── chat.js          — text input → POST /api/ask → bubble thread
```

Top-level navigation switches views via `body[data-view]` (same SPA pattern as
the labeler UI). All views consume the same `/api/observations` endpoint with
different query params and rendering.

---

## Backend module breakdown

```
src/web/__init__.py
src/web/app.py                 — FastAPI app factory, middleware
src/web/auth.py                — token middleware (lift from labeler/ui/auth.py)
src/web/routes/
├── __init__.py
├── status.py                  — /api/status, /health
├── stream.py                  — /api/stream, /api/frame (MJPEG generator)
├── observations.py            — /api/observations*, image variants
└── chat.py                    — /api/ask
src/web/stream_buffer.py       — thread-safe ring buffer (~30 frames retained)
src/web/box_cache.py           — last-known YOLO box with TTL
src/web/frame_annotator.py     — PIL: draw box, label, fade alpha
src/web/observation_store.py   — read-only adapter over observations.jsonl
src/web/templates/index.html
src/web/static/...
```

`VisionCapture` modifications (minimal):

```python
# src/vision/capture.py
def _cycle(self, ...):
    raw_frame = self._capture_raw()
    self.stream_buffer.publish(raw_frame)   # NEW — non-blocking
    if self._motion_gate(raw_frame):
        ...                                  # existing path unchanged
```

`BirdAgent` modifications (minimal):

```python
# src/agent/bird_agent.py
def _cycle(self):
    ...
    if detection_box:
        self.box_cache.update(detection_box, species, confidence)  # NEW
    ...
```

The `stream_buffer` and `box_cache` are dependency-injected so the web app
imports them; the agent never imports anything web-related (preserves current
dependency direction).

---

## Authentication

Mirrors `tools/labeler/ui/auth.py` exactly:

- `AVIS_WEB_TOKEN` in `.env`, minimum 16 characters, server refuses to start
  without it.
- Middleware checks `X-Avis-Token` header or `?token=` query param.
- `hmac.compare_digest` for timing-attack resistance.
- `/health` is the only unauthenticated path.

For the MJPEG stream specifically, the token must be in the query param
(`<img src="/api/stream?token=...">`) because browsers don't pass custom
headers on `<img>` requests. This is acceptable because the stream URL is
loaded inside an authenticated page — the token is already exposed to the
user who has it.

---

## Rollout plan

1. **Local laptop development.** Run on Kane or Dan's laptop with mocked
   stream buffer and observation store reading from a copied
   `observations.jsonl`. Get all views working without Pi dependencies.
2. **Pi deployment.** SSH-deploy, add to systemd, expose on port 8000 inside
   Tailscale network only. Kane and Dan invited to tailnet. Verify multi-
   viewer behavior.
3. **Friend invitations.** Tailscale per-friend invite, document in README.
4. **Demo prep.** ngrok script ready, never auto-started.
5. **Final.** Investigation followup doc with measured numbers (frame rate,
   bandwidth per viewer, NPU contention check, latency).

---

## Success criteria

- Page loads in under 1 second on Tailscale.
- MJPEG stream sustains 5fps with up to 3 simultaneous viewers without
  agent watchdog trips.
- Classification boxes appear within 1 second of dispatch.
- All three image variants accessible per observation.
- Timeline filter by species and date range works on the full
  ~21,000-record observation history.
- Chat endpoint round-trips a question to BirdAnalystAgent and returns an
  answer with tool-use trace under 30 seconds.
- Token middleware rejects all requests without a valid token (one test per
  endpoint).
- Zero changes to dispatched observation behavior — web app is read-only
  with respect to production state.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Pi CPU spike from JPEG encoding at 5fps | Profile before merge; downscale to 480×270 or drop to 3fps if needed |
| picamera2 contention if a second consumer is added later | Document explicitly that `stream_buffer.publish()` is the only allowed second consumer |
| NPU contention if YOLO-on-stream is added later | Keep last-known-box decision documented; reject "annotate every frame" feature requests |
| `observations.jsonl` grows unbounded, full-scan filter gets slow | Index by date once we exceed 100k records; current 21k is fine for full scan |
| Ring buffer memory growth | Fixed-size ring (default 30 frames @ 30KB = ~1MB), oldest evicted |
| Friend abuses access | Tailscale per-device revoke is one click |
| Token leaks in chat history or logs | Never log full URLs; redact `?token=` in any log line |
| Browser tab left open consumes bandwidth indefinitely | Inactive-stream-disconnect timeout of 5 min; reconnect on focus |

---

## Rollback plan

Web app runs as a separate systemd service (`avis-web.service`), independent
of `avis.service`. Disabling it does not affect the agent.

```bash
sudo systemctl stop avis-web
sudo systemctl disable avis-web
```

`VisionCapture.publish()` is a no-op if `stream_buffer` is None (default).
`BirdAgent.box_cache.update()` is a no-op if `box_cache` is None.
Reverting `feat/web-dashboard` removes the web app entirely; production
agent behavior is unchanged.

---

## Out of scope (future PRs)

- Webhook integration (separate, already stubbed in `notifier.py`)
- Push notification triggering from the web UI
- Multi-feeder federation
- Video clips (we have audio + still frame; video would require new capture path)
- User accounts / per-user permissions (single token model is sufficient)
- Public unauthenticated demo mode

---

## Open questions for Kane + Dan to decide before coding

- [ ] Do we want the chat endpoint to stream tokens (SSE) or wait for full
      response? BirdAnalystAgent calls Gemini which can be slow; streaming is
      a better UX but more code.
- [ ] Default landing view: live, recent, or timeline?
- [ ] Theme — reuse labeler UI's six-theme system or pick a single deep-space
      bioluminescent palette consistent with Drift?
- [ ] Mobile-first responsive, or desktop-only for MVP?
- [ ] Show suppressed observations (`dispatched=False`) by default in the
      timeline, or filter them out unless explicitly requested?

---

## File checklist for Dan's Claude session handoff

These are the files Dan should paste into his Claude chat at the start so
his Claude has the same context as Kane's:

1. This investigation doc
2. `SESSION_NOTES.md` (Kane's gist, latest comment)
3. `CHANGELOG.md` (full file)
4. `ROADMAP.md`
5. `docs/ARCHITECTURE.md`
6. `tools/labeler/ui/server.py`, `routes.py`, `auth.py`, `review_store.py`,
   `templates/index.html`, `static/app.js`, `static/styles.css`
   (the reference FastAPI + auth + vanilla JS implementation in this repo)
7. `src/notify/notifier.py` (webhook stub he may wire later)
8. `src/agent/bird_analyst_agent.py` and `src/agent/tools/` (for `/api/ask`)
9. `src/data/schema.py` (`BirdObservation` is the JSON shape returned)
10. `src/vision/capture.py` (the file getting `stream_buffer.publish()`)
11. A 100-record sample of `logs/observations.jsonl` (real data shape)
12. `configs/paths.yaml` and `configs/notify.yaml` (YAML conventions)

Dan's first prompt to his Claude:

> "Continuing Avis — Phase 8C web dashboard. Pasting investigation doc plus
> reference files. Start with `src/web/app.py` skeleton, `src/web/auth.py`
> mirroring `tools/labeler/ui/auth.py`, and `/health` endpoint. Tests
> alongside each module."

---

## Build order (suggested PR sequence)

Each PR mergeable independently, each adds one capability.

1. **PR 1: scaffold + auth + health** — `src/web/app.py`, `auth.py`, 
   `/health`, systemd unit. Deployable, does nothing useful, proves the
   auth wall works.
2. **PR 2: status + observations read-only** — `/api/status`,
   `/api/observations`, `/api/observations/{id}`, basic JSON. No frontend yet.
3. **PR 3: stream buffer + MJPEG** — `VisionCapture.publish()`,
   `stream_buffer.py`, `/api/stream`, `/api/frame`. Live preview works in
   browser.
4. **PR 4: image variants** — schema additions, dispatch path saves all
   three, three image endpoints.
5. **PR 5: box cache + annotation** — `box_cache.py`, `frame_annotator.py`,
   stream now shows boxes.
6. **PR 6: HTML SPA — live + recent views** — `index.html`, base CSS,
   live.js, recent.js, view router.
7. **PR 7: timeline + gallery + detail views** — three more views, filter
   controls, image tabs in detail view.
8. **PR 8: chat endpoint + UI** — `/api/ask`, chat.js, bubble thread.
9. **PR 9: Tailscale docs + ngrok script + README** — deployment finish.

Estimated total: 2–3 weeks of cafe sessions.

---

*Edit this document as decisions firm up. Anything in "Open questions"
needs an answer before its corresponding PR starts.*
