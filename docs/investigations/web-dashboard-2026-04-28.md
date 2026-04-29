# Investigation: Web Dashboard (Phase 8C)

**Date:** 2026-04-28
**Authors:** Kane Cruz-Walker, Daniel Wen
**Branch:** `feat/web-dashboard` → stacked on `claude/hardcore-chaum-b97b2c`
**Status:** Implemented (PRs 1–10 shipped, PR open at
[avis-birdfeeder#65](https://github.com/kanecruzwalker/avis-birdfeeder/pull/65)). Last updated 2026-04-29.

> **Note from 2026-04-29.** This document is preserved as the original
> planning artifact. Inline annotations (`> Status:` blocks, ✅ marks
> on the build-order checklist) reflect what actually shipped and how
> it deviated from the plan. The "Deltas from plan" section near the
> end summarises the meaningful differences for future readers.

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

> **Resolved 2026-04-29.** Decisions taken during implementation:

- [x] **Chat: wait-for-full response** (not SSE). The wire shape stays
      simple, response times under 30 s in practice, and PR 8 ships
      sooner. Streaming can be revisited if a user complains.
- [x] **Default landing view: live.** It's the only view that
      benefits from being the entry point — the others are
      explore-on-demand.
- [x] **Theme: reuse labeler UI's six-theme system** (warm / pollen /
      mono × light / dark). Visual consistency between dashboard and
      labeler outweighs aesthetic novelty.
- [x] **Mobile-first responsive: yes.** Tailscale + phone hotspot is
      the primary access pattern; the labeler UI already validated
      touch targets.
- [x] **Suppressed observations: hidden by default.** A "Show
      suppressed" toggle in the recent / timeline / gallery filter
      bars surfaces them when wanted. Friends and demo audiences see
      a clean dispatched-only stream by default.

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

> **Status 2026-04-29.** PRs 1–9 shipped as planned; PR 10 was added
> to wire the cross-process MJPEG bridge that the original plan
> referenced as a deferred concern. Commits live on
> `claude/hardcore-chaum-b97b2c` stacked on `feat/web-dashboard`;
> PR open at
> [avis-birdfeeder#65](https://github.com/kanecruzwalker/avis-birdfeeder/pull/65).

1. ✅ **PR 1: scaffold + auth + health** (`c06d325`) — `src/web/app.py`,
   `auth.py`, `/health`, systemd unit. Deployable, proves the auth
   wall works.
2. ✅ **PR 2: status + observations read-only** (`aad00c4`) —
   `/api/status`, `/api/observations`, `/api/observations/{id}`.
3. ✅ **PR 3: stream buffer + MJPEG** (`b283c2b`) —
   `VisionCapture.publish()`, `stream_buffer.py`, `/api/stream`,
   `/api/frame`. Live preview works in browser when both processes
   share an instance.
4. ✅ **PR 4: image variants** (`74fe8ca`) — schema addition
   (`image_path_full`), dispatch path saves all three, three image
   endpoints behind `/api/observations/{id}/image/{variant}`.
5. ✅ **PR 5: box cache + annotation** (`1102cf9`) — `box_cache.py`,
   `src/util/frame_annotator.py` (lifted to `src.util` so
   `src.vision` could import without inverting the dependency
   arrow), annotation at publish time.
6. ✅ **PR 6: HTML SPA — live + recent views** (`6940ea3`) —
   `index.html`, `styles.css` (six-theme system), `app.js`,
   `views/{live,recent}.js`, hash router, theme switcher, 30 s
   status poll.
7. ✅ **PR 7: timeline + gallery + detail views** (`e02c2ff`) — three
   new view modules + shared filter bar; router upgraded to parse
   `#/<view>/<id>` so detail can deep-link.
8. ✅ **PR 8: chat endpoint + UI** (`e44b6eb`) — `POST /api/ask`,
   `chat.js`, bubble thread. Wait-for-full (no SSE);
   `BirdAnalystAgent` imported lazily so missing langchain doesn't
   block dashboard boot.
9. ✅ **PR 9: Tailscale + ngrok docs + script** (`f7b268d`) —
   `scripts/avis-web-ngrok.sh`, `docs/WEB_DASHBOARD.md`, README +
   `PI_DEPLOYMENT.md` updates.
10. ✅ **PR 10: cross-process MJPEG bridge** (`57b804b`, *added during
    implementation*) — `src/web/shared_frame_bridge.py` +
    `VisionCapture.attach_preview_sink()`. Both processes activate
    via `AVIS_STREAM_SHM=<name>`; `/api/stream` now serves frames
    across the `avis.service` ↔ `avis-web.service` boundary in
    production.

Total: 13 commits on top of `origin/feat/web-dashboard` (PRs 4–10
plus three quality-cleanup commits and the AVIS_WEB_TOKEN-leak fix).
318 tests passing.

Estimated effort vs actual: original estimate was 2–3 weeks of cafe
sessions. PRs 4–10 + cleanup landed in ~3 sessions over Apr 28–29
(prior PRs 1–3 were the bulk of the calendar time).

---

## Deltas from plan

What actually shipped that the original plan didn't anticipate, or
that diverged once we hit the code:

- **PR 10 added.** The plan acknowledged that `VisionCapture.publish()`
  is a no-op when `stream_buffer is None` and that the dashboard
  process needed a way to share the buffer with the agent process.
  The plan didn't propose how. PR 10 ships single-slot
  `multiprocessing.shared_memory` with a sequence-number polling
  subscriber. Activated by setting `AVIS_STREAM_SHM=<name>` on both
  systemd units; unset = same as the original "no-op" path.
- **`frame_annotator.py` lives in `src/util/`, not `src/web/`.** The
  plan put it under `src.web`, but `src.vision` needs to import it
  for at-publish-time annotation, and `src.web → src.vision` is the
  established arrow. Moving the module to `src.util` keeps the
  arrow and lets both sides import freely.
- **`auth.py` deduplicated into `src/util/web_auth.py`.** The plan
  said "lift from `tools/labeler/ui/auth.py`". We did, then kept
  noticing both copies drifting; PR 4 cleanup pulled the shared
  logic into `src/util/web_auth.py` and made both surfaces 25-line
  re-exports. The labeler test suite confirms parity.
- **`/api/observations` `dispatched` is tri-state, not bool.** The
  plan listed `dispatched (bool, default true)`. The SPA's "Show
  suppressed" toggle needed a third state ("both"); the route
  accepts `true` / `false` / `all` (string), default `true`.
- **Chat response shape is wider than the plan.** Plan said
  `{ answer, tools_used }`. Actual shape mirrors
  `AnalystResponse.to_dict()`: `answer`, `tools_called` (not
  `tools_used`), `confidence`, `llm_available`, `error`,
  `generated_at`. The wider shape lets the SPA distinguish "LLM
  unavailable / fallback" from a real answer.
- **`/api/status` agent-status heuristic is named.** The plan
  alluded to "agent status"; the implementation pins it to three
  states (`live` < 60 s, `idle` < 10 min, `stale` otherwise) based
  on `observations.jsonl` mtime, surfaced in the topbar chip.
- **AVIS_WEB_TOKEN no longer printed in the startup banner.**
  Original banner echoed the full token. Caught during PR 4 review;
  banner now shows a fingerprint only (`xxxx…yyyy`).
- **`docs/WEB_DASHBOARD.md` is the operator surface.** The plan
  mentioned README updates and a "Tailscale invite, document in
  README" item. The doc grew larger than a README section deserves
  (~290 lines covering install, Tailscale, ngrok, troubleshooting,
  bridge setup), so it lives as its own file under `docs/` with
  README + `PI_DEPLOYMENT.md` pointing at it.

What the plan *anticipated* and the implementation honoured:

- ✅ Single-source-of-truth frame publish from `VisionCapture._cycle()`
  (no second picamera2 consumer, no second NPU caller).
- ✅ Last-known-box overlay with fade; box updates after fusion
  regardless of dispatch outcome (so suppressed detections still
  flash a box in the live view).
- ✅ Token middleware mirroring labeler/ui exactly
  (`hmac.compare_digest`, header or query param, `/health` only
  public route).
- ✅ Six-theme system lifted from labeler verbatim; component
  styles dashboard-specific.
- ✅ Read-only adapter over `observations.jsonl`; the dashboard
  never writes anything the agent reads.

---

## Followup investigation (TODO)

The original "Final" rollout step called for "investigation followup
doc with measured numbers (frame rate, bandwidth per viewer, NPU
contention check, latency)." Not yet done — needs a Pi session with
the bridge enabled and a couple of test viewers. Track separately
under Phase 8D demo prep.

---

*Edit this document as decisions firm up. Anything in "Open questions"
needs an answer before its corresponding PR starts.*
