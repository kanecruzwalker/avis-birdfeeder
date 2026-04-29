# Changelog

All notable changes to Avis are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

### Added
- **Phase 8C — web dashboard backend** (`feat/web-dashboard`, three commits).
  - PR 1 (`c06d325`): FastAPI scaffold under `src/web/` with `AVIS_WEB_TOKEN`
    middleware (header or `?token=` query, `hmac.compare_digest`), `/health`
    as the only public route, and an independent `scripts/avis-web.service`
    systemd unit so stopping the dashboard never touches the agent. CLI
    validates the token before binding the socket. See
    `docs/investigations/web-dashboard-2026-04-28.md` for the full design.
  - PR 2 (`aad00c4`): read-only API over `logs/observations.jsonl`. New
    endpoints: `GET /api/status` (uptime, counts, agent_status heuristic
    derived from the file's mtime), `GET /api/observations` (newest-first
    cursor pagination with from/to/species/dispatched filters, limit
    clamped to 500), `GET /api/observations/{id}`. ID convention:
    `YYYYMMDDTHHMMSSffffff` UTC, sortable and reversible. The store is
    mtime-cached and thread-safe; the dashboard never writes to the file.
  - PR 3 (`b283c2b`): live MJPEG preview. New `src/web/stream_buffer.py`
    fixed-size ring (default 30 frames, ~1MB) with thread-safe
    `publish()` / `subscribe()`; condvar wakes subscribers; configurable
    subscriber cap (default 5) maps overflow to 503 + `Retry-After`.
    `VisionCapture.capture_frames()` publishes a 640×360 q=75 JPEG every
    cycle into an injected sink — no-op when the sink is `None`, so the
    agent stays runnable without the dashboard. New endpoints:
    `GET /api/stream` (multipart/x-mixed-replace MJPEG) and
    `GET /api/frame` (single most-recent JPEG). Per-frame wait timeout
    configurable via `app.state.stream_wait_timeout` (5 s prod, 0.3 s in
    tests).
  - PR 4: image variants. New schema field
    `BirdObservation.image_path_full` (optional, default `None` —
    backward compatible with all existing `observations.jsonl` records).
    `BirdAgent` now saves the full uncropped frame and, when YOLO mode
    produced a box, an annotated full frame, right after the cooldown
    gate passes. Filenames derive from the existing cropped path:
    `<stem>_full.png` and `<stem>_annotated.png`. New endpoint
    `GET /api/observations/{id}/image/{cropped|full|annotated}` serves
    each variant as `image/png`; the annotated path is derived from
    `image_path_full` by suffix swap, so the schema stays minimal. 404
    when the variant's path field is `None` or the file is missing on
    disk.
  - PR 5: box cache + live-preview annotation. New
    `src/web/box_cache.py` (single-slot cache with monotonic-time TTL
    and linear fade — defaults 3 s TTL with the trailing 1 s fading
    to alpha 0). New `src/util/frame_annotator.py` (pure PIL helper
    that decodes a JPEG, alpha-composites a green box + label, and
    re-encodes; lives in `src.util` so `src.vision` can import it
    without inverting the `src.web → src.vision` dependency arrow).
    `BirdAgent.__init__` accepts an optional `box_cache`; after
    fusion (regardless of dispatch outcome), the cache is updated
    with the latest YOLO box + species + confidence so suppressed
    detections still flash a box in the live preview.
    `VisionCapture.__init__` accepts an optional `box_source`; when
    set, `_maybe_publish_preview` scales the cached camera-native
    box to the preview JPEG's pixel space and overlays it before
    publishing. Annotation happens at publish time (single CPU
    cost regardless of viewer count), keeping all box-dependent
    work in the agent process.
  - PR 6: HTML SPA — live + recent views. New
    `src/web/routes/pages.py` (`GET /` serves the dashboard shell,
    no auth — the bundle has no secrets, and the SPA's first boot
    needs to read `?token=` from the URL before any API call).
    New `src/web/static/` bundle: `index.html`, `styles.css` (six
    themes lifted from `tools/labeler/ui` for visual consistency,
    plus dashboard-specific components — agent chip, live stage,
    observation cards, suppressed toggle, toast), `app.js` (token
    bootstrap → URL strip → localStorage cache, hash router for
    `#/live` ↔ `#/recent`, theme switcher, 30 s status poll), and
    `views/{live,recent}.js` (split per the investigation doc's
    SPA layout). The recent view colour-codes confidence (low/mid/
    high) and visually distinguishes dispatched vs suppressed so
    the deployment-data narrative ("scene-floor noise vs real
    birds") stays legible in the UI. Tri-state filter
    `dispatched=all` added to `/api/observations` for the SPA's
    "Show suppressed" toggle (the route's previous `bool | None`
    contract docstring claimed this behavior but FastAPI rejected
    non-bool values — small route fix to match the contract).
  - PR 7: timeline + gallery + detail views. Three new view modules
    in `src/web/static/views/` plus a shared filter bar (window /
    species / suppressed) reused between timeline and gallery.
    `timeline.js` renders one SVG marker per observation on a
    horizontal time axis; markers are anchor links to the detail
    view, color-coded by confidence band, half-size + greyed when
    suppressed. `gallery.js` fetches cropped thumbnails into a
    responsive auto-fill grid (~160 px tiles), with an overlay
    showing species, confidence, and timestamp. `detail.js` mounts
    on the parameterized hash route `#/detail/<id>`, fetches the
    record from `/api/observations/{id}` for the metadata panel,
    and renders an image-tab strip (cropped / annotated / full)
    that lazy-loads each variant via `/api/observations/{id}/image/
    {variant}` and falls back to a "not available" message on 404.
    Router upgraded to parse `#/<view>/<id>` so detail can be
    deep-linked. Topbar nav extended with Timeline + Gallery links.
  - PR 8: chat endpoint + UI. New `src/web/routes/chat.py` exposes
    `POST /api/ask`, which proxies the user's question to
    `BirdAnalystAgent.answer()` and returns its full structured
    response (`answer`, `tools_called`, `confidence`, `llm_available`,
    `error`, `generated_at`). The analyst is wired in via
    `create_app(analyst=...)` and stashed on `app.state.analyst`; when
    no analyst is configured (the default in tests, and any deploy
    without `GEMINI_API_KEY`), the route returns 503 with an operator
    hint. The (synchronous) `answer()` call is offloaded to a
    threadpool so the LangChain blocking call doesn't pin the
    uvicorn event loop. `python -m src.web` now opportunistically
    constructs the analyst from `configs/` when `GEMINI_API_KEY` is
    set, with all import + init failures degrading silently to "chat
    disabled" — `BirdAnalystAgent` is imported lazily inside that
    helper so missing langchain / google-genai installs don't block
    `--help`. New `src/web/static/views/chat.js` renders a bubble
    thread (user / assistant) with a collapsed "tools used" summary
    under each assistant turn and a 2000-char-capped composer; Enter
    sends, Shift+Enter inserts a newline. New `view-chat` section in
    `index.html`, topbar nav link, and theme-aware bubble styles in
    `styles.css`. No SSE / streaming — wait-for-full response keeps
    the wire shape and frontend simple, and matches the investigation
    doc's success criterion (under 30 seconds round-trip).
  - PR 10: cross-process MJPEG bridge via `multiprocessing.shared_memory`.
    New `src/web/shared_frame_bridge.py` (~150 LOC of bridge code,
    ~100 LOC of docstring) defines `SharedFramePublisher` (single-
    writer slot, atomic seq increment after payload write,
    monotonic u64) and `SharedFrameSubscriber` (polling reader with
    torn-read retry + a `start_pump(stream_buffer)` helper that
    forwards new frames into a duck-typed sink in a daemon thread).
    Single-slot, not a ring — at 5 fps the existing `StreamBuffer`
    already drops to "newest only" semantics, so a ring would be
    pure ceremony. Polling at 50 ms (no cross-process condvar);
    256 KB segment (~8x headroom over a 30 KB JPEG); little-endian
    layout matches every platform we ship to. Header reset on
    publisher reattach so a fresh agent doesn't latch subscribers
    onto a stale frame from a crashed prior run; OS-rounded segment
    sizes (Windows page-aligns to 4 KB) accepted via `>=` check.
    New `attach_preview_sink()` method on `VisionCapture` lets the
    agent's `experiment_orchestrator.main()` wire the publisher
    after construction without threading a kwarg through three
    `from_config` chains. Both processes activate the bridge by
    setting `AVIS_STREAM_SHM=<name>` to the same value (recommend
    putting it in the shared `.env` so both systemd units see it);
    unset = today's behaviour. Dashboard's startup banner reports
    bridge state; agent logs `Cross-process MJPEG bridge enabled` to
    the journal. 15 new tests in
    `tests/web/test_shared_frame_bridge.py` (lifecycle, round-trip,
    monotonic seq, single-slot skip-to-latest, oversized frame
    rejection, type validation, stale-segment header reset, pump
    forwarding, pump idempotency + survives sink exceptions). 318
    total tests passing. `WEB_DASHBOARD.md` gets a "Shared-memory
    bridge" section with enable steps and troubleshooting; the
    "/api/stream returns 503" entry updated with the three
    likeliest causes (bridge not enabled / boot order / agent not
    publishing).
  - PR 9: deployment finish — Tailscale + ngrok docs and demo script.
    New `scripts/avis-web-ngrok.sh` opens an ngrok HTTP tunnel to
    port 8000 and prints a clipboardable demo URL with `?token=`
    filled in; refuses to start if `AVIS_WEB_TOKEN` is unset (an
    unauthenticated ngrok URL is a footgun), polls ngrok's local
    API for the public URL, and tears down on Ctrl-C. New
    `docs/WEB_DASHBOARD.md` is the operator guide: auth model and
    SPA token handoff flow, Pi install (env + systemd unit + verify),
    Tailscale per-friend invite + revoke, ngrok demo discipline,
    daily SSH commands, agent-chip semantics, and troubleshooting
    for the issues we've actually hit (weak token, 503 on stream
    pre-bridge, Magic-DNS-only-on-tailnet, stale `localStorage`
    token after rotation, missing `GEMINI_API_KEY` on chat).
    `README.md` gets a short "Web dashboard" section pointing at the
    new doc; `PI_DEPLOYMENT.md` gets a "Web dashboard" subsection
    between the systemd watchdog and troubleshooting blocks. No
    code changes; pure docs + script.
- New tests: 16 in `tests/web/test_box_cache.py` (TTL, fade,
  thread-safety), 10 in `tests/util/test_frame_annotator.py`
  (round-trip, alpha fast-path, pixel sanity, robustness), 6 in
  `tests/vision/test_capture_preview.py` (publish-path integration
  with fake stream-buffer + box-source), 13 in
  `tests/web/test_routes_pages.py` (HTML shell incl. all view
  sections + nav links, static bundle, auth boundary), 2 added to
  `tests/web/test_routes_observations.py` (`dispatched=all`
  tri-state, 422 on garbage), 13 in
  `tests/web/test_routes_chat.py` (503 when unconfigured, auth
  wall, Pydantic validation, response shape, `tools_called` and
  `llm_available` passthrough, threadpool offload smoke check).
  303 total tests passing across web, util, vision (excluding
  torch-heavy modules), data, and labeler-auth suites. Pure
  additive change to `src/vision/capture.py` (+94 lines, 0
  deletions); 59 existing `tests/vision/test_capture.py` cases
  still pass.

---

  ### Fixed
- Orchestrator A/B rotation timer now fires unconditionally on schedule.
  Previously, when the LLM analyst path was active and the LLM consistently
  returned `switch_mode=None`, the timer-based rotation never fired, leaving
  the system stuck in a single mode. Observed 311-minute single-mode window
  in 2026-04-26 deployment versus the configured 30-minute rotation. Fix
  moves rotation to the top of `_run_cycle()` where it runs every cycle
  regardless of LLM availability.
- Removed duplicate `run()` method body in `ExperimentOrchestrator` left
  over from a prior merge.

---

### Fixed
- `VisualClassifier.from_config` no longer assigns `hailo_enabled` as a one-tuple — boolean checks against `hardware.yaml hailo.enabled` now work correctly.
- `BirdAgent._cycle` now propagates `detection_mode` from `CaptureResult` to `BirdObservation`, fixing field-tag tracking that always reported `fixed_crop`.

---

Phase 8 — Track 3 visual classifier retraining (PR #N feature/track-3-retraining)
Added

docs/investigations/track-3-retraining-2026-04-25.md — full
investigation document covering hypothesis, methodology, split
strategy, ablation plan (V0/V1/V2/V3), success criteria, risks, and
rollback plan.
notebooks/phase8_track3_training.py — feature extraction (cached to
track3_features_cache.npz, gitignored at 34MB) plus training of four
LogReg head variants. Each variant tunes C ∈ {0.01, 0.1, 1.0, 10.0, 100.0} on the deployment val set with class_weight="balanced". All
four bundles saved to models/visual/sklearn_pipeline_track3_v{0,1,2,3}.pkl.
notebooks/phase8_track3_evaluation.py — evaluates each variant on
both NABirds test (672 records, 19 species) and deployment test
(642 records, up to 23 classes depending on variant). Produces 8
confusion matrices, 4 per-class F1 plots, 4 classification reports,
one comparison_table.csv, and winner.txt + winner_rationale.md.
notebooks/track3_override_winner.py — one-off operational script.
Backs up the current production model to
sklearn_pipeline_v0_backup.pkl, deploys V2 to
sklearn_pipeline.pkl, writes the V2 selection rationale to
winner_rationale.md documenting the V1-vs-V2 decision.
tools/labeler/ui/make_deployment_splits.py — chronological 70/15/15
splits from verified_labels.jsonl by capture_timestamp.
Stratification check warns if any class has fewer than 3 records in
val or test. Produces data/splits/deployment_{train,val,test}.csv.
notebooks/results/phase8/track3_retraining/ — full evaluation
artifacts (confusion matrices, per-class F1 plots, classification
reports, comparison table, winner rationale). Tracked in git.
models/visual/sklearn_pipeline_track3_v{0,1,2,3}.pkl — all four
variants tracked (~220KB each, 10 .pkl files total at ~2.4MB).
Permits exact-numbers reproducibility without re-running training.
models/visual/sklearn_pipeline_v0_backup.pkl — preserved baseline
for rollback.
models/baselines/audio_knn_baseline.pkl — Phase 3 audio baseline
(110KB) now tracked for reproducibility of the phase 7 evaluation.
notebooks/classifier_retrain_experiment_2026-04-23.py — earlier
exploratory script preserved as methodology history.
configs/species.yaml — added CALT (California Towhee, Melozone
crissalis) entry under year-round residents.

Changed

models/visual/sklearn_pipeline.pkl — production model now serves V2
(20 classes, NABirds + deployment HOFI/SOSP/AMCR + CALT). Previous
baseline preserved at sklearn_pipeline_v0_backup.pkl for rollback.
Pi agent picks up new model on next git pull + systemctl restart.
.gitignore — added rules to exclude regenerable large files:
notebooks/results/**/*_features_cache.npz (34MB feature caches
produced by the training script, recreated in 30-60 minutes by
re-running) and models/baselines/visual_svm_baseline.pkl (537MB,
exceeds GitHub's 100MB limit, regenerable from
notebooks/visual_baseline.ipynb).

Findings (deployment data)

CALT classification fixed. V0 misclassified 100% of CALT records
as MOCH or MODO because CALT was not in the visual vocabulary. V2
correctly classifies 67/70 CALT test records (96% recall, per-class
F1 = 0.77). Confusion matrix off-diagonal CALT-as-MOCH/MODO drops
from baseline rate to under 2%.
Deployment macro F1 improvement. V0: 0.131 → V2: 0.736 (Δ +0.605).
NABirds preservation. V0 macro F1: 0.931 → V2: 0.921 (within the
0.05 tolerance threshold, no catastrophic forgetting on the 19
NABirds species).
V2 vs V1 selection. V1 had higher raw deploy_macro_f1 (0.776 vs
V2's 0.736) but was evaluated on only 184 of 642 deployment test
records (29%) — V1 has no class for CALT/NONE/UNKNOWN and excludes
those records from its evaluation entirely. V2 was evaluated on 254
records including CALT and is the larger-scope evaluation. V1 cannot
classify CALT, so deploying V1 would leave the headline problem
unfixed. Full reasoning in
notebooks/results/phase8/track3_retraining/winner_rationale.md.
UNKNOWN class did not learn. V3 attempted explicit NONE and
UNKNOWN classes for abstention. NONE worked (per-class F1 = 0.77).
UNKNOWN failed (F1 = 0.14) — only 84 training records and the
heterogeneous nature of the class (blur + multi-bird + weird angle)
prevented coherent representation learning. 33 of 58 UNKNOWN test
records were predicted as CALT. Recommendation for next iteration
is to drop UNKNOWN as a class and use threshold-based abstention via
the existing ScoreFuser logic instead. NONE may be retained as a
class in a future variant since it works.

Limitations

All 4280 verified records are from a single day (2026-04-24, ~21
hours of captures). Splits test ~2-hour temporal generalization, not
multi-day. A more robust evaluation will be possible after additional
days of deployment data accumulate.
AMCR test set has only 10 records; per-class F1 numbers for AMCR are
noisy (a single record difference shifts F1 by 0.1).
Deployment test set cam-skew is 14.5/85.5 (cam0/cam1) versus train
27.8/72.2. Slight evaluation bias toward cam1 conditions.
The evaluation script's automatic decision rule (best deploy_macro_f1
with NABirds tolerance) is flawed when variants evaluate on different
test set sizes. A revised rule for future Track-N evaluations would
require evaluation set size parity (e.g., variants must score on at
least 95% of the largest variant's eval set to qualify).

Verification

Laptop-side: V2 deployed and verified loading — 20 classes, C=0.1,
track3_variant=v2, val F1=0.842. VisualClassifier.from_config()
loads cleanly with the new label map.
Pi-side: Pending first post-merge git pull and systemctl restart avis-agent. Expected log line: "VisualClassifier loaded | classes=20".
Rollback path: cp models/visual/sklearn_pipeline_v0_backup.pkl models/visual/sklearn_pipeline.pkl then restart agent. < 5 minutes.

Notes

The track3_features_cache.npz (34MB) is intentionally gitignored —
it's a deterministic function of the input images and can be
regenerated by running phase8_track3_training.py (~30-60 min on
CPU). Tracking it would bloat the repo without aiding reproducibility.
The models/baselines/visual_svm_baseline.pkl (537MB) is also
gitignored — exceeds GitHub's per-file limit. The Phase 3 baseline
remains regenerable from notebooks/visual_baseline.ipynb.

Test count

No new tests in this PR (research-mode notebooks/scripts, not src/
changes). 123 existing tests still passing across the labeler module.
Future PR: add the bird-agent-side _label_map length test that
asserts the 20-class shape of the deployed pipeline.

---


### Added
- **Layer 2 — Labeling assistant review UI** (PR #N)
  - `tools/labeler/ui/` — token-authenticated FastAPI web UI for verifying
    Layer 1 pre-labels. Three-view SPA (queue, review, verified) with
    six-theme system (warm/pollen/mono × light/dark), keyboard shortcuts,
    touch gestures, and mobile-responsive layout.
  - `tools/labeler/ui/review_store.py` — in-memory store with atomic JSONL
    persistence (append on first verify, atomic rewrite on correction).
    Optimistic concurrency via `client_load_time` round-trip prevents
    silent overwrites when reviewing from multiple devices.
  - `tools/labeler/ui/auth.py` — `AVIS_WEB_TOKEN` middleware with
    `hmac.compare_digest` for timing-attack-resistant comparison. Token
    accepted via `X-Avis-Token` header or `?token=` query parameter.
  - `tools/labeler/ui/inspect_verified.py` — diagnostic script: schema
    validation, per-species distribution, OTHER breakdown, agreement-rate
    analysis, duplicate/orphan detection.
  - `tools/labeler/ui/inspect_unreviewed.py` — backlog analysis with
    agreement-rate cross-reference for bulk-action recommendations.
  - `tools/labeler/ui/README.md` — setup, daily workflow,
    NONE/UNKNOWN/OTHER distinctions, results-interpretation guide with
    worked example from the first reviewer session.
  - `docs/investigations/labeling-assistant-ui-2026-04-25.md` — full
    design rationale, architecture, success criteria, risks, rollback.
  - `tests/labeler/ui/` — 76 new tests (30 store, 15 auth, 31 routes).
  - 116 total Layer 2 tests passing; 123 across the labeler module.

- **Schema additions** (additive, backward-compatible) (PR #N)
  - `OTHER` sentinel in `tools/labeler/schema.py` — for confidently-
    identified out-of-vocabulary species.
  - `other_species_code` field — 4-letter custom code stored when
    `species_code == "OTHER"`. Pydantic validators enforce the
    `OTHER ↔ code` invariant and reject collisions with known codes.
  - 22 new tests in `tests/labeler/test_schema.py` covering validation,
    format constraints, and known-code collision detection.

- **Configuration**
  - `AVIS_WEB_TOKEN` documented in `.env.example`. Server refuses to
    start without a token of at least 16 characters.
  - `requirements.txt` — added `fastapi==0.118.0`,
    `uvicorn[standard]==0.32.0`, `python-multipart==0.0.18`,
    `httpx==0.28.1` (test client).

### Findings (deployment data)
- First reviewer pass: 4280 of 8276 captures verified (52% coverage).
- Visual species observed at the feeder during this period: AMCR
  (American Crow, 198 records, 100% pre-labeler agreement), SOSP
  (Song Sparrow, 1014 records, 96.4%), HOFI (House Finch, 284 records,
  69.9%), CALT (California Towhee, 521 records via OTHER — out-of-
  vocabulary, no native pre-labeler support).
- Nine species the pre-labeler proposed (MODO, MOCH, WREN, HOSP, OCWA,
  AMRO, HOORI, SPTO, EUST) had zero agreement across 535 records;
  appear to be visual hallucinations on out-of-distribution captures.
- Implication for Track 3 retraining: target intervention is adding
  CALT to the visual classifier vocabulary using the 521 reviewer-
  confirmed examples. Other vocabulary additions are not supported by
  this deployment data.

### Notes
- Zero changes to `src/` runtime code — production agent on the Pi is
  untouched.
- Bulk-confirm script (`tools/labeler/ui/bulk_confirm.py`) is included
  but was not run for this dataset; current 4280 records reflect
  manual review only.

---

### Added
- Upgraded camera capture from 1536×864 @ 120fps to 2304×1296 @ 30fps
  (IMX708 binned mode) to provide 2.25× more pixels per bird as input
  to the classifier. All `feeder_crop` coordinates rescaled 1.5× to
  preserve the real-world framing at the new pixel grid. Investigation
  rationale, hypothesis, and success criteria documented in
  `docs/camera-quality-2026-04-23.md`.

### Changed
- `configs/hardware.yaml`: `capture_width`/`capture_height`/`capture_fps`
  and all three `feeder_crop*` blocks updated to the new 2× coordinate
  system. Real-world feeder framing is preserved.
- `scripts/dev_config.py`: `PI_OVERRIDES` for `feeder_crop_cam0` and
  `feeder_crop_cam1` rescaled to match the new capture resolution.
- `tests/vision/test_capture.py`: module-level capture dimension constants
  updated to match new defaults. `TestAdaptiveYoloCrop` retained original
  1536×864 fixture dimensions (tests are resolution-independent math).

---

### Added
- Systemd watchdog integration for service self-healing. The orchestrator now
  emits `READY=1`, `WATCHDOG=1` per cycle, and `STOPPING=1` signals via
  sdnotify. Pairs with a systemd service override (`WatchdogSec=300`,
  `Restart=always`) to automatically restart the service if it stops
  heartbeating for 5 minutes. Graceful no-op when sdnotify is unavailable.
  Added `sdnotify>=0.3.2` to `requirements-pi.txt`. See
  `docs/deployment.md` → "Systemd watchdog" for Pi setup.

---

### Phase 8 — Bird-Presence Gate (Branch 2)

#### Added
- `BirdDetector` protocol in `src/vision/detector.py` with `CPUYOLODetector`
  implementation using ultralytics YOLOv8s on CPU. Runs between motion
  detection and species classification to filter out empty-feeder frames
  that previously got classified as noise.
- `GATE_REASON_*` string constants in `src/data/schema.py` (plain strings
  rather than StrEnum for maximum backward compatibility with serialized
  observations).
- `BirdObservation.gate_reason` field — records why a suppressed observation
  was suppressed. Values: `no_bird_detected`, `below_confidence_threshold`,
  `species_cooldown`. `None` when the observation was dispatched or when the
  record predates Branch 2.
- `CaptureResult.gate_passed`, `CaptureResult.gate_reason`,
  `CaptureResult.gate_confidence` fields — communicate gate state from
  `VisionCapture` to `BirdAgent`. `gate_passed` defaults to `True` for
  backward compatibility with code paths that don't run the gate.
- `configs/hardware.yaml: detector.*` block — selects detector backend
  (`cpu` or `hailo`) and per-backend settings (model path, confidence
  threshold, imgsz).
- `BirdAgent._log_gate_suppressed()` helper — synthesizes a sentinel
  `species_code="NONE"` observation when both cameras' gates block AND
  no audio detection fills the gap, so gate-blocked motion events are
  preserved in `observations.jsonl` for ablation analysis.

#### Changed
- `BirdAgent._cycle()` now checks `gate_passed` per camera before invoking
  the visual classifier. Pre-existing suppression paths (threshold, cooldown)
  now also populate `gate_reason` on the logged observation.
- `VisionCapture._process_frame()` runs the bird-presence gate after the
  motion gate but before classifier preprocessing. Gate failure produces a
  `CaptureResult` with `frame=None, gate_passed=False`.
- `ultralytics>=8.4.0,<9.0.0` added to `requirements.txt` and `requirements-pi.txt`.

#### Context
- Full investigation and design rationale:
  `docs/investigations/hailo-2026-04-22.md`.
- Follow-on branches: `fix/hailo-classifier-normalization` (Branch 3),
  `refactor/orchestrator-agentic-windows` (Branch 4),
  `feat/hailo-yolo-compilation` (Branch 5, deferred post-report).



### Changed
- `VisionCapture.__init__` now accepts a `hailo_enabled` parameter that controls
  shared VDevice creation, replacing the previous `detection_mode == "yolo"` gate.
  This decouples Hailo infrastructure availability from model-level decisions
  about which components use the NPU. Preserves all existing functionality;
  resolves the issue where `detection_mode: "fixed_crop"` could not share a
  VDevice with the Hailo classifier path. See
  `docs/investigations/hailo-2026-04-22.md` for full context.

---

### Phase 8 — Observation Logging for Full Classification Stream (PR #51 feat/log-suppressed-observations)

#### Added
- **`BirdObservation.dispatched: bool`** field on the schema, default `True`.
  When True: observation passed the confidence threshold and cooldown gates
  and was dispatched via the notifier. When False: observation was classified
  but suppressed due to sub-threshold confidence or active cooldown. Default
  True preserves backward compatibility — existing observations.jsonl records
  without the field deserialize as dispatched=True, matching historical
  reality where only dispatched observations were logged.
- **`Notifier.log_suppressed(observation)`** public method. Writes to the
  same `observations.jsonl` as `dispatch()` but marks `dispatched=False`
  and produces no user-facing side effects (no push, no webhook, no email,
  no print). Used by the agent for threshold and cooldown gate failures.
- **Three new test classes, 12 tests total**: `TestLogSuppressed` (5),
  `TestDispatchedField` (3), `TestCycleSuppressedLogging` (4) verifying
  file write behavior, dispatched marking, no side effects, media path
  preservation, immutability, backward compatibility, and agent wiring.

#### Fixed
- **Orphan observations** — During April 20 deployment, 5186 of 5624
  captured frames (92%) had no corresponding entry in observations.jsonl.
  Frames were captured, classified, and fused, but when
  `BirdAgent._cycle()` returned early from the confidence threshold or
  cooldown gate, the observation object was garbage-collected with only a
  DEBUG log line. This made it impossible to analyze visual classifier
  behavior on real bird events that were simply below the user-notification
  threshold (confidence 0.20 at testing level). PR #50's color and focus
  fixes could not be measured against the April 20 baseline because only
  the tip of the iceberg was in the log.

#### Changed
- **`BirdAgent._cycle()`** restructured to populate media paths
  (`audio_path`, `image_path`, `image_path_2`) BEFORE the threshold gate
  so suppressed observations retain image/audio references for later
  analysis. Threshold gate failure now calls `notifier.log_suppressed()`
  instead of returning None silently. Cooldown gate failure now calls
  `notifier.log_suppressed()` instead of returning None silently. Dispatch
  path unchanged — media paths still populated, `notifier.dispatch()`
  still called.
- **`Notifier.dispatch()`** now explicitly marks `dispatched=True` via
  `model_copy` before logging, so the logged record reflects reality.

#### Verification
- Laptop-side: 116 tests in `tests/notify/test_notifier.py` and
  `tests/agent/test_bird_agent.py` combined, all passing.
- Pi-side (April 21 03:58 PDT deploy): service active, new `dispatched`
  field present on every record, `dispatched=False` records confirmed in
  live observations.jsonl. In 30-minute indoor window post-deploy: 2
  dispatched + 110 suppressed = 112 classifications logged, vs. ~2 that
  would have been logged pre-PR. 55× increase in log visibility.

#### Backward compatibility
- `observations.jsonl` consumers that don't know about `dispatched` field
  see additional records with `dispatched: false`; they can ignore the
  field and treat all records as before
- `observation_tools.py` LLM agent tools can filter `dispatched=True` to
  preserve existing "user saw it" semantics
- Pre-PR records loaded from `observations.jsonl` deserialize with
  `dispatched=True` by default, matching their historical meaning

### Test count
- 616 passing, 6 deselected (hardware), CI green

---

### Phase 8 — Vision Color Format and Continuous Autofocus (PR #50 fix/vision-color-and-focus)

#### Fixed
- **BGR/RGB color channel swap** — picamera2's `RGB888` format returns
  bytes in B-G-R order in the numpy array, a long-standing libcamera
  convention. Every frame fed to the frozen EfficientNet (trained on
  NABirds RGB) had red and blue channels swapped. Confirmed by observing
  that orange feeder peels appeared blue in saved PNGs — the direct
  signature of an R↔B swap. Classifier was being given systematically
  miscolored inputs throughout Phase 5-8 evaluation.

- **Fixed infinity focus** — Pi Camera Module 3 autofocus was never
  configured. libcamera defaults to manual focus at lens position 0.0
  (infinity). Birds at the feeder sit ~30cm from the lens (macro range),
  well outside the fixed focus plane, producing soft captures throughout
  Phase 5-8 evaluation.

Together these explain the observation-log pattern of audio detecting
real birds at 0.80+ confidence while vision returns scene-floor
predictions on the same frames.

#### Changed
- **`src/vision/capture.py`** — `_open_cameras()`:
  - Switch format from `RGB888` to `BGR888` (true RGB memory layout —
    libcamera's `BGR888` string maps to R-G-B byte order in the numpy
    array, inverse of intuition)
  - Add `libcamera.controls.AfModeEnum.Continuous` on both cameras after
    configure
  - Graceful fallback if libcamera/AF unavailable (logs warning, continues
    with fixed focus)
  - Comprehensive docstring explaining both gotchas for future contributors

#### Verification
- Laptop-side: all existing tests in `tests/vision/test_capture.py` pass.
- Pi live validation prior to merge: `test_pr50_live.py` against both
  cameras — both opened successfully, `AfMode.Continuous` accepted by
  both sensors, autofocus confirmed active (cam0 LensPosition observed
  moving 2.20 → 2.07 → 1.80 across 3 frames — lens physically hunting).
  Visual color inspection of saved PNGs shows true-to-life colors.
- Post-merge production validation (April 21 03:17 PDT): journalctl
  confirms `Continuous autofocus enabled on cam0`, `Continuous autofocus
  enabled on cam1`, `Both cameras opened and started`. Service stable.

#### Known limitations
- Does NOT address crop region coverage. Static per-camera crops
  (`feeder_crop_cam0` = `{x:630, y:130, w:700, h:580}`,
  `feeder_crop_cam1` = `{x:420, y:130, w:700, h:580}`) remain fixed.
  Birds that land outside the crop are invisible to the classifier
  regardless of color or focus correction. Phase 8+ option:
  bird-detection-gate architectural change making YOLO always-on rather
  than A/B.

#### Expected impact
- EfficientNet trained on RGB now receives RGB (previously received BGR).
  Feature extraction should align with training distribution.
- Feeder-distance birds now in focus plane. Feather patterns, head shape,
  eye rings legible for classifier.
- Empirical validation pending April 21 daylight bird activity.

### Test count
- 616 passing (unchanged), 6 deselected, CI green

---

### Phase 8 — Notifier Network Resilience (PR #49 fix/notifier-timeout-and-retry)

#### Fixed
- **Pushover push failures on marginal WiFi** — initial notifier used a fixed
  5-second timeout in `_push()` with no retry logic. Field testing on April 20
  with home WiFi at -72 dBm signal and 11-22 KB/s measured upload speeds
  produced a 100% failure rate on image-attached pushes: at ~15 KB/s, a 500KB
  capture frame takes ~35 seconds to upload, but the 5s timeout fires first.
  Between 17:19 and 17:44 PDT, 14 consecutive push notifications failed with
  `urlopen error The write operation timed out` — including the peak HOFI
  detection at 80.7% fused confidence during a 10-minute feeder visit. All 14
  observations were correctly classified and persisted to `observations.jsonl`
  with valid `image_path` entries pointing to captured frames on disk — only
  the network upload to Pushover failed. Text-only pushes (`_push_text` at
  ~500 bytes with a 10s timeout) succeeded intermittently, masking the
  problem in casual use.

#### Added
- `src/notify/notifier.py` — new private `_post_to_pushover(data, content_type, *, context) -> bool`
  helper method used by both `_push()` and `_push_text()`. Features:
  - **Payload-scaled timeout**: `timeout = base + (payload_kb * per_kb_multiplier)`.
    Default 10s base + 0.1s/KB yields 60s for 500KB images, 10s for small
    text pushes. Ratio targets a conservative 10 KB/s upload floor.
  - **Retry with exponential backoff**: 3 attempts by default, spacing
    `backoff * 2^(attempt-1)` seconds between them — 2s / 4s / 8s at
    default base=2.0.
  - **Failure-mode discrimination**: retries on transient network errors
    (`OSError`, `urllib.error.URLError`, `TimeoutError`). Does NOT retry
    on Pushover API-level rejections (`status != 1`) — those won't recover
    on retry and would just hammer the API.
  - **Observability**: each attempt logs distinct lines. Success logs
    payload size, timeout used, and retry count when succeeded on retry ≥ 2.
- `src/notify/notifier.py` — four new constructor parameters with defaults:
  `push_base_timeout_seconds: float = 10.0`, `push_per_kb_timeout_seconds: float = 0.1`,
  `push_max_attempts: int = 3`, `push_retry_backoff_seconds: float = 2.0`.
  Stored as `self.` attributes and consumed by the helper.
- `configs/notify.yaml` — new `push.network` block with all four knobs and
  extensive comments explaining payload scaling, retry semantics, and
  failure-mode discrimination. `from_config()` reads the block with defaults.
- `tests/notify/test_notifier.py` — new `TestNetworkResilience` class with
  6 tests covering: timeout-scales-with-payload-size, retries-on-timeout,
  success-after-retry, no-retry-on-API-rejection, exponential-backoff-timing,
  end-to-end `_push()` with image attachment recovers from transient failure.

#### Changed
- `src/notify/notifier.py` — `_push()` and `_push_text()` refactored to call
  `_post_to_pushover()` instead of maintaining duplicate inline
  `urllib.request.urlopen(...)` blocks. Removes ~40 lines of duplicated
  error handling; both push paths now share identical resilience mechanics.
- `src/notify/notifier.py` — ruff auto-fix: `socket.timeout` → `TimeoutError`
  alias (UP041) in the exception handler tuple.

#### Verification
- Laptop-side: 71 tests in `tests/notify/test_notifier.py` (65 existing + 6
  new), 616 tests across full suite, 0 failures, 0 new warnings. CI green.
- Pi-side (April 20 19:14 PDT deploy):
  - Startup text push delivered: `Pushover push sent: system text push | 0.2KB, timeout=10.0s`
  - Production image-push validation with threshold temporarily dropped to
    `0.005` to force dispatches: **5 consecutive 520KB image pushes succeeded
    in 52 seconds** (19:24:22 AMCR, 19:24:31 HOFI, 19:24:40 WBNU, 19:25:09
    AMCR, 19:25:14 WBNU), all with `timeout=62.3s` scaled payload. Zero
    retries fired, meaning first-attempt uploads completed within budget on
    the same WiFi that had 14 consecutive failures 4 hours earlier.
  - Threshold reverted to 0.2 and service confirmed active.
- Config merge conflict between Pi-local `channels.push: true` override and
  committed `push.network` block was resolved manually via `nano` (stash →
  pull → stash pop → resolve markers → `git stash drop`). YAML re-validated
  with `python -c "import yaml; yaml.safe_load(...)"` before restart.

#### Backward compatibility
- All new constructor parameters have defaults → existing `Notifier(...)`
  callers unchanged.
- `push.network` block is optional in `notify.yaml` → existing deployments
  without it use constructor defaults.
- No public API changes visible outside the notifier module.

#### Follow-ups opened (future PRs, not blocking)
- Config drift pattern: any PR that structurally changes `notify.yaml` /
  `hardware.yaml` / `thresholds.yaml` will conflict with Pi-local overrides.
  `scripts/dev_config.py` handles re-applying overrides after a clean pull
  but does not resolve merge conflicts. Future option: `configs/*.local.yaml`
  overlay pattern, gitignored, merged at `from_config()` load time.
- Add YAML-validation step to `pi-deploy` shortcut (run `yaml.safe_load`
  before `systemctl restart`) — would have caught the merge-marker
  corruption in 1s instead of a 30s restart-loop.
- Pi-deploy observations expected overnight: first "succeeded on attempt 2/3"
  line will confirm the retry logic engaging (rather than just the scaled
  timeout carrying everything).

### Test count
- 616 passing, 6 deselected (hardware), CI green

---

### Fixed (Phase 8)
- **Hailo YOLO mode: fix `HAILO_STREAM_NOT_ACTIVATED(72)` on shared VDevice** — the shared Hailo VDevice created by `VisionCapture` for YOLO detection was instantiated without a scheduling algorithm. With the HailoRT 4.23.0 `InferModel` API, this causes every inference call to fail with `HAILO_STREAM_NOT_ACTIVATED(72)` and write zeros to the output buffer, masking all bird detections. The fix is to create the VDevice with `HailoSchedulingAlgorithm.ROUND_ROBIN`, matching the pattern already used in `src/vision/hailo_extractor.py` and `scripts/benchmark_hailo.py`. A new `_create_shared_vdevice()` helper in `src/vision/capture.py` centralizes this requirement so any future code needing a shared VDevice gets the correct configuration by default. Also removes a duplicated preprocessing block in `HailoDetector.detect()` that was a leftover from an earlier edit.
- **Hailo YOLO mode: remove manual `activate()`/`deactivate()` under scheduler** — follow-up to the ROUND_ROBIN fix above. Once the VDevice switched to scheduler-managed activation, the manual `self._configured.activate()` call in `HailoDetector.open()` started raising `HAILO_INVALID_OPERATION(6)` with the message "Manually activate a core-op is not allowed when the core-op scheduler is active!". Under a ROUND_ROBIN scheduler the chip handles activation and deactivation at inference time, so `HailoDetector.open()` now only configures the model and `HailoDetector.close()` releases the reference without calling `deactivate()`. This matches the existing pattern in `HailoVisualExtractor`. Reproduced on Pi hardware: prior error was swallowed by the detector's try/except and the service fell back to `fixed_crop` silently, so YOLO mode appeared to load but immediately deactivated itself every cycle.

---

### Phase 8 — Audio device lookup by name (fix/audio-device-lookup-by-name)

#### Added
- `src/audio/capture.py` — `AudioCapture` now resolves the sounddevice
  index by name substring (`device_name`) with fallback to `device_index`.
  Lookup happens lazily on first `capture_window()` call, cached for
  subsequent captures. New private `_resolve_device_index()` method
  encapsulates the full resolution chain with descriptive error messages
  listing available devices when nothing resolves.
- `configs/hardware.yaml` — new optional `microphone.device_name` key
  (defaults to `"USB PnP Audio Device"`). Committed `device_index` changed
  from `1` to `0` to match the deployed Pi hardware.
- `tests/audio/test_capture.py` — new file, 32 unit tests across 7 classes
  covering initialization, `from_config()`, device resolution (name match,
  substring match, first match wins, input-channel filtering, fallback
  paths, out-of-range errors), energy gate (below/above/near threshold),
  WAV output (file creation, timestamp format, mono int16), device
  resolution caching (resolves once, reuses cached index), and error
  handling (sounddevice not installed, recording failures). Mocks
  `sounddevice` via `patch.dict(sys.modules, ...)` since capture.py
  imports the module lazily inside function bodies.

#### Fixed
- Fifine USB mic sounddevice index was non-deterministic across reboots
  and USB replug events. On the April 19 deployment, the index silently
  shifted from 1 to 0, causing every audio capture cycle to return
  `audio_result: null` for hours before anyone noticed. PR-A's override
  tool (`scripts/dev_config.py`) was a workaround — this PR is the
  structural fix. Audio capture now works regardless of enumeration order,
  across reboots, on different Pi hardware, and when new USB audio devices
  (future webcams with built-in mics, etc.) are plugged in.

#### Changed
- `docs/PI_DEPLOYMENT.md` — "Microphone not capturing" troubleshooting
  entry rewritten to reflect the new by-name resolution. Shows how to
  read the new startup log lines (`Audio device resolved by name: ... → index N`)
  and gives three decision branches based on what the logs report. Old entry
  pointed at this PR as a future fix; new entry treats by-name lookup as
  the default and documents how to diagnose remaining edge cases.

#### Verification
- Laptop-side: 610 tests passing, 6 deselected (hardware-only), full
  suite including 32 new `test_capture.py` tests and unchanged existing
  578 tests.
- Pi-side: deferred to first post-merge `pi-deploy`. Will verify via
  startup log line (`Audio device resolved by name: 'USB PnP Audio Device'
  → index 0`) and continued RMS values in `pi-logs` confirming audio
  capture unchanged from current production behavior.

#### Follow-up — can be removed after this lands
- The hand-edited `microphone.device_index: 0` on the Pi's local
  `configs/hardware.yaml` is no longer needed — the committed default is
  now `0`, and even if it weren't, the by-name lookup would find the
  Fifine correctly.

---

### Phase 8 — Pi Tooling Recovery (fix/recover-pi-tooling)

---

### Phase 8 — Pi Tooling Recovery (fix/recover-pi-tooling)

#### Recovered and versioned Pi deployment tooling
- `pi.ps1` — Laptop-side PowerShell shortcuts for managing the deployed Pi
  via SSH. Dot-source into any session with `. .\pi.ps1`. Functions:
  `pi-ssh`, `pi-status`, `pi-logs`, `pi-logs-since [time]`, `pi-stop`,
  `pi-start`, `pi-restart`, `pi-run [seconds]`, `pi-config-check`,
  `pi-pull`, `pi-deploy`. Host is configurable via `$env:AVIS_PI_HOST`
  with a fallback to the LAN mDNS hostname `birdfeeder01@birdfeeder.local`
  — no hardcoded IP addresses in the repo.
- `scripts/install_service.sh` — one-command Pi systemd setup. Previously
  lived only on the Pi; now committed with executable bit preserved.
- `docs/PI_DEPLOYMENT.md` — consolidated Pi deployment guide covering
  first-time setup, laptop-side `pi.ps1` configuration, SSH key auth, the
  config override model, daily workflow, and troubleshooting entries for
  every issue we've hit during Phase 5–8 deployment (YAML corruption,
  audio device shift, Hailo errors, mDNS resolution).
- `docs/SETUP.md` — Pi Deployment section replaced with a pointer to
  `PI_DEPLOYMENT.md` plus a quick-reference for the daily deploy command
  and feeder crop tuning workflow. Removed stale `nano` instructions that
  referenced incorrect YAML keys.

#### Replaced `dev_config.sh` with Python-based override tool
- `scripts/dev_config.py` — new Python rewrite using `yaml.safe_load` /
  `yaml.safe_dump` to apply Pi-local config overrides by real key path.
  Declarative `PI_OVERRIDES` list at the top of the file is the only thing
  anyone needs to edit when overrides change. Backs up each config to
  `configs/*.yaml.bak` before modifying, validates all configs parse
  cleanly after application, and exits non-zero with a clear error message
  on any failure. Idempotent: safe to run multiple times.
- `scripts/dev_config.sh` — deleted. `pi-pull` now calls the Python script.
- `.gitignore` — added `configs/*.yaml.bak` rule so auto-generated backups
  never land in commits.

#### Bugs fixed (silent failures in the old `dev_config.sh`)
- `s/threshold: 0.70/threshold: 0.10/` never matched anything. The actual
  key is `confidence_threshold`, not `threshold`. Every `git pull` on the
  Pi had silently left the dispatch threshold at the committed default
  of 0.70 rather than the 0.10 we thought was being applied. Observation
  logs captured during this period reflect threshold 0.70, not 0.10.
- `s/enabled: false/enabled: true/` matched broadly. It only affected
  `hailo.enabled` because that was the only `enabled: false` in the file,
  but any future config block with the same pattern would have been
  silently flipped too. New tool targets the exact key path.
- Per-camera crop overrides (`feeder_crop_cam0`, `feeder_crop_cam1`) were
  not handled at all — the multi-line YAML block couldn't be managed with
  sed. They had to be manually uncommented after every `git pull`. Now
  baked in as structured override values.

#### Verification status
- Laptop-side: `dev_config.py` tested against committed `configs/*.yaml`
  — all seven overrides applied cleanly, backups written, configs still
  parse after application, `git checkout configs/` restores clean state.
- `pi.ps1` loads without errors, banner prints, `pi-status` verified
  end-to-end against deployed Pi (systemd `active (running)` confirmed,
  live log tail rendered correctly).
- Pi-side: `dev_config.py` hardware verification happens on first post-
  merge pull. Old `dev_config.sh` remains on the Pi until that point.

### Test count
- No new tests in this PR (tooling-only change). Existing 578 still pass.

---

### Phase 8 — Live Deployment Tuning (feat/per-camera-crop)

#### Per-camera crop override
- `src/vision/capture.py` — added `crop_x_cam1`, `crop_y_cam1`,
  `crop_width_cam1`, `crop_height_cam1` params to `__init__()` with
  `None` defaults falling back to shared crop values. `from_config()`
  reads optional `feeder_crop_cam0` and `feeder_crop_cam1` blocks from
  `hardware.yaml`, falling back to shared `feeder_crop` if absent.
  `_process_frame()` selects crop region by `camera_index` — cam0 uses
  primary crop, cam1 uses override crop.
- `configs/hardware.yaml` — added `feeder_crop_cam0` and
  `feeder_crop_cam1` optional override blocks with comments. Deployed
  values tuned during live calibration session 2026-04-19:
  cam0 x:630, cam1 x:420, y:130, 700×580.
- Backward compatible — existing deployments without per-camera keys
  use shared `feeder_crop` unchanged.

### Test count
- TBD — existing capture tests pass, new per-camera tests needed

---

### Phase 7 — Held-out Evaluation + Pi Autonomous Deployment (PR feat/phase7-evaluation)

#### Final evaluation on held-out test set
- `notebooks/phase7_evaluation.ipynb` — complete evaluation on data never
  touched during training, validation, or hyperparameter selection.
  Fixed KNN feature extraction (preprocess_file returns spectrograms not
  raw MFCCs — was producing 256-dim vectors instead of 80-dim), fixed SVM
  feature extraction (reads HOG params from bundle not hardcoded values),
  fixed experiments.csv append to 14-column schema.
- `notebooks/results/phase7/` — 7 evaluation artifacts:
  audio_birdnet_confusion_matrix.png, audio_birdnet_per_class_f1.png,
  visual_efficientnet_confusion_matrix.png, visual_efficientnet_per_class_f1.png,
  model_comparison_table.csv, ablation_dataset_size.png,
  fusion_weight_sensitivity.png
- `notebooks/results/experiments.csv` — 15 rows total, 5 new Phase 7
  held-out rows appended (KNN, BirdNET, SVM, EfficientNet, fused)
- `notebooks/audio_baseline.ipynb` — re-run to regenerate
  audio_knn_baseline.pkl, baseline evaluation artifacts frozen
- `notebooks/visual_baseline.ipynb` — re-run to regenerate
  visual_svm_baseline.pkl, baseline evaluation artifacts frozen

#### Results (held-out test set — unbiased final estimates)
- Audio KNN (MFCC mean+std):          macro F1 = 0.012  n=86
- Audio BirdNET pretrained:           macro F1 = 0.776  n=86   (67× KNN)
- Visual SVM (HOG + color hist):      macro F1 = 0.118  n=672
- Visual Frozen EfficientNet+LogReg:  macro F1 = 0.931  n=672  (7.9× SVM)
- Fused BirdNET+EfficientNet:         macro F1 = 0.945  coverage=96%
- Fusion weight sensitivity: optimal audio=0.05, F1=0.974
- Dataset size ablation: F1 flattens above 50% training data — pretrained
  features dominate, not dataset size

#### Pi autonomous deployment
- `scripts/avis.service` — installed to /etc/systemd/system/ on Pi
  sudo systemctl enable avis confirmed — starts automatically on every boot
  Hardware verified April 18 2026: active (running), Gemini calling,
  cameras open, detections firing within 10s of power-on
- `scripts/install_service.sh` — one-command Pi systemd setup script
- `pi.ps1` — PowerShell dot-source file for laptop Pi management.
  pi-ssh, pi-status, pi-logs, pi-stop, pi-start, pi-restart,
  pi-run (smoke test), pi-pull, pi-deploy

#### BaselineOptimizer stub
- `src/agent/baseline_optimizer.py` — AutoML agent stub, architecture
  fully documented. Targets OpenClaw framework for long-running agentic
  loops. Perceive→reason→act→memory over feature/hyperparameter search space.
  NotImplementedError on all public methods until Phase 8.
- `tests/agent/test_baseline_optimizer.py` — 3 tests covering stub behavior

### Test count
- 578 passing, 6 deselected (hardware), CI green

---

### Phase 6+ — Agentic LLM Layer (PR #42 feat/agentic-llm-layer)

#### Dual-agent architecture
- `src/agent/bird_analyst_agent.py` — BirdAnalystAgent: custom Gemini tool-calling agent
  via langchain-google-genai. advise() path called by orchestrator every 30min,
  answer() path for reactive user queries. Every cycle logged to analyst_decisions.jsonl.
  Graceful fallback: returns None when LLM unavailable, orchestrator falls back to
  fixed schedule.
- `src/agent/langchain_analyst.py` — LangChainAnalyst: LangGraph ReAct agent with
  3 memory layers (conversation buffer K=10, entity store, session tool cache).
  get_graph_diagram() returns Mermaid diagram of perceive→reason→act→memory state machine.
- `src/agent/tools/` — 14 shared tools (framework-agnostic):
  observation_tools.py (perceive), system_tools.py (perceive),
  action_tools.py (act), calibration_tools.py (self-tune).
  build_langchain_tools() adapter in langchain_tools.py injects runtime context.
- `src/agent/experiment_orchestrator.py` — ExperimentOrchestrator: autonomous Pi
  system controller. Boot notification, LLM advise() path, fixed-schedule fallback,
  daily .md/.json summary dispatch. Entry point for systemd boot via main().
- `src/notify/report_builder.py` — ReportBuilder: DailySummaryReport and
  ExperimentWindowReport from observations.jsonl. Outputs .md and .json.
- `src/notify/notifier.py` — added _push_text() for system-level plain-text push
- `src/data/schema.py` — detection_mode field on BirdObservation for A/B tracking
- `scripts/avis.service` — systemd unit for Pi boot autostart
- `configs/hardware.yaml` — orchestrator: and llm: config blocks added
- `requirements.txt` — langchain-core, langchain-google-genai, langgraph, langchain
  (google-generativeai removed — protobuf conflict with tensorflow-cpu)

#### Agent self-calibration
- Calibration tools close the autonomous loop: agent observes declining confidence,
  runs fusion weight sweep, applies better weights to thresholds.yaml autonomously.
  No human intervention required.

#### Hardware validation (Pi, April 17 2026)
- timeout 60 python -m src.agent.experiment_orchestrator confirmed:
  cameras open, Gemini called successfully, detection logged, autonomous feeder
  alert pushed ("Feeder activity dropped 96% over 3 days")
- LLM path: analyst=True | llm=True confirmed on Pi with gemini-2.5-flash

#### Notebooks
- notebooks/agent_demo.ipynb — presentation demo, USE_SYNTHETIC toggle,
  both agents running with live LLM, calibration charts, memory state visible
- notebooks/phase7_evaluation.ipynb — held-out test set evaluation scaffold

### Test count
- 575 passing, 6 deselected (hardware), CI green

---

### Phase 6 — YOLO detection pipeline (PR #40)

#### HailoDetector — YOLOv8s bird detection on HAILO8L
- `src/vision/hailo_detector.py` — new `HailoDetector` class wrapping
  YOLOv8s HEF (pre-installed at /usr/share/hailo-models/yolov8s_h8l.hef).
  Accepts full 1536×864 frames, resizes to 640×640 internally, decodes NMS
  output buffer, returns Detection(x1,y1,x2,y2,confidence,class_id) in
  original frame coordinates. NMS buffer: 80 classes × (4 + max_proposals × 20)
  bytes. Count field per class is float32 not uint32 — verified on hardware.
  PIL fallback for frame resize when cv2 unavailable (CI).
- `src/vision/capture.py` — adds detection_mode param ("fixed_crop"|"yolo")
  read from hardware.yaml hailo.detection_mode. Motion gate always uses
  fixed_crop for efficiency. In yolo mode: YOLO runs on full frame, falls back
  to fixed_crop if no bird detected. HailoDetector lazy loaded, closed in
  stop(). CaptureResult gains detection_mode and detection_box fields.
- `tests/vision/test_hailo_detector.py` — 35 unit tests (3 hardware
  deselected in CI), NMS buffer decoder tested with synthetic buffers
  matching exact Hailo YOLOv8 output format verified on Pi hardware.
- `configs/hardware.yaml` — detection_mode: fixed_crop (safe committed
  default), yolo_hef path, yolo score/proposal/confidence thresholds.
- `requirements.txt` — numpy pinned to 1.26.4 (numpy 2.x breaks
  torch.from_numpy on Python 3.11). opencv moved to requirements-pi.txt only.
- `requirements-pi.txt` — opencv-python==4.10.0.84 added for frame resizing.

#### Hardware validation (Pi, April 15 2026)
- YOLO running each cycle: "Camera 0: YOLO no bird — falling back to fixed_crop"
- Clean shutdown: "HailoDetector closed" confirmed, no segfault
- Notifications firing with image attachments confirmed

### Phase 6 — Shared Hailo VDevice (PR #41)

#### VDevice conflict fix — YOLO + EfficientNet both on NPU
- `src/vision/capture.py` — VisionCapture creates one shared VDevice eagerly
  in __init__ when detection_mode=yolo. Adds get_shared_vdevice() accessor.
  stop() releases shared VDevice after detector and cameras are closed.
- `src/vision/hailo_detector.py` — accepts optional shared_vdevice param.
  open() uses it instead of creating a new VDevice. close() only releases
  VDevice if it owns it.
- `src/vision/hailo_extractor.py` — accepts optional shared_vdevice param.
  open() uses it instead of creating its own. close() only releases if it
  owns it.
- `src/vision/classify.py` — VisualClassifier.__init__ and from_config()
  accept shared_vdevice param, passed to HailoVisualExtractor on first
  _load_hailo() call.
- `src/agent/bird_agent.py` — from_config() passes
  vision_capture.get_shared_vdevice() to VisualClassifier.from_config().

#### Hardware validation (Pi, April 15 2026)
- Both YOLO and EfficientNet confirmed running on NPU simultaneously
- Log confirmed: "Shared Hailo VDevice created (YOLO mode)"
- Log confirmed: "Visual predict: backend=hailo" on both cameras
- Log confirmed: "Shared Hailo VDevice released" on clean shutdown
- No HAILO_OUT_OF_PHYSICAL_DEVICES(74) errors

### Test count
- 443 passing, 0 failing, CI green

---

### Phase 6 — Hailo visual classifier wiring (PR #39)

#### VisualClassifier Hailo integration
- `src/vision/classify.py` — adds `hailo_enabled` and `hailo_hef_path` params
  to `__init__()`. `from_config()` reads `hailo.enabled` and
  `hailo.models.visual_hef` from `configs/hardware.yaml` automatically.
  New `_load_hailo()` method attempts to open `HailoVisualExtractor` on first
  `predict()` call when enabled, falling back silently to CPU PyTorch path if
  `hailo_platform` is unavailable or HEF is missing. `predict()` routes to
  Hailo or CPU — both paths produce identical `(1, 1280) float32` features
  for the sklearn LogReg head. `BirdAgent` and `ScoreFuser` are unaware of
  which backend is active.
- `configs/hardware.yaml` — fixes `hailo.models.visual_hef` path from
  `models/hailo/` to `models/visual/` (correct location of compiled HEF on
  Pi). Sets `hailo.enabled: false` as safe committed default — set `true`
  locally on Pi only, never committed (same pattern as `push: false`).

#### Hardware validation (Pi, April 15 2026)
- Full agent run confirmed with Hailo active: both cameras opened, audio
  capturing, Hailo HEF loaded and VDevice ready on first predict() call.
- Log confirmed: `Hailo inference active — HEF loaded from
  models/visual/efficientnet_b0_avis_v2.hef`
- Isolated inference test confirmed: backend=hailo, species prediction
  returned (WBNU), clean shutdown with no segfault.

#### Tests
- All 408 existing tests pass — Hailo path only activates when
  `hailo_enabled=True` and HEF exists, neither of which is true in
  the test environment.

### Test count
- 408 passing, 0 failing, CI green

---

### Phase 6 — Hailo HAILO8L hardware inference benchmark

#### Hailo EfficientNet-B0 compilation and deployment (this PR)
- `src/vision/hailo_extractor.py` — `HailoVisualExtractor` class wrapping
  HailoRT InferModel API. Accepts (224, 224, 3) uint8 frames, returns
  (1, 1280) float32 features for the existing sklearn LogReg head. Requires
  `HailoSchedulingAlgorithm.ROUND_ROBIN` for correct output from HailoRT 4.23.0.
  Falls back gracefully when `hailo_platform` unavailable (laptop/CI).
- `configs/hardware.yaml` — updated `hailo.models.visual_hef` to point to
  compiled `models/visual/efficientnet_b0_avis_v2.hef`.
- `scripts/benchmark_hailo.py` — reproducible latency benchmark: CPU 82.60ms,
  Hailo ResNet-50 raw 0.25ms (332×), Hailo EfficientNet-B0 13.04ms (6.3×).
- `scripts/compile_hailo_hef.py` — documents full compilation pipeline:
  ONNX export, calibration data export, DFC Docker steps, SE block avgpool
  shift delta fix, and Pi deployment instructions.
- `notebooks/hailo_benchmark.ipynb` — three-part benchmark narrative with
  live Pi results. Charts saved to `notebooks/results/`.
- `notebooks/results/experiments.csv` — 3 new Phase 6 rows appended.

#### Key technical findings
- EfficientNet-B0 compiled to HEF via Hailo DFC 3.32.0 in Docker (WSL2).
- HEF compiled for HailoRT 4.22.0 loads and runs correctly on 4.23.0
  (forward compatibility confirmed).
- SE block avgpool shift delta error resolved with model script:
  `pre_quantization_optimization(global_avgpool_reduction, division_factors=[7,7])`
- `ROUND_ROBIN` scheduling required — without it, HailoRT 4.23.0 returns
  `HAILO_STREAM_NOT_ACTIVATED(72)` and fills output buffer with zeros.

#### Tests
- `tests/vision/test_hailo_extractor.py` — 15 unit tests (mocked for CI),
  3 hardware integration tests marked `@pytest.mark.hardware`.

#### Hardware validation
- Benchmark confirmed on Pi (HAILO8L firmware 4.23.0):
  CPU baseline 82.60ms, Hailo EfficientNet-B0 13.04ms = 6.3× speedup.

### Test count
- 408 passing, 0 failing, CI green

---

### Phase 6 — Notification polish

#### Push image attachment (PR #36)
- `src/notify/notifier.py` — `_push()` now sends captured frame as multipart
  attachment when `push.attach_image` is true in `notify.yaml` and a valid
  image file is available on the observation. Selects best available frame
  (`image_path` then `image_path_2` fallback). Falls back to text-only
  silently if file is missing, too large, or unreadable. Audio-only
  detections include a note in the message body.
- `src/notify/notifier.py` — Add `_build_multipart()` module-level helper
  for multipart/form-data encoding. No external dependencies — uses stdlib
  `uuid`, `io`, `mimetypes` only.
- `src/notify/notifier.py` — Extend `__init__` with `push_attach_image`
  and `push_max_attachment_bytes` parameters. `from_config()` reads new
  `push:` block from `notify.yaml`.
- `configs/notify.yaml` — Add `push:` config block with `attach_image: true`
  and `max_attachment_bytes: 2500000`. Mirrors existing `webhook:` block.

#### Vision capture fix (PR #36)
- `src/vision/capture.py` — `_save_frame()` now saves the cropped frame
  (400×400px, 50–200KB) instead of the full-resolution frame (1536×864px,
  1–3MB). Keeps attachments well within Pushover's 2.5MB limit and
  represents exactly what the classifier saw.
- `src/vision/capture.py` — `output_dir` resolved to absolute path at
  construction time via `Path.resolve()`. `image_path` on every
  `CaptureResult` is now always absolute regardless of working directory.
  `raw_frame` preserved in memory for Phase 6 stereo estimation.

#### Tests
- `tests/notify/test_notifier.py` — Add `TestBuildMultipart` (8 tests) and
  `TestNotifierPushAttachment` (8 tests). Update `_make_notifier()` helper
  with `push_attach_image=False` default. Add 3 new `TestFromConfig` tests.
- `tests/vision/test_capture.py` — New file, 42 tests covering `__init__`,
  `from_config`, `_save_frame`, `_compute_motion`, `_update_background`,
  `_process_frame`, and `CaptureResult`. Zero hardware dependencies.

#### Hardware validation
- Pushover notifications confirmed delivered with attached cropped frame
  on Pi deployment. Three species detected during validation session:
  White-breasted Nuthatch (74%, 79%), Black Phoebe (99%).

### Test count
- 396 passing, 0 failing, CI green

---

### Changed
- `notebooks/results/phase5/` - moved phase5 result images into dedicated subfolder

---

### Hardware deployment (Phase 5 complete)
- Pi 5 running Debian Trixie (Python 3.13.5) with Hailo AI HAT+ confirmed
- Dual IMX708 cameras (indices 0 and 1, 1536×864) confirmed via rpicam-hello
- Fifine USB mic (sounddevice index 1, 48kHz) confirmed
- Two-venv architecture: Python 3.13 venv for picamera2/visual pipeline,
  pyenv Python 3.11 for BirdNET/tflite_runtime subprocess bridge
- `scripts/audio_inference.py` — standalone BirdNET inference script for
  Python 3.11 subprocess bridge (PR #30)
- System validated live — fused detections confirmed:
  Black Phoebe 100%, Mourning Dove 90%, House Finch 92%, House Sparrow 79%
- Pending: camera physical mounting, feeder crop tuning, Hailo compilation

---

## [0.5.0] — Phase 5 Hardware Deployment (Software Complete)

### Added
- `src/audio/capture.py` — `AudioCapture`: Fifine USB mic (device index 1,
  48kHz native, no resampling), 3-second windows, RMS energy gate discards
  silent frames before BirdNET inference (PR #26)
- `src/vision/capture.py` — `VisionCapture`: dual Pi Camera Module 3 via
  picamera2, simultaneous capture, rolling background motion gate, feeder
  crop ROI applied before 224×224 downsampling, saves raw frames to
  `data/captures/images/` (PR #26)
- `src/vision/stereo.py` — `StereoEstimator` Phase 6 stub: full interface
  defined (calibrate, estimate, _rectify, _compute_disparity,
  _disparity_to_depth), all methods raise `NotImplementedError` until
  Phase 6 stereo calibration (PR #26)
- `configs/hardware.yaml` — Pi hardware constants: mic device index, sample
  rate, camera indices, capture resolution, feeder crop zone (x, y, w, h),
  stereo baseline, Hailo device address (PR #26)
- `src/notify/notifier.py` — `_push()` implemented via Pushover API (urllib,
  no SDK); credentials from `.env`; graceful degradation when credentials
  missing; `_webhook()` Phase 6 stub for future web app backend (PR #26)
- `src/notify/notifier.py` — `enable_webhook`, `webhook_url`,
  `webhook_timeout_seconds`, `webhook_auth_header` parameters and
  `notify.yaml` webhook config block (PR #26)
- Push notification confirmed working on device — House Finch, 87%
  confidence (PR #26)

### Changed
- `src/audio/classify.py` — `AudioClassifier` updated to BirdNET inference
  via birdnetlib (F1=0.776 vs F1=0.089 CNN from scratch). `predict()` now
  takes a WAV file path. Added `NoBirdDetectedError` for graceful degradation
  when no SD species detected (PR #26)
- `src/vision/classify.py` — `VisualClassifier` updated to frozen
  EfficientNet-B0 backbone + sklearn StandardScaler + LogisticRegression
  pipeline (F1=0.931 vs F1=0.097 fine-tuned). Added `camera_index` param
  passed through to `ClassificationResult` (PR #26)
- `src/agent/bird_agent.py` — `_cycle()` wired to live `AudioCapture` and
  `VisionCapture`. Both cameras classify independently. Cooldown suppression
  via `_is_on_cooldown()` prevents notification spam for repeat detections.
  `NoBirdDetectedError` handled as soft audio failure (PR #26)
- `src/data/schema.py` — `ClassificationResult` gains `camera_index` field.
  `BirdObservation` gains `visual_result_2`, `image_path_2`, `detection_box`,
  `estimated_depth_cm`, `estimated_size_cm`, `stereo_calibrated` — all
  optional with `None` defaults, fully backward compatible (PR #26)
- `src/fusion/combiner.py` — `fuse()` accepts optional `visual_result_2`.
  `_select_best_visual()` picks higher confidence or averages when both
  cameras agree on species (PR #26)
- `configs/notify.yaml` — added push/webhook channel toggles and webhook
  config block. `push: false` committed as safe default — set `true` on
  Pi deployment (PR #26)
- `configs/paths.yaml` — added Phase 5 model paths: `visual_frozen_extractor`,
  `visual_sklearn`, `stereo_calibration`, Hailo `.hef` paths (PR #26)
- `.gitignore` — expanded `models/**/*.pt` to cover subdirectory weights,
  removed accidentally tracked binary files (PR #26)
- `notebooks/visual_efficientnet.ipynb` — added cell 28: saves
  `frozen_extractor.pt` + `sklearn_pipeline.pkl` with verification round-trip
  (PR #26)
- `.env.example` — added `PUSHOVER_USER_KEY`, `PUSHOVER_APP_TOKEN`,
  `WEBHOOK_AUTH_TOKEN` with setup instructions (PR #27)
- `docs/SETUP.md` — added Phase 5 model artifact generation section (PR #26)
- `docs/DATASETS.md` — added model artifacts table (PR #26)

### Test count
- 331 passing, 0 failing, CI green

---

## [0.4.3] — Phase 4 Frozen EfficientNet + Linear Classifier

### Added
- `notebooks/visual_efficientnet.ipynb` Section 11 — frozen EfficientNet-B0
  feature extractor + LogisticRegression; test accuracy=0.930, macro F1=0.931,
  weighted F1=0.930 (19 species, n=672); 8x improvement over SVM baseline (PR #X)
- `notebooks/results/visual_linear_confusion_matrix.png`,
  `visual_linear_per_class_f1.png` — evaluation plots for frozen+linear approach

### Known results
- Best C=0.1 on val set; all 19 species beat SVM baseline individually
- DOWO and SOSP: F1=1.00; HOSP lowest at F1=0.85

---

## [0.4.2] — Phase 4 BirdNET Pretrained Audio

### Added
- `notebooks/audio_birdnet.ipynb` Section 10-13 — BirdNET pretrained inference
  via birdnetlib 0.9.0; test accuracy=0.744, macro F1=0.776, weighted F1=0.823
  (18 species, n=86); 4x improvement over KNN baseline (PR #X)
- `notebooks/results/audio_birdnet_confusion_matrix.png`,
  `audio_birdnet_per_class_f1.png` — BirdNET evaluation plots
- `resampy==0.4.3` — required for birdnetlib MP3 decoding via librosa

### Known results
- 66/86 test files got a known-species detection (coverage 77%)
- BLPH, DOWO, HOSP, MODO: F1=1.00; ANHU lowest at F1=0.67

---

## [0.4.1] — Phase 4 Classifier Modules + Agent Wiring

### Added
- `src/audio/classify.py` — AudioClassifier wrapping `_build_audio_cnn`;
  lazy loading, from_config() reads paths.yaml, predict() on mel spectrograms
- `src/vision/classify.py` — VisualClassifier wrapping EfficientNet-B0 via
  timm; lazy loading, HWC->CHW transpose at inference
- `src/agent/bird_agent.py` — from_config() and _cycle() implemented;
  graceful degradation per modality, threshold gate before dispatch
- `scripts/generate_label_map.py` — derives label maps from split CSVs;
  writes models/label_map.json, audio_label_map.json, visual_label_map.json
- `tests/audio/test_classify.py` — 18 tests
- `tests/vision/test_classify.py` — 19 tests
- `tests/agent/test_bird_agent.py` — 20 tests
- `tests/scripts/test_generate_label_map.py` — 10 tests

---

## [0.4.0] — Phase 4 Baseline CNN Models

### Added
- `notebooks/audio_birdnet.ipynb` — CNN from scratch on mel spectrograms;
  test accuracy=0.116, macro F1=0.089 (18 species, n=86); underperforms
  KNN baseline — insufficient data for CNN from scratch
- `notebooks/visual_efficientnet.ipynb` — EfficientNet-B0 fine-tuned;
  test accuracy=0.103, macro F1=0.097 (19 species, n=672); overfits on
  limited data
- `notebooks/results/experiments.csv` — cleaned 6-row canonical log with
  deduplication guard in all Phase 4 notebooks
- `notebooks/results/phase3/`, `phase4/` — result PNGs organized by phase

### Changed
- `requirements.txt` — added birdnetlib==0.9.0, tensorflow-cpu==2.21.0,
  resampy==0.4.3 under Phase 4 BirdNET inference section

---

## [0.3.2] — Phase 3 Fusion + Notify

### Added
- `src/fusion/combiner.py` — `ScoreFuser` fully implemented: equal, weighted,
  and max confidence fusion strategies; winner-takes-all species disagreement
  handling; graceful single-modality fallback; `from_config()` reads
  `configs/thresholds.yaml` (PR #18)
- `src/notify/notifier.py` — `Notifier` fully implemented: JSONL log channel
  appends to `logs/observations.jsonl`; print channel formats via
  `message_template`; `from_config()` reads `notify.yaml` + `paths.yaml`;
  push/email deferred to Phase 5 (PR #18)
- `tests/fusion/test_combiner.py` — expanded from 7 to 44 tests covering all
  strategies, disagreement resolution, single-modality fallback, and
  `from_config()` (PR #18)
- `tests/notify/test_notifier.py` — 30 new tests: init, log, print, dispatch,
  and `from_config()` (PR #18)

---

## [0.3.1] — Phase 3 Visual Baseline

### Added
- `notebooks/visual_baseline.ipynb` — SVM classifier on HOG + color histogram
  features (26340-dim vector); C selection on val set (best C=10.0);
  test accuracy=0.213, macro F1=0.121 (19 species, n=672) (PR #17)
- `notebooks/results/visual_baseline_*.png` — frozen C-selection, confusion
  matrix, and per-class F1 plots
- `notebooks/results/experiments.csv` — second row appended (SVM visual)
- `models/baselines/visual_svm_baseline.pkl` — trained SVM + scaler +
  label encoder saved for Phase 4 comparison (gitignored)

### Changed
- `requirements.txt` — added scikit-image==0.24.0 for HOG feature extraction

### Known results
- Top performer: DOWO F1=0.81 (distinctive black/white pattern)
- YRUM: high recall (0.78) but low precision — class imbalance artifact
- 12 species scored F1=0.00 — expected with HOG+color on limited data

---

## [0.3.0] — Phase 3 Audio Baseline

### Added
- `notebooks/audio_baseline.ipynb` — KNN classifier on MFCC features
  (80-dim mean+std vector, n_mfcc=40); k selection on val set (best k=3);
  test accuracy=0.302, macro F1=0.191 (18 species, n=86) (PR #16)
- `notebooks/results/audio_baseline_*.png` — frozen k-selection, confusion
  matrix, and per-class F1 plots
- `notebooks/results/experiments.csv` — running experiment log, first row
  appended (KNN audio)
- `models/baselines/audio_knn_baseline.pkl` — trained KNN + scaler +
  label encoder saved for Phase 4 comparison (gitignored)
- `notebooks/` directory — established with `results/` subdirectory for
  all notebook output artifacts

### Changed
- `requirements.txt` — added scikit-learn==1.8.0, matplotlib==3.10.8,
  pandas==3.0.2 under new Phase 3 section
- `.gitignore` — added `models/**/*.pkl` to exclude trained baseline artifacts

### Known results
- Top performers: OCWA F1=0.80, WCSP F1=0.63, HOFI F1=0.55
- Thin-data species (ANHU, BLPH, DOWO, MODO, YRUM): F1=0.00 — data
  limitation, not model failure

---

## [0.2.4] — Phase 2 Split Generation

### Added
- `src/data/splitter.py` — stratified 60/20/20 train/val/test split generator
  for both audio and visual modalities, `NABIRDS_CLASS_MAP` covering all 20
  SD species including plumage variants, deterministic via fixed seed
- `scripts/generate_splits.py` — CLI with `--audio-only`, `--visual-only`,
  `--train-ratio`, `--val-ratio` flags, reads all config from YAML
- `tests/data/test_splitter.py` — 32 synthetic tests, no real dataset files
  required (PR #12)

### Fixed
- `configs/thresholds.yaml` — added `splits` section (`train_ratio`,
  `val_ratio`, `random_seed`), fixed `audio_weight`/`visual_weight`
  indentation (were at root level, now correctly nested under `fusion`)

### Known issues
- CAVI (California Scrub-Jay) has no NABirds visual data — NABirds predates
  the 2016 AOU split of Western Scrub-Jay. Audio training unaffected.
  To be addressed in Phase 3.

---

## [0.2.3] — Phase 2 Species Expansion

### Changed
- `configs/species.yaml` — expanded from 15 to 20 SD region species (PR #11)
  - Removed 5 non-SD species: BCCH, NOCA, WTSP, CEDW, YWAR
  - Added 10 genuine SD backyard/feeder species: AMCR, SPTO, BLPH, HOSP,
    EUST, WCSP, HOORI, WBNU, OCWA, YRUM
  - Entries grouped by resident vs seasonal
  - Source: eBird SD frequency data + San Diego Field Ornithologists checklist

---

## [0.2.2] — Phase 2 Data Pipeline

### Added
- `src/data/downloader.py` — Xeno-canto API v3 pagination, quality filtering
  (A/B), idempotent download loop with metadata sidecar, NABirds structural
  verification utilities (PR #11)
- `scripts/download_datasets.py` — CLI with `--dry-run`, `--species`,
  `--max-per-species` flags, API key from `.env` (PR #11)
- `tests/data/test_downloader.py` — 37 synthetic tests, all network calls
  mocked, no internet required (PR #11)
- `docs/DATASETS.md` — dataset sources, licenses, manual NABirds setup
  steps, split schema
- `.env.example` — added `XENO_CANTO_API_KEY` (PR #10)

---

## [0.2.1] — Phase 2 Vision Preprocessing

### Added
- `src/vision/preprocess.py` — full image preprocessing pipeline:
  `load_image`, `resize`, `normalize`, `augment`, `preprocess_frame`,
  `preprocess_file` (PR #8)
- `tests/vision/test_preprocess.py` — 40 unit tests, fully synthetic,
  no hardware or real image files required (PR #8)
- Output is HWC float32 (224, 224, 3), ImageNet-normalized — CHW
  transpose deferred to classify.py in Phase 4

---

## [0.2.0] - Phase 2 Audio Preprocessing

### Added
- `src/audio/preprocess.py` — full WAV → mel spectrogram pipeline:
  `load_wav`, `normalize`, `to_mel_spectrogram`, `preprocess_file`,
  `preprocess_array` (PR #6)
- `tests/audio/test_preprocess.py` — 31 unit tests, fully synthetic,
  no hardware or real audio files required (PR #6)

### Fixed
- CI workflow now installs from `requirements.txt` so librosa and all
  runtime dependencies are available during test runs (PR #6)
- Removed unused `librosa.display` import from `preprocess.py` (PR #6)


---

## [0.1.1] - CI and Docs Cleanup

### Added
- `ROADMAP.md` — full 6-phase development plan with status tracking (PR #4)
- `docs/ONBOARDING.md` — contributor setup guide for new team members (PR #1)
- GitHub Actions CI workflow — lint + format + tests on every PR (PR #2)
- `.github/pull_request_template.md` — structured PR checklist (PR #2)

### Fixed
- Pydantic `model_` namespace warning in `ClassificationResult` — added
  `model_config = {"protected_namespaces": ()}` (PR #1)
- CI badge URL corrected in `README.md` (PR #3)
- README team table updated, Daniel handle placeholder noted (PR #3)

---

## [0.1.0] — Phase 1 Scaffold

### Added
- Initial repository scaffold (PR #1)
  - Full `src/` module structure with stubs and docstrings
  - `src/data/schema.py` — Pydantic models: `ClassificationResult`,
    `BirdObservation`, `Modality`
  - `configs/` YAML system — species, thresholds, paths, notify
  - `tests/` mirroring `src/` — 28 passing tests, 0 warnings
  - `docs/ARCHITECTURE.md` — system design and dependency rules
  - `docs/SETUP.md` — clone, install, run instructions
  - `docs/CONTRIBUTING.md` — branch naming, commit format, PR rules
  - `requirements.txt` and `requirements-dev.txt` — all versions pinned
  - `.env.example` — documented environment variables
  - `.gitignore` — excludes venv, datasets, model weights, logs
  - `pyproject.toml` — ruff and pytest configuration
  - Custom non-commercial license

---

<!--
TEMPLATE for new entries — add above [Unreleased] when starting a phase:

## [0.X.0] — Phase N Description

### Added
- New features or files

### Changed
- Changes to existing functionality

### Fixed
- Bug fixes

### Removed
- Removed features
-->