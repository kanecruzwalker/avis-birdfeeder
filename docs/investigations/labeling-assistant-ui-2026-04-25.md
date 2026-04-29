# Labeling Assistant Review UI — Investigation

**Date:** 2026-04-25
**Branch:** `feat/labeling-assistant-ui`
**Author:** Kane Cruz-Walker
**Related:** `docs/investigations/labeling-assistant-2026-04-24.md` (Layer 1)

## Hypothesis

A web-based human-review UI layered on top of the Layer 1 pre-labeler will produce high-quality verified labels at ~10 seconds per correct pre-label and ~30 seconds per correction — sufficient to verify the existing 2828 pre-labels in a focused 6-8 hour review session. The resulting `verified_labels.jsonl` becomes the curated training dataset that Layer 3 exports and Layer 4 retrains on.

## Motivation

Layer 1 (PR #59, merged 2026-04-24) produced 2828 pre-labels with the following distribution on the overnight batches:

| Species | Count | % |
|---|---|---|
| NONE (empty feeder) | 1911 | 67.6% |
| SOSP | 399 | 14.1% |
| HOFI | 181 | 6.4% |
| AMCR | 101 | 3.6% |
| MODO | 93 | 3.3% |
| MOCH | 63 | 2.2% |
| WREN | 30 | 1.1% |
| Others (9 species) | 50 | 1.8% |

The Layer 1 smoke test established that pre-label accuracy is 100% on in-vocab species and empty feeders but 0% on California Towhee (CALT), which is out-of-vocab and consistently mislabeled as MOCH or WREN. At ~90% of the 63 MOCH + 30 WREN pre-labels being CALT, the review UI must surface the OOV problem clearly rather than forcing reviewer acceptance of the closest in-vocab match.

Manual labeling at ~60 seconds per image would take ~47 hours for 2828 images. Review-mode labeling — confirm the pre-label in ~3 seconds for the ~76% of correct predictions, correct in ~30 seconds for the rest — should finish the same 2828 images in roughly 6-8 hours of focused work. The review UI is the multiplier that makes retraining feasible inside the CS 450 course timeline.

## Scope of this PR

Layer 2 only. No changes to Layer 1 pre-labeler behavior. No training pipeline. No Pi-side code.

This PR delivers:
1. A FastAPI-based web UI for reviewing pre-labels image-by-image
2. Schema additions for out-of-vocab species (`OTHER` sentinel + `other_species_code` field)
3. Read/write layer for `pre_labels.jsonl` (read) and `verified_labels.jsonl` (write)
4. Token-authenticated access suitable for laptop + phone use
5. Responsive UI with keyboard shortcuts (laptop) and touch gestures (mobile)
6. Verified-list view for auditing already-reviewed labels
7. Test suite covering schema additions, store operations, and route handlers

## Architecture

### Three concentric layers

```
┌───────────────────────────────────────────────────────────────┐
│  Browser (laptop or phone)                                    │
│  - Responsive SPA, vanilla JS, Tailscale or localhost         │
│  - Keyboard: Enter/1-9/N/U/O/S/B     Touch: tap/swipe/long    │
└──────────────────────────────┬────────────────────────────────┘
                               │ JSON over HTTP + AVIS_WEB_TOKEN
┌──────────────────────────────▼────────────────────────────────┐
│  FastAPI server (tools/labeler/ui/)                           │
│  - routes.py:  /api/next, /api/verify, /api/verified, /stats  │
│  - auth.py:    AVIS_WEB_TOKEN middleware                      │
│  - static/:    images served from data/captures/images/       │
└──────────────────────────────┬────────────────────────────────┘
                               │ File I/O
┌──────────────────────────────▼────────────────────────────────┐
│  ReviewStore (tools/labeler/ui/review_store.py)               │
│  - read pre_labels.jsonl on startup, index in memory          │
│  - append verified_labels.jsonl on each verify                │
│  - atomic rewrite on correction of already-verified label     │
│  - stats aggregation (by species, by confidence, coverage)    │
└───────────────────────────────────────────────────────────────┘
```

### Schema additions

All changes to `tools/labeler/schema.py` are **additive** — existing 2828 `PreLabel` records in `pre_labels.jsonl` continue to validate and deserialize without migration.

```python
# Added to schema.py

SENTINEL_OTHER = "OTHER"  # bird visible but species is OOV (e.g. CALT)

ALLOWED_CODES = KNOWN_SPECIES_CODES + (SENTINEL_NO_BIRD, SENTINEL_UNKNOWN, SENTINEL_OTHER)


# New optional field on PreLabelResponse (reviewers populate, not Gemini in v1.0)
class PreLabelResponse(BaseModel):
    # ... existing fields ...
    other_species_code: Optional[str] = Field(
        default=None,
        description="4-letter code for out-of-vocab species when species_code=OTHER.",
    )


# New required field on VerifiedLabel when species_code=OTHER
class VerifiedLabel(BaseModel):
    # ... existing fields ...
    other_species_code: Optional[str] = None

    @model_validator(mode="after")
    def _other_requires_code(self):
        if self.species_code == "OTHER" and not self.other_species_code:
            raise ValueError("species_code=OTHER requires other_species_code")
        return self
```

**Prompt version:** Staying on `v1.0` for this PR. The `OTHER` sentinel is a reviewer-side construct for the first round. When Layer 1 is re-run in the future (after we have CALT training data), we bump to `v1.1` and teach Gemini to use `OTHER` directly. This keeps the 2828 existing records untouched.

### File I/O model (verified_labels.jsonl)

**Append-only + atomic rewrite on correction.**

- Normal path: a verify action appends one JSON line to `verified_labels.jsonl`. Fast, concurrent-safe.
- Correction path: if the reviewer re-verifies an already-verified image (different label), the store rewrites the entire file atomically (write to temp file → fsync → rename). File stays one record per `image_filename`.
- On startup: load existing `verified_labels.jsonl`, index by `image_filename`, count matches against `pre_labels.jsonl` to compute coverage.

Rationale: normal review produces 99% append-only writes, so we get append performance in the common case. Corrections are rare enough that rewriting 3000-10000 lines is fine (10-50ms on SSD). Downstream Layer 3 reads a clean one-record-per-image file with no group-by-newest logic.

### Optimistic concurrency

The store tracks a `client_load_time` on each `/api/next` response (UTC timestamp when the server handed the image to a client). The client echoes this back on `/api/verify`. Conflict resolution:

| Server state at verify time | Client's load_time | Action |
|---|---|---|
| No existing verified label | any | Accept (append) |
| Existing verified label, `verified_at < client_load_time` | any | Accept (correction — user re-opened a verified image) |
| Existing verified label, `verified_at > client_load_time` | any | **409 Conflict** — return existing record, client prompts user |

Client on 409: shows modal "This image was already verified as {species} by another session. Overwrite with your choice?" If confirmed, client re-posts with `force_overwrite: true` → server accepts.

Costs ~40 lines total (new field on response, new field on request, 15-line conflict check in store, 10-line JS prompt, 4 tests). Does not require reviewer identity, websockets, or per-image locks.

### Default view: group-by-pre-label-species

Reviewer opens the UI, sees an accordion/tabbed landing page:

```
┌────────────────────────────────────────────────────────┐
│  Review Queue                    Token: ••••••••••••   │
├────────────────────────────────────────────────────────┤
│  NONE      1911    [====------]   842/1911  44%        │
│  SOSP       399    [========--]   289/399   72%        │
│  HOFI       181    [=======---]    98/181   54%        │
│  AMCR       101    [==========]   101/101  100% ✓      │
│  MODO        93    [====------]    40/93    43%        │
│  MOCH        63    [-----------]    0/63     0% ⚠      │
│  WREN        30    [-----------]    0/30     0% ⚠      │
│  ... (9 more species)                                  │
├────────────────────────────────────────────────────────┤
│  [ Review all  ]  [ Filter: unverified ▾ ]             │
│  [ Review verified ]  [ Filter: by confidence ▾ ]      │
└────────────────────────────────────────────────────────┘
```

Tapping a species bucket enters review mode filtered to that species. The ⚠ on MOCH/WREN is a soft visual cue — the reviewer is expected to find CALT mislabels in those buckets and the filter naturally front-loads that work. Any species can also be reviewed globally via "Review all".

### Review screen (mobile and desktop)

```
┌────────────────────────────────────────────────────────┐
│  ← back                 12/63 MOCH         skip →      │
├────────────────────────────────────────────────────────┤
│                                                        │
│              [captured image, fit to viewport]         │
│                                                        │
├────────────────────────────────────────────────────────┤
│  Pre-label:  MOCH  (0.85)                              │
│  "Grey-brown bird with long tail on feeder, plain     │
│   back consistent with mockingbird."                   │
│  Audio hint: — (none)                                  │
├────────────────────────────────────────────────────────┤
│  [ ✓ Confirm MOCH ]  [ ✗ Correct ▾ ]  [ Skip ]         │
├────────────────────────────────────────────────────────┤
│  Quick correct:  HOFI HOSP SOSP CALT CAVI ANHU ···     │
│  Other:          [ other_code: ____ ]                  │
│  Also:           [ NONE ]  [ UNKNOWN ]                 │
└────────────────────────────────────────────────────────┘
```

**Laptop keyboard shortcuts:**
- `Enter` — Confirm pre-label
- `1-9` — Quick-select from the quick-correct row (most frequent species configured first)
- `N` — NONE
- `U` — UNKNOWN
- `O` — OTHER (opens other_species_code input)
- `S` / `→` — Skip (no verdict recorded, advance)
- `B` / `←` — Back to previous image

**Mobile touch:**
- Tap the image → zoom
- Tap "Confirm" button → large tap target, records pre-label as verified
- Tap "Correct" → expands picker, species buttons sized ≥44px
- Swipe left → skip
- Swipe right → confirm
- Long-press image → open full-resolution in new tab

### Verified-list view

Separate route `/verified` shows all confirmed labels, filterable by species code and date range. Each row is clickable to re-open in the review screen for correction. This directly addresses "we should be able to see our confirmed list if we want to double check what we selected as true."

### Authentication

`AVIS_WEB_TOKEN` from `.env`. All routes except `/health` require the token as either:
- Query parameter: `?token=<value>` (used by the frontend for image URLs)
- HTTP header: `X-Avis-Token: <value>` (used by API calls)

Bind to `127.0.0.1` by default. Opt-in `--host 0.0.0.0` for phone access over Tailscale. Matches the Phase 8C (future web dashboard) pattern so we can reuse the auth middleware when that ships.

## Design decisions

### Why FastAPI + vanilla JS (not React, not Flask)?

- **FastAPI over Flask:** async-friendly, built-in Pydantic integration (our schemas become request/response models for free), automatic OpenAPI docs for debugging. Already the chosen framework for Phase 8C web dashboard — consistency reduces future cognitive load.
- **Vanilla JS over React:** this is a ~500-line frontend for a single-person tool. A React setup adds `npm install`, a build step, node_modules, and a bundle configuration for zero user-facing benefit. Vanilla JS with modern browser APIs (fetch, CSS Grid, touch events) handles every requirement in this spec.
- **Single HTML file with inline CSS/JS:** no bundler, no hot reload, just open index.html in the browser via FastAPI's Jinja2 renderer. Easier for a future collaborator (or Claude itself) to read, modify, or replace.

### Why `tools/labeler/ui/` (not `src/labeler_ui/`)?

Matches Layer 1's `tools/labeler/` placement. These are dev-time tools, not runtime production code. Putting them under `src/` implies production use and adds them to the default import graph. `tools/` keeps them clearly categorized and gitignored if needed.

### Why `OTHER` + `other_species_code` (not free-text)?

Free-text species fields guarantee future headaches: "California Towhee" vs "california towhee" vs "CATH" vs "CALT" vs "Cal Towhee" all become distinct labels. Structured 4-letter codes enforce normalization and pattern-match how the rest of the system represents species.

The `OTHER` approach also gives us a clean "discovery" signal: Layer 3 can count `OTHER + other_species_code=CALT` occurrences and surface "we saw this 47 times, consider adding it to species.yaml." This closes the loop on the CALT-invisible problem structurally.

### Why "group by species" as default, not "newest first"?

Reviewer efficiency. Within a species bucket, images tend to cluster visually — same bird from multiple angles at the same feeder visit. Review-then-confirm runs into a rhythm: "yep, that's a HOFI; yep, another HOFI; yep, HOFI..." that can't form when species are randomly interleaved. The user explicitly requested this ordering.

Filters are still available for "all unverified", "low confidence first", and "most recent" — so the default optimizes for the common case without locking out other modes.

### Why reject cross-record hints for now?

Earlier design considered a sidebar saying "You corrected 3 MOCH pre-labels to CALT recently, consider applying that pattern." We rejected this for two reasons:

1. **Prejudicial:** telling the reviewer "this might also be CALT" biases them toward applying the hint even when the image is a real MOCH. The whole point of human review is unbiased verification; assistive hints undercut that.
2. **Premature:** Layer 3 (dataset export) and Layer 4 (retraining) are better places to aggregate and act on OOV patterns. The UI's job is to collect clean signal; interpretation belongs downstream.

## Success criteria

**Functional:**
- Loads 2828 pre-labels in under 2 seconds on typical laptop hardware
- Renders review screen in under 500ms per image on phone over Tailscale
- Keyboard shortcuts work on laptop; touch gestures work on phone
- `OTHER + other_species_code` validates at schema layer and persists correctly
- Verified-list view shows all records, filterable, clickable to re-open
- File writes are atomic — interrupting the server mid-write doesn't corrupt the JSONL
- Concurrent verify on same image returns HTTP 409 with existing record; client offers explicit overwrite

**Quality:**
- 90%+ of reviewers can complete a full review session (100 images) without documentation
- Out-of-vocab species (CALT) can be captured in under 20 seconds including typing the code
- Previously-verified labels can be corrected in under 10 seconds

**Operational:**
- `python -m tools.labeler.ui` starts the server, prints URL and token
- Clear error messages when `AVIS_WEB_TOKEN` missing or `pre_labels.jsonl` empty
- Health endpoint (`/health`) returns 200 OK without token
- Tests cover auth, store I/O, and route handlers (50+ tests total)

## Risks

- **Reviewer fatigue biases verification:** long review sessions accept more pre-labels as correct than a fresh reviewer would. Mitigation: session timer, suggested break after 30 minutes, ability to resume later.
- **Tailscale network drops mid-session:** reviewer loses in-flight verification. Mitigation: append-only JSONL means at most one image is lost; server acks each verify before advancing the client.
- **Concurrent review (laptop + phone simultaneously):** two devices could verify the same image to different values. Mitigation: optimistic concurrency via a `pre_label_version` stamp returned with each `/api/next` response and required on `/api/verify`. If a verified label already exists that was written AFTER the client loaded, the server returns HTTP 409 Conflict with the existing record, and the client shows an explicit "this was already verified as X — overwrite?" prompt. Corrections of previously-verified images (intentional re-opens) are distinguished from concurrency conflicts by comparing write-time against client-load-time. This is last-write-wins with awareness rather than pessimistic locking; full multi-user coordination remains out of scope as unjustified given the solo+Daniel usage profile.
- **Image path portability:** pre_labels.jsonl stores absolute laptop paths that won't resolve on a teammate's fork. Mitigation: UI serves images by `image_filename` via a configured `--images-dir`, never trusting the stored absolute path. Documented in README.
- **Schema change breaks existing 2828 records:** only additive fields with defaults are added. Existing records parse unchanged. Unit tests verify round-trip of existing pre_labels.jsonl format.

## Rollback

Layer 2 is pure dev tooling; no production deployment impact. To roll back:

1. Delete `data/labels/verified_labels.jsonl` — the 2828 pre-labels remain untouched
2. `git revert` the PR — tools/labeler/ui/ disappears
3. Layer 1 pre-labeler continues working normally

No Pi impact at any point. Production agent keeps running.

## Follow-up work (separate PRs)

- `feat/layer3-training-export` — verified_labels.jsonl → folder-structured training set with temporal splits
- `feat/layer4-classifier-retrain` — retrain LogReg head on verified data, deploy sklearn_pipeline_v2.pkl to Pi
- `feat/species-discovery` — analyze `OTHER + other_species_code` aggregations, surface candidate additions to `configs/species.yaml`
- `feat/add-calt-species` — add California Towhee (CALT) to species list, regenerate splits, retrain BirdNET species filter
- `prompt/v1.1-other-sentinel` — teach Gemini to use OTHER in future pre-labeling runs (after CALT is added)

## Open questions (not blocking this PR)

1. Should verified labels feed back into Gemini prompts as few-shot examples? Potentially useful for Layer 1 v1.1, but introduces a retroactive-bias risk in the current dataset.
2. Should "skipped" images be tracked separately? Current design doesn't record skips. If we want "review coverage" metrics later, this becomes necessary.