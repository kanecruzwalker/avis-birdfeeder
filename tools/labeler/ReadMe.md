# Avis labeling-assistant review UI (Layer 2)

A token-authenticated web UI for verifying the Gemini-generated pre-labels
written to `data/labels/pre_labels.jsonl` by Layer 1. Outputs verified
records to `data/labels/verified_labels.jsonl`, which Track 3 retraining
consumes.

This is a dev-time internal tool, not a deployed service. It's designed
for a single reviewer (you) plus optional phone-over-Tailscale access for
mobile review sessions.

---

## Why this exists

Layer 1 (`tools/labeler/pre_labeler.py`) writes one Gemini pre-label per
capture to `pre_labels.jsonl`. Those pre-labels are useful but not
ground-truth — Gemini is sometimes confidently wrong, particularly on
species visually similar to but outside its prompted vocabulary. Layer 2
lets you walk through pre-labels image-by-image, confirm or correct
them, and produce a clean verified dataset suitable for retraining the
visual classifier (Track 3).

The end-to-end flow:

```
captures/images/*.png  ──► Layer 1 (Gemini) ──► pre_labels.jsonl
                                                     │
                                                     ▼
                                            ┌──────────────────┐
                                            │  Layer 2 review  │  ← you
                                            │  UI (this tool)  │
                                            └──────────────────┘
                                                     │
                                                     ▼
                                            verified_labels.jsonl
                                                     │
                                                     ▼
                                          Track 3 retraining (PR-3)
```

---

## Setup

### 1. Install dependencies

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

This pulls in `fastapi`, `uvicorn[standard]`, `python-multipart`, and
`httpx` (used by the test suite).

### 2. Generate an auth token

```powershell
python -c "import secrets; print('AVIS_WEB_TOKEN=' + secrets.token_urlsafe(32))"
```

Copy the printed line into your `.env` file. The token must be at least
16 characters or the server refuses to start.

### 3. (Optional) Self-host the UI fonts

The UI defaults to system fonts and looks fine that way. For the full
warm-naturalist look, drop the variable WOFF2 files in:

```
tools/labeler/ui/static/fonts/Fraunces-Variable.woff2
tools/labeler/ui/static/fonts/Geist-Variable.woff2
```

Both fonts are open-source (SIL OFL). Skip this step entirely and the
CSS gracefully falls back to Georgia + system UI.

### 4. Run it

```powershell
python -m tools.labeler.ui
```

The banner prints the URL to open. It includes `?token=...` already —
click the full URL, don't truncate it.

```
================================================================
  url:    http://localhost:8765/?token=Xq3v...8nA
================================================================
```

---

## Daily workflow

You'll spend most of your time in the **Review** screen. The shape of a
single review is:

1. Look at the image.
2. Decide what bird (if any) you're seeing.
3. Press one button.

That's it. The store auto-saves, fetches the next image, and updates the
bucket counts. The interactions:

| Action | Keyboard | Phone |
|---|---|---|
| Confirm pre-label | `Enter` | tap Confirm button or **swipe right** |
| Skip (don't record) | `S` or `→` | tap Skip or **swipe left** |
| Quick-correct to species | `1`–`9`, `N`, `U` | tap quick-correct button |
| OTHER (out-of-vocab) | `O` then type 4-letter code | tap OTHER button |
| Back to summary | `Esc` | tap Back |
| Previous image | `←` | browser back gesture |

The image briefly haloes green when you confirm, amber when you correct.
That's your "got it" signal. The next image fades in 180 ms after.

### Switching themes

Top-right of the toolbar has three colored pills (warm / pollen / mono)
plus a sun/moon toggle. Choices are persisted in localStorage. First
load honors your OS dark-mode preference.

---

## The species code system

Every verified record has a `species_code`. There are 23 valid values:

- **20 known species codes** — `HOFI`, `SOSP`, `AMCR`, `MOCH`, etc. Defined in
  `configs/species.yaml`. These are the classes the visual classifier
  was trained on and can predict directly.
- **Three sentinels** — `NONE`, `UNKNOWN`, `OTHER`. Each means something
  specific.

### NONE vs UNKNOWN vs OTHER — the distinction that matters

This is the most-asked question and getting it right matters for
retraining quality.

| Code | Meaning | When to use |
|---|---|---|
| **NONE** | No bird in the frame at all | Empty feeder, leaves blowing, false trigger |
| **UNKNOWN** | A bird IS visible but I can't identify it | Too blurry, too distant, weird angle, motion-blurred |
| **OTHER** + 4-letter code | I CAN identify this bird and it's *not* in the model's vocabulary | California Towhee (CALT), Black-headed Grosbeak (BHGR), etc. |

**Decision rule when you're unsure:**

> *If I had to bet $20, could I name the species?*
> - Yes, and it's in the 20 known codes → use that code (Confirm or quick-correct)
> - Yes, but it's not in the 20 codes → use OTHER + the 4-letter code
> - No → UNKNOWN

The Merlin app on your phone is a useful tiebreaker. If Merlin is
confident, you can use its answer. If Merlin is also unsure, that's
itself a signal to use UNKNOWN.

### Why OTHER matters more than you'd expect

Every OTHER record represents a human catching something the model
couldn't even name. They're disproportionately valuable for the
project's research direction:

- A few OTHER records of the same code (say, 5 BHGR) tell you a species
  exists in the deployment that the vocabulary didn't anticipate.
- A *lot* of OTHER records of one code (say, 521 CALT) is a hard mandate
  to expand the vocabulary. CALT alone goes from "interesting one-off"
  to "must-add" at that volume.

Layer 1 doesn't emit OTHER — it only knows the 20 known codes plus
NONE/UNKNOWN. So OTHER is exclusively a human-review signal.

---

## Optimistic concurrency

If you accidentally have the UI open on two devices and try to verify
the same image from both, the second one to submit gets a 409 conflict
modal. The modal shows what the first session recorded and asks if you
want to overwrite. Cancel keeps the first session's record; Overwrite
replaces it with yours. The store always atomically rewrites the
`verified_labels.jsonl` on overwrite — there's no risk of duplicate
records on disk.

This is implemented via a `client_load_time` round-trip on every fetch.
You don't need to think about it; the UI handles it. If you ever see
the conflict modal, it means you (or an instance of you) actually did
verify that image twice from different sessions.

---

## Inspecting your verified dataset

Two read-only diagnostic scripts ship with the UI:

### `inspect_verified.py` — what's in your verified_labels.jsonl?

```powershell
python -m tools.labeler.ui.inspect_verified
```

Prints:

- **Schema validation** — does every record parse cleanly against the
  `VerifiedLabel` Pydantic model?
- **Distribution by species** — how many records of each code, with
  sentinel markers
- **OTHER breakdown** — every 4-letter out-of-vocab code you used,
  with counts (this is the project-relevant gold)
- **Agreement rate with pre-labeler** — overall, plus per-species. Tells
  you which buckets the pre-labeler is reliable on and which need human
  review.
- **Duplicates and orphans** — both should be empty
- **Reviewer notes** — last 5 records where you wrote a comment
- **Time span and rate** — how long you've been at it
- **Overall coverage** — % of pre-labels you've reviewed

### `inspect_unreviewed.py` — what's left to review?

```powershell
python -m tools.labeler.ui.inspect_unreviewed
```

Prints:

- Distribution of the unreviewed backlog by Gemini's pre-label
- Confidence histogram for the dominant unreviewed pre-label
- Cross-reference: agreement rate on already-reviewed records of that
  pre-label, with a recommendation about whether to bulk-confirm,
  spot-check, or continue manual review

---

## Interpreting results

This is the most useful part of the README to revisit. Here's how to
read the inspection output and what conclusions to draw.

### Worked example — Kane's first major session

After ~20 hours of review across 4280 records, `inspect_verified.py`
printed:

```
Distribution by verified species_code
─────────────────────────────────────
  NONE     2104  ← sentinel
  SOSP     1014
  OTHER     521  ← sentinel
  HOFI      284
  AMCR      208
  UNKNOWN   149  ← sentinel

OTHER → other_species_code breakdown
─────────────────────────────────────
  CALT    521

Agreement with pre-labeler
─────────────────────────────────────
  Per-pre-label-species (≥3 reviewed):
    NONE     2097/2100  ( 99.9%)
    SOSP     1002/1039  ( 96.4%)
    AMCR      198/198   (100.0%)
    HOFI      267/382   ( 69.9%)
    MODO        0/225   (  0.0%)
    MOCH        0/156   (  0.0%)
    WREN        0/66    (  0.0%)
    HOSP        0/50    (  0.0%)
    ...
```

**What this means in plain language:**

The deployment ecology turned out to be much narrower than the model's
20-species vocabulary expected. Only four species visited the feeder
visually during this period: AMCR, HOFI, SOSP, and CALT.

- **AMCR (100% agreement)** — perfectly classified, no retraining needed
  for this class. American Crow has a distinctive silhouette that's hard
  to confuse.
- **SOSP (96.4%)** — solid, with a small error rate worth understanding
  but not urgent.
- **HOFI (69.9%)** — workable but the lowest in-vocab agreement. ~30% of
  HOFI pre-labels needed correction. Worth investigating what those 115
  corrections went to (likely SOSP or CALT confusion).
- **CALT (521 OTHER records)** — the dominant out-of-vocab finding. The
  model has no class for California Towhee, so it substituted whatever
  in-vocab class looked closest — usually MOCH (Northern Mockingbird) or
  MODO (Mourning Dove).
- **MODO/MOCH/WREN/HOSP/etc. (0% agreement)** — these are visual
  hallucinations. The pre-labeler proposed these species but none
  actually visited. Human review caught them all.

**The retraining mandate this implies:**

Adding **CALT** to the visual classifier's vocabulary, with the 521
reviewer-confirmed examples as training data, is the dominant Track 3
intervention. Other vocabulary additions are not supported by this
data — the species the pre-labeler hallucinated didn't actually visit
the feeder.


### Selection bias to acknowledge in any writeup

- **Camera trigger bias.** The cameras only saw what triggered a
  capture. Smaller, shyer, or quieter species might visit and not
  trigger. Hummingbirds in particular are unlikely to trigger
  motion-based capture due to their hover behavior.
- **Diurnal coverage.** Capture distribution likely skews toward
  certain hours. Species with off-peak activity patterns may be
  underrepresented.
- **Single-feeder generalization.** "Four species visit" is true at
  Kane's hillside feeder. The retrained model will be feeder-specific
  unless additional sites are sampled.
- **Merlin as ground-truth proxy.** Merlin is excellent but not
  infallible at unusual angles or partial views. UNKNOWN records are
  high quality (honestly uncertain), but verified species records
  inherit Merlin's accuracy when used as a tiebreaker.

None of these change the headline conclusion (CALT must be added). All
are worth a sentence in the project writeup.

### Reading agreement rates more generally

When you run `inspect_verified.py` and look at per-species agreement,
the patterns mean different things:

| Pattern | Reading | Action |
|---|---|---|
| **High agreement (>95%) on hundreds of records** | Pre-labeler is reliable on this class | Trust the pre-label, spot-check rather than reviewing every one |
| **Moderate agreement (70-95%)** | Pre-labeler usable but error-prone | Continue manual review; corrections are useful training signal |
| **Low agreement (<50%) on a substantial bucket** | Pre-labeler systematically wrong on this class | Investigate WHY — is the true species OOV? Is the visual signal weak? Is there confusion with another class? |
| **Zero agreement on a real-sized bucket** | Either the species doesn't visit, or there's a systematic confusion (often: real species is OOV) | Look at what the human corrections went to — that's the answer |

### Bulk-confirming a high-confidence bucket

If `inspect_unreviewed.py` says you have ~4000 unreviewed pre-labels
that are all NONE with 99%+ agreement on already-reviewed examples, you
have two reasonable options:

**Option A: Spot-check, then bulk-confirm.** Open the UI, filter to
NONE, and review a random 50-100 of the unreviewed records. If they all
look like empty feeder shots, you can be confident the rest are too.
Then run a small one-off script to write `agreed=true` NONE records for
the remainder.

**Option B: Continue manual review.** Slower but more defensible if
you'll be presenting the dataset publicly. ~3.5 records/min on
empty-frame review is achievable.

A bulk-confirm helper script is intentionally not shipped here — the
right approach depends on your accuracy bar. If you want one, the logic
is ~30 lines: iterate `pre_labels.jsonl`, skip filenames already in
`verified_labels.jsonl`, write a `VerifiedLabel` with
`agreed_with_pre_label=true` for each remaining `species_code=NONE`
record. Validate with `inspect_verified.py` afterward.

---

## Architecture

The UI is intentionally simple — single-process FastAPI server, vanilla
JS frontend, no build step.

```
tools/labeler/ui/
├── __init__.py
├── __main__.py           ── CLI entry: parses args, loads .env, runs uvicorn
├── auth.py               ── AVIS_WEB_TOKEN dependency, hmac-compare
├── review_store.py       ── In-memory store + atomic JSONL persistence
├── routes.py             ── FastAPI routes for /api/*, /image/*, /
├── server.py             ── App factory, /health, static + template mounts
├── inspect_verified.py   ── Diagnostic for verified_labels.jsonl
├── inspect_unreviewed.py ── Diagnostic for the unreviewed backlog
├── templates/
│   └── index.html        ── Single SPA template, three views
└── static/
    ├── styles.css        ── Six themes (warm/pollen/mono × light/dark)
    ├── app.js            ── Vanilla JS, no framework
    └── fonts/            ── (optional) self-hosted Fraunces + Geist
```

Persistence model:

- **Pre-labels are read-only** to the UI. Layer 1 owns
  `pre_labels.jsonl`; the UI never touches it.
- **First verification of an image appends** a record to
  `verified_labels.jsonl`. Atomic via `tempfile + os.replace`.
- **Correcting a verified record rewrites** the entire JSONL file with
  the corrected record in place. Atomic via temp + rename. The file
  stays one record per image — no duplicates, no append-only history.
  If you need pre-correction state, that's what git is for.

If you crash the server mid-write, you'll see one of:
- Empty record at end of file (recoverable: last line is partial JSON)
- Old version intact (rewrite hadn't started)
- New version intact (rewrite completed)

The atomic-rename guarantee means you'll never see a corrupted file
with mixed old/new content. `inspect_verified.py` will flag any case it
can't parse cleanly.

---

## Running the test suite

```powershell
.venv\Scripts\activate
python -m pytest tests/labeler/ -v
```

Current count: 116 tests across schema (40), store (30), auth (15),
and routes (31). All should pass on a fresh checkout.

---

## Troubleshooting

**"AVIS_WEB_TOKEN is not set" at startup**
You need to add `AVIS_WEB_TOKEN=<value>` to your `.env`. See setup
step 2.

**Page loads but shows "Missing authentication token"**
You went to `http://localhost:8765/` without `?token=...` in the URL.
Use the FULL URL the banner prints, including the query parameter.

**Modal appears on first page load with empty content**
This was a CSS specificity bug fixed in a later patch — make sure your
`tools/labeler/ui/static/styles.css` includes the rule
`[hidden] { display: none !important; }` near the top of the Base
section. Hard-reload the browser (`Ctrl+Shift+R`) to bust the CSS cache.

**Phone can't reach the UI**
Default bind is `127.0.0.1`. For Tailscale/LAN access:
```
python -m tools.labeler.ui --host 0.0.0.0
```
Then on phone, open the URL with your laptop's Tailscale IP, e.g.
`http://100.83.131.6:8765/?token=...`. The token query parameter is
required even on phone.

**`inspect_verified.py` reports orphans or duplicates**
Don't ignore this. Orphans mean a verified record points to a filename
not in `pre_labels.jsonl` — most likely a pre-labels file got
truncated or replaced. Duplicates mean the atomic-rewrite path didn't
run; there's likely a bug worth investigating before continuing.

---

## What's next (Track 3 retraining)

You'll consume `verified_labels.jsonl` from the retraining pipeline:

```python
import json
with open("data/labels/verified_labels.jsonl") as fh:
    for line in fh:
        record = json.loads(line)
        # record["species_code"] is the verified label
        # record["other_species_code"] is set when species_code == "OTHER"
        # record["image_path"] points to the capture
        # record["agreed_with_pre_label"] is the audit flag
```

For training class generation, treat the union `(species_code if !=
"OTHER" else other_species_code)` as your target label. Filter
`UNKNOWN` records out of the training set (they're calibration data,
not training data). NONE records are typically used as a negative class
or excluded depending on classifier architecture.