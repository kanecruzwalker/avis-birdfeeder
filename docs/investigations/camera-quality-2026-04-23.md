# Camera Quality Upgrade (2× resolution) — Investigation

**Date:** 2026-04-23
**Branch:** `feat/camera-quality-2x`
**Author:** Kane Cruz-Walker

## Hypothesis

Capturing at **2304×1296 @ 30fps** instead of the current **1536×864 @ 120fps** will produce measurably better classifier performance on feeder-scale birds, by providing **2.25× more pixels per bird** to the classifier.

## Motivation

Branch 3 (`feat/adaptive-yolo-crop`, deployed 2026-04-23) tested whether tighter bounding-box-driven cropping would help the classifier with small feeder birds. Over 4 rotations per mode (2 hours per mode) the two modes produced near-identical results:

| Metric | fixed_crop | yolo (adaptive) |
|---|---|---|
| Dispatches per window | 5.75 | 5.50 |
| Gate pass rate | 35.1% | 34.7% |
| Visual confidence mean | 0.147 | 0.150 |
| Visual confidence stdev | 0.008 | 0.009 |
| Species dispatched | 6 | 7 |

Adaptive crop fired correctly (we have image evidence of 300×300 and 410×510 crops being saved), but classifier output was statistically indistinguishable between modes. This indicates the classifier's out-of-distribution (OOD) failure is **feature-level, not crop-level**: improving input framing does not help when the classifier cannot extract useful features from low-resolution feeder birds under any framing.

The remaining lever is input *resolution*. If each bird is represented by more pixels, the classifier has more feature information to work with. This branch tests that hypothesis.

## Quantitative reasoning

With current 1536×864 capture and a typical small bird occupying ~5% of the `feeder_crop_cam0` region (700×580):

- Bird in raw frame: ~150×150 pixels
- Feeder crop: 700×580
- After classifier downsample to 224×224: bird occupies ~50×50 pixels

With 2304×1296 capture and the feeder_crop scaled 1.5× (1050×870):

- Bird in raw frame: ~225×225 pixels (2.25× more data)
- Feeder crop: 1050×870
- After classifier downsample to 224×224: bird occupies ~75×75 pixels

The post-downsample bird size is ~2.25× larger, and the underlying pixels are cleaner (longer per-frame exposure = less sensor noise). This is a strictly richer input signal.

## Why not 3× (4608×2592)?

IMX708 supports 4608×2592 at 14 fps. Considered but rejected for this branch:

- **Disk impact:** ~2-3 MB per PNG × 1 cycle/sec = ~200 GB/day. Our `/mnt/data` has ~180 GB free. Would fill in ~22 hours without aggressive cleanup.
- **Memory:** 34 MB per raw frame × 2 cameras = 68 MB held per cycle. Fine for Pi 5, but tight if stereo or temporal smoothing ever holds multiple frames.
- **PNG save time:** ~2-3 seconds to encode and write per frame. Could block the 1s cycle loop.
- **Unknown long-term stability:** Pi 5 may thermally throttle at sustained full-res capture; not yet tested.

**Staging approach:** Ship 2× first, measure. If 2× shows material improvement, we know resolution is a productive lever and 3× becomes worth exploring in a follow-up branch. If 2× shows no improvement, 3× almost certainly won't help either.

## Changes in this branch

- **`configs/hardware.yaml`** — `capture_width/height/fps` updated; all three `feeder_crop*` blocks scaled 1.5×
- **`scripts/dev_config.py`** — Pi overrides for `feeder_crop_cam0/cam1` scaled 1.5×
- **`tests/vision/test_capture.py`** — assertions updated to match new defaults
- **`docs/camera-quality-2026-04-23.md`** — this document

## Success criteria

After 4+ hours of live deployment with A/B rotation active, measured against today's 2026-04-23 pre-upgrade baseline:

**Primary (must-have):**
- Visual confidence distribution is meaningfully different from baseline (Kolmogorov–Smirnov or similar). Specifically, `stdev` > 0.015 (vs baseline 0.008-0.009), indicating the classifier is responding to input differences rather than outputting its prior.

**Secondary (nice-to-have):**
- Dispatched species diversity ≥ 7 species per mode (vs baseline 6-7)
- Mean visual confidence trend upward (not critical — stability also acceptable)
- Classifier's top-1 species matches ground truth (from manual frame inspection) on at least 2 of 5 spot-checked bird frames

**Null hypothesis:** Classifier outputs remain distributed as ~0.147 ± 0.008 regardless of capture resolution. If this holds, the classifier's OOD problem is orthogonal to input resolution, and the remaining lever is retraining (Track 3 work).

## Operational checks

- [ ] Service starts cleanly on boot with new config
- [ ] Both cameras open at 2304×1296
- [ ] `feeder_crop` coords accurately frame the physical feeder (verify with test capture)
- [ ] Agent cycle time stays under 5 seconds
- [ ] CPU usage stays under 80% sustained
- [ ] Disk fill rate stays below 100 GB/day (we have ~180 GB free → 1.8+ days buffer)

## Risks

- **Feeder crop miscalibration:** Scaling by 1.5× assumes pure proportional scaling. If lens distortion behavior differs at the new resolution, the crop may partially miss the feeder. Mitigation: capture a test frame post-deploy and verify visually.
- **Disk pressure:** 2× capture will ~2× the per-frame PNG size. Not immediate, but monitor.
- **Regression from performance issues:** If cycle time degrades, revert hardware.yaml to previous values.

## Rollback

The change is pure config. To revert:

```bash
git revert <this-commit-sha>
# or edit hardware.yaml back to:
#   capture_width: 1536
#   capture_height: 864
#   capture_fps: 120
# and feeder_crop_cam0/cam1 back to (630,130,700,580) and (420,130,700,580)
```

## Follow-up branches (conditional)

- **`feat/camera-quality-3x`** — If 2× produces measurable improvement, test 4608×2592 @ 14fps. Requires: disk cleanup policy in place, CPU/memory profiling, thermal stability testing.
- **`feat/classifier-retrain-feeder-distribution`** — If 2× shows no improvement, the classifier's feature-space OOD is the real problem. Retrain the LogReg head on Pi-captured frames with actual feeder-distribution labeling.

## Data comparison plan

After 4+ hours of 2× deployment, run the same time-windowed A/B measurement script used for Branch 3 analysis. Produce a side-by-side table:

| Metric | Baseline (1×, 2026-04-23) | 2× deployment (this branch) |
|---|---|---|
| Dispatches/window | 5.6 avg | ? |
| Visual conf mean | 0.148 | ? |
| Visual conf stdev | 0.009 | ? |
| Species diversity | 6-7 | ? |
| Gate pass rate | 35% | ? |

Decision tree:
- If all metrics improve materially → keep 2×, proceed to 3× investigation
- If mixed results → document findings, keep 2× as neutral baseline improvement
- If all metrics worse → revert, reconsider feeder_crop calibration
- If no change → classifier is the bottleneck; work on Track 3