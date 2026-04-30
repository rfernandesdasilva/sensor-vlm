# DialFRED Clean Strict Target 3000 Results

## Status

Partially measured.

This experiment scales the current clean-balanced DialFRED/ALFRED setup to a 3000-row target while preserving the strict confidence thresholds used by `clean-balanced-1500`.

## Dataset And Filtering

```text
Dataset: DialFRED labels + ALFRED raw image frames
Target rows: 3000
Filter: ambiguous_mean <= 0.25 or ambiguous_mean >= 0.75
Balancing: equal ambiguous/not-ambiguous rows inside train, valid_seen, and valid_unseen
Expected constraint: actual rows may remain far below 3000 if clean not-ambiguous rows or matched images are limited
```

## Commands

```powershell
python -m sensor_vlm.prepare_clean_manifest --target-rows 3000 --negative-max 0.25 --positive-min 0.75 --output artifacts\features\dialfred-clean-strict-target3000_manifest.csv
python -m sensor_vlm.build_features text-manifest --manifest artifacts\features\dialfred-clean-strict-target3000_manifest.csv --output artifacts\features\dialfred-clean-strict-target3000_text_features.npz
python -m sensor_vlm.build_features multimodal-manifest --manifest artifacts\features\dialfred-clean-strict-target3000_manifest.csv --output artifacts\features\dialfred-clean-strict-target3000_multimodal_features.npz
python -m sensor_vlm.train train-cache --features artifacts\features\dialfred-clean-strict-target3000_text_features.npz --checkpoint artifacts\checkpoints\dialfred-clean-strict-target3000_text_mlp.pt --report artifacts\reports\dialfred-clean-strict-target3000-text-results.txt --epochs 50 --batch-size 64
python -m sensor_vlm.train train-cache --features artifacts\features\dialfred-clean-strict-target3000_multimodal_features.npz --checkpoint artifacts\checkpoints\dialfred-clean-strict-target3000_multimodal_mlp.pt --report artifacts\reports\dialfred-clean-strict-target3000-multimodal-results.txt --epochs 50 --batch-size 16
```

## Results

The strict target-3000 manifest still produced only 290 balanced rows.

```text
Labels before filtering:        4,574
Labels after confidence filter: 3,642
Rows with existing images:      3,642
Rows written:                     290

Train:        212 -> 106 not ambiguous, 106 ambiguous
Valid seen:    44 ->  22 not ambiguous,  22 ambiguous
Valid unseen:  34 ->  17 not ambiguous,  17 ambiguous
```

Text-only result:

```text
Accuracy:              0.471
Macro F1:              0.454
Balanced accuracy:     0.471
Not-ambiguous recall:  0.294
Ambiguous recall:      0.647
Confusion matrix:      [[5, 12], [6, 11]]
```

Text artifacts:

```text
Manifest:       artifacts/features/dialfred-clean-strict-target3000_manifest.csv
Feature cache:  artifacts/features/dialfred-clean-strict-target3000_text_features.npz
Checkpoint:     artifacts/checkpoints/dialfred-clean-strict-target3000_text_mlp.pt
Report:         artifacts/reports/dialfred-clean-strict-target3000-text-results.txt
```

The full strict BLIP-2 multimodal cache was attempted multiple times, but Windows terminated the long-running extraction before the cache could be saved. The detached run reached row 214/290 before aborting with:

```text
forrtl: error (200): program aborting due to window-CLOSE event
```

## Caveat

If this run again produces only a few hundred rows, the result should be described as a strict-clean ablation rather than a true 3000-row experiment.
