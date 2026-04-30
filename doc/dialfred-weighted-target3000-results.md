# DialFRED Weighted Target 3000 Results

## Status

Measured for text-only.

This experiment preserves the natural class skew after clean filtering and image matching, then relies on the existing class-weighted binary loss during training.

## Dataset And Filtering

```text
Dataset: DialFRED labels + ALFRED raw image frames
Target rows: 3000
Filter: ambiguous_mean <= 0.25 or ambiguous_mean >= 0.75
Balancing: disabled
Training: class-weighted BCE loss
Purpose: test whether more rows beat strict per-split class balancing
```

## Commands

```powershell
python -m sensor_vlm.prepare_clean_manifest --target-rows 3000 --negative-max 0.25 --positive-min 0.75 --no-balance --output artifacts\features\dialfred-weighted-target3000_manifest.csv
python -m sensor_vlm.build_features text-manifest --manifest artifacts\features\dialfred-weighted-target3000_manifest.csv --output artifacts\features\dialfred-weighted-target3000_text_features.npz
python -m sensor_vlm.build_features multimodal-manifest --manifest artifacts\features\dialfred-weighted-target3000_manifest.csv --output artifacts\features\dialfred-weighted-target3000_multimodal_features.npz
python -m sensor_vlm.train train-cache --features artifacts\features\dialfred-weighted-target3000_text_features.npz --checkpoint artifacts\checkpoints\dialfred-weighted-target3000_text_mlp.pt --report artifacts\reports\dialfred-weighted-target3000-text-results.txt --epochs 50 --batch-size 64
python -m sensor_vlm.train train-cache --features artifacts\features\dialfred-weighted-target3000_multimodal_features.npz --checkpoint artifacts\checkpoints\dialfred-weighted-target3000_multimodal_mlp.pt --report artifacts\reports\dialfred-weighted-target3000-multimodal-results.txt --epochs 50 --batch-size 16
```

## Results

The weighted target-3000 manifest reached the requested 3,000 rows, but the data is heavily skewed toward ambiguous examples.

```text
Labels before filtering:        4,574
Labels after confidence filter: 3,642
Rows with existing images:      3,642
Rows written:                   3,000

Train:        2,185 -> 79 not ambiguous, 2,106 ambiguous
Valid seen:     415 -> 22 not ambiguous,   393 ambiguous
Valid unseen:   400 -> 17 not ambiguous,   383 ambiguous
```

Text-only result:

```text
Accuracy:              0.890
Macro F1:              0.563
Balanced accuracy:     0.605
Not-ambiguous recall:  0.294
Ambiguous recall:      0.916
Confusion matrix:      [[5, 12], [32, 351]]
```

Artifacts:

```text
Manifest:       artifacts/features/dialfred-weighted-target3000_manifest.csv
Feature cache:  artifacts/features/dialfred-weighted-target3000_text_features.npz
Checkpoint:     artifacts/checkpoints/dialfred-weighted-target3000_text_mlp.pt
Report:         artifacts/reports/dialfred-weighted-target3000-text-results.txt
```

## Caveat

This is not directly comparable to balanced runs unless macro F1, balanced accuracy, and per-class recall are emphasized over weighted F1.
