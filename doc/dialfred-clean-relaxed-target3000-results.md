# DialFRED Clean Relaxed Target 3000 Results

## Status

Measured for text-only.

This experiment tests whether slightly relaxed label thresholds can increase the usable DialFRED/ALFRED training set without including midpoint/noisy annotations.

## Dataset And Filtering

```text
Dataset: DialFRED labels + ALFRED raw image frames
Target rows: 3000
Filter: ambiguous_mean <= 0.33 or ambiguous_mean >= 0.67
Balancing: equal ambiguous/not-ambiguous rows inside train, valid_seen, and valid_unseen
Purpose: measure the scale-vs-label-noise tradeoff against the strict run
```

## Commands

```powershell
python -m sensor_vlm.prepare_clean_manifest --target-rows 3000 --negative-max 0.33 --positive-min 0.67 --output artifacts\features\dialfred-clean-relaxed-target3000_manifest.csv
python -m sensor_vlm.build_features text-manifest --manifest artifacts\features\dialfred-clean-relaxed-target3000_manifest.csv --output artifacts\features\dialfred-clean-relaxed-target3000_text_features.npz
python -m sensor_vlm.build_features multimodal-manifest --manifest artifacts\features\dialfred-clean-relaxed-target3000_manifest.csv --output artifacts\features\dialfred-clean-relaxed-target3000_multimodal_features.npz
python -m sensor_vlm.train train-cache --features artifacts\features\dialfred-clean-relaxed-target3000_text_features.npz --checkpoint artifacts\checkpoints\dialfred-clean-relaxed-target3000_text_mlp.pt --report artifacts\reports\dialfred-clean-relaxed-target3000-text-results.txt --epochs 50 --batch-size 64
python -m sensor_vlm.train train-cache --features artifacts\features\dialfred-clean-relaxed-target3000_multimodal_features.npz --checkpoint artifacts\checkpoints\dialfred-clean-relaxed-target3000_multimodal_mlp.pt --report artifacts\reports\dialfred-clean-relaxed-target3000-multimodal-results.txt --epochs 50 --batch-size 16
```

## Results

The relaxed target-3000 manifest still produced only 290 balanced rows.

```text
Labels before filtering:        4,574
Labels after confidence filter: 3,786
Rows with existing images:      3,786
Rows written:                     290

Train:        212 -> 106 not ambiguous, 106 ambiguous
Valid seen:    44 ->  22 not ambiguous,  22 ambiguous
Valid unseen:  34 ->  17 not ambiguous,  17 ambiguous
```

Text-only result:

```text
Accuracy:              0.471
Macro F1:              0.421
Balanced accuracy:     0.471
Not-ambiguous recall:  0.176
Ambiguous recall:      0.765
Confusion matrix:      [[3, 14], [4, 13]]
```

Artifacts:

```text
Manifest:       artifacts/features/dialfred-clean-relaxed-target3000_manifest.csv
Feature cache:  artifacts/features/dialfred-clean-relaxed-target3000_text_features.npz
Checkpoint:     artifacts/checkpoints/dialfred-clean-relaxed-target3000_text_mlp.pt
Report:         artifacts/reports/dialfred-clean-relaxed-target3000-text-results.txt
```

## Interpretation Template

If relaxed filtering improves macro F1 and not-ambiguous recall, it likely adds useful coverage. If it increases row count but hurts balanced accuracy, label noise is probably outweighing the benefit of scale.
