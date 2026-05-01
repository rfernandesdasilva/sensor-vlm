# Sensor-VLM Early Results

## Summary

This report captures the first end-to-end multimodal Sensor-VLM runs using DialFRED ambiguity labels paired with extracted ALFRED image subsets.

The pipeline is working end to end:

1. DialFRED rows provide instructions, ambiguity labels, question types, and task/trial/subgoal IDs.
2. ALFRED raw image frames are extracted from `full_2.1.0.7z` for matching task/trial/subgoal rows.
3. BLIP-2 extracts visual scene features, captions, VQA answers, and caption-variance features.
4. Sentence Transformers encode instruction and caption/VQA text.
5. A compact MLP trains on the fused multimodal feature vectors.

## Dataset Subset: 60 Rows

The first real multimodal smoke run used a balanced 60-row subset.

```text
Rows: 60
Train: 20
Validation seen: 20
Validation unseen/test: 20
Ambiguous: 30
Not ambiguous: 30
```

Ambiguity type distribution:

```text
appearance: 34
location: 14
direction: 11
other: 1
```

Feature cache:

```text
artifacts/features/alfred_subset_multimodal_features.npz
```

Feature shape:

```text
60 x 1538
```

Each feature vector contains:

```text
BLIP-2 pooled Q-Former image embedding
+ instruction text embedding
+ generated caption/VQA text embedding
+ caption similarity and caption ambiguity scalar features
```

## Quantitative Results

Checkpoint:

```text
artifacts/checkpoints/alfred_subset_multimodal_mlp.pt
```

Evaluation report:

```text
artifacts/reports/alfred_subset_multimodal_report.txt
```

Validation:

```text
Best validation F1: 0.690
Best validation accuracy: 0.550
```

Test set, using `valid_unseen`:

```text
Accuracy: 0.55
Macro F1: 0.52
```

Class-level test results:

```text
Class           Precision   Recall   F1
Not Ambiguous   0.60       0.30    0.40
Ambiguous       0.53       0.80    0.64
```

Confusion matrix:

```text
                    Pred Not Ambiguous   Pred Ambiguous
Actual Not Ambiguous          3                 7
Actual Ambiguous              2                 8
```

## Qualitative Observations

BLIP-2 generated reasonable scene captions for the extracted ALFRED frames. Examples:

```text
Instruction: Go to the basketball, Pick up the basketball.
Caption: a 3d rendering of a bedroom with a bed and a door
Label: Not Ambiguous
Predicted probability ambiguous: 0.494
```

```text
Instruction: Go to the bathtubbasin, Pick up the cloth.
Caption: a 3d rendering of a bathroom with a sink and toilet
Label: Not Ambiguous
Predicted probability ambiguous: 0.487
```

```text
Instruction: Go to the bed, Take the pillow.
Caption: a bed with a laptop on it and a pillow
Label: Not Ambiguous
Predicted probability ambiguous: 0.489
```

The model is already more sensitive to the ambiguous class than the not-ambiguous class. On this tiny test set, ambiguous recall is `0.80`, while not-ambiguous recall is only `0.30`.

## Discussion

These results should be treated as an early smoke-test result, not a final benchmark. The first subset was only 60 examples, so the metrics are noisy and sensitive to the sampled rows.

Still, the run validates the main project hypothesis and implementation path:

- DialFRED can provide ambiguity supervision.
- ALFRED can provide matching visual observations.
- BLIP-2 can extract usable scene-level embeddings and captions.
- The MLP can train on fused image/text features.
- The end-to-end image + instruction pipeline is operational.

The current weakness is that the classifier overpredicts ambiguity on the unseen split. This is expected with a small training set and limited visual variation.

## Next Steps

## Dataset Subset: 300 Rows

A larger 300-row run was completed after the initial smoke test.

```text
Rows: 300
Train: 100
Validation seen: 100
Validation unseen/test: 100
Ambiguous: 150
Not ambiguous: 150
```

Ambiguity type distribution:

```text
appearance: 172
location: 69
direction: 54
other: 5
```

Feature cache:

```text
artifacts/features/alfred_subset_300_multimodal_features.npz
```

Feature shape:

```text
300 x 1538
```

Checkpoint:

```text
artifacts/checkpoints/alfred_subset_300_multimodal_mlp.pt
```

Evaluation report:

```text
artifacts/reports/alfred_subset_300_multimodal_report.txt
```

Validation:

```text
Best validation F1: 0.714
Best validation accuracy: 0.640
```

Test set, using `valid_unseen`:

```text
Accuracy: 0.48
Macro F1: 0.47
```

Class-level test results:

```text
Class           Precision   Recall   F1
Not Ambiguous   0.47       0.38    0.42
Ambiguous       0.48       0.58    0.53
```

Confusion matrix:

```text
                    Pred Not Ambiguous   Pred Ambiguous
Actual Not Ambiguous         19                31
Actual Ambiguous             21                29
```

Caption ambiguity score distribution:

```text
Mean: 0.761
Min: 0.480
Max: 0.911
```

The 300-row model improved validation performance but still overpredicts ambiguity on the unseen split. Ambiguous recall reached `0.58`, but many not-ambiguous examples were incorrectly marked ambiguous.

This suggests the model is picking up a useful ambiguity signal, but the current setup needs better negative examples, more data, and likely better frame selection before test accuracy stabilizes.

## Next Steps

Run a larger subset, such as 500-1000 examples, to get more stable metrics.

Recommended next command:

```powershell
python -m sensor_vlm.extract_alfred_subset --max-rows 500 --output-dir data\alfred_subset_500 --manifest artifacts\features\dialfred_alfred_subset_500_manifest.csv
```

Then build features and train:

```powershell
python -m sensor_vlm.build_features multimodal-manifest --manifest artifacts\features\dialfred_alfred_subset_500_manifest.csv --output artifacts\features\alfred_subset_500_multimodal_features.npz

python -m sensor_vlm.train train-cache --features artifacts\features\alfred_subset_500_multimodal_features.npz --checkpoint artifacts\checkpoints\alfred_subset_500_multimodal_mlp.pt --report artifacts\reports\alfred_subset_500_multimodal_report.txt --epochs 50 --batch-size 16
```

