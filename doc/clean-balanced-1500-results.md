# Clean Balanced Sensor-VLM Results

## Summary

This run implements the clean-data v2 experiment:

1. Filter noisy DialFRED labels using strict annotator-agreement thresholds.
2. Link the remaining labels to ALFRED trajectory metadata.
3. Select an existing raw frame nearest the subgoal midpoint.
4. Balance labels after image validation, inside each split.
5. Train text-only and multimodal models on the same clean balanced rows.

The requested target was 1500 rows, but strict filtering made not-ambiguous examples the limiting class. The largest locally available clean balanced dataset was 290 rows.

## Data Preparation

Label filter:

```text
ambiguous_mean <= 0.25 -> Not Ambiguous
ambiguous_mean >= 0.75 -> Ambiguous
```

Dataset:

```text
Rows: 290
Feature shape: 290 x 1538 for multimodal features

Train:        212 -> 106 not ambiguous, 106 ambiguous
Valid seen:    44 ->  22 not ambiguous,  22 ambiguous
Valid unseen:  34 ->  17 not ambiguous,  17 ambiguous
```

## Artifacts

```text
Manifest:            artifacts/features/clean-balanced-1500_manifest.csv
Text features:       artifacts/features/clean-balanced-1500_text_features.npz
Multimodal features: artifacts/features/clean-balanced-1500_multimodal_features.npz

Text checkpoint:       artifacts/checkpoints/clean-balanced-1500_text_mlp.pt
Multimodal checkpoint: artifacts/checkpoints/clean-balanced-1500_multimodal_mlp.pt

Text report:       artifacts/reports/clean-balanced-1500-text-results.txt
Multimodal report: artifacts/reports/clean-balanced-1500-multimodal-results.txt
```

## Results

```text
Model        Accuracy   Macro F1   Not-Ambig Recall   Ambig Recall
Text-only    0.471      0.46       0.35               0.59
Multimodal   0.588      0.58       0.71               0.47
```

Class-level multimodal results:

```text
Class           Precision   Recall   F1    Support
Not Ambiguous   0.57       0.71    0.63      17
Ambiguous       0.62       0.47    0.53      17
```

## Interpretation

This is the clearest evidence so far that the model is learning useful signal from the visual features. On the same clean balanced test split, the multimodal model improves over the text-only baseline:

```text
Accuracy: 0.471 -> 0.588
Macro F1: 0.46  -> 0.58
```

The important change is that the multimodal model no longer only overpredicts ambiguity. It improves not-ambiguous recall from `0.35` to `0.71`, meaning visual context helps the classifier recognize clearer instructions.

## Caveats

The test set is small: only 34 examples. These results are promising, but they should be presented as an early clean-data result rather than a final benchmark.

The strict label filter leaves very few not-ambiguous examples, especially in `valid_unseen`. To scale this result, the next step is to extract more raw ALFRED frames for clean not-ambiguous rows, or relax thresholds slightly while still excluding `ambiguous_mean = 0.5` examples.

## Presentation-Safe Claim

> After cleaning noisy labels and evaluating on the same balanced subset, the multimodal model outperformed the text-only baseline: accuracy improved from about 47% to 59%, and macro F1 improved from 0.46 to 0.58. The result is small-scale, but it suggests that visual context helps the model distinguish clear instructions from ambiguous ones.
