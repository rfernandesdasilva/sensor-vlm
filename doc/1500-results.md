# Sensor-VLM 1500-Row Results

## Summary

This run trained the multimodal Sensor-VLM classifier on a 1500-row DialFRED + ALFRED image manifest.

The pipeline completed end to end:

1. Built a 1500-row ALFRED image manifest.
2. Extracted BLIP-2 multimodal features.
3. Trained the MLP ambiguity classifier.
4. Wrote the evaluation report to `artifacts/reports/1500-results.txt`.

## Dataset

```text
Rows: 1500
Feature shape: 1500 x 1538

Train:        764
Valid seen:   385
Valid unseen: 351

Ambiguous:     1014
Not ambiguous: 486
```

The final matched subset is not perfectly balanced. Ambiguous examples remain the majority class.

## Artifacts

```text
Manifest:    artifacts/features/dialfred_alfred_subset_1500_manifest.csv
Features:    artifacts/features/1500-results_multimodal_features.npz
Metadata:    artifacts/features/1500-results_multimodal_features.metadata.json
Checkpoint:  artifacts/checkpoints/1500-results_multimodal_mlp.pt
Report:      artifacts/reports/1500-results.txt
```

## Quantitative Results

```text
Best validation F1: 0.718

Test accuracy:  0.641
Test F1:        0.772
Test precision: 0.769
Test recall:    0.775
```

Class-level results on `valid_unseen`:

```text
Class           Precision   Recall   F1    Support
Not Ambiguous   0.16       0.16    0.16      76
Ambiguous       0.77       0.77    0.77     275

Accuracy:       0.64
Macro F1:       0.47
Weighted F1:    0.64
```

## Interpretation

The larger 1500-row run improved overall accuracy compared with the earlier 300-row run, but the model still struggles with the not-ambiguous class.

The strongest presentation-safe claim is:

> Scaling to 1500 rows improved overall test accuracy to about 64%, and the model remains sensitive to ambiguous instructions. However, class-level results show that it still overpredicts ambiguity and needs better negative examples or balancing before we can claim robust ambiguity detection.

## Caveat

Do not present the weighted F1 alone as the main result. Because the test split contains many more ambiguous examples than not-ambiguous examples, the weighted score hides weak performance on the not-ambiguous class. Use macro F1 and class-level recall when discussing limitations.
