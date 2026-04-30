# Sensor-VLM 1500 Balanced Results

## Summary

This run retrained the multimodal MLP using a balanced subset of the completed 1500-row BLIP-2 feature cache. No new BLIP-2 extraction was required.

The purpose of this run was to evaluate the classifier under a fairer 50/50 label distribution.

## Dataset

```text
Rows: 972
Feature shape: 972 x 1538

Train:        600  -> 300 not ambiguous, 300 ambiguous
Valid seen:   220  -> 110 not ambiguous, 110 ambiguous
Valid unseen: 152  -> 76 not ambiguous, 76 ambiguous
```

## Artifacts

```text
Features:    artifacts/features/1500-balanced-results_multimodal_features.npz
Metadata:    artifacts/features/1500-balanced-results_multimodal_features.metadata.json
Checkpoint:  artifacts/checkpoints/1500-balanced-results_multimodal_mlp.pt
Report:      artifacts/reports/1500-balanced-results.txt
```

## Quantitative Results

```text
Best validation F1: 0.584

Test accuracy:  0.415
Test F1:        0.524
Test precision: 0.441
Test recall:    0.645
```

Class-level results on the balanced `valid_unseen` split:

```text
Class           Precision   Recall   F1    Support
Not Ambiguous   0.34       0.18    0.24      76
Ambiguous       0.44       0.64    0.52      76

Accuracy:       0.41
Macro F1:       0.38
Weighted F1:    0.38
```

## Comparison With Imbalanced 1500-Row Run

```text
Run                     Test Set Balance       Accuracy   Macro F1
1500-results            76 not / 275 ambig      0.641      0.47
1500-balanced-results   76 not / 76 ambig       0.415      0.38
```

The balanced run makes the model's weakness clearer. The imbalanced run looked stronger because the test set had many more ambiguous examples, and the model is better at predicting the ambiguous class.

## Interpretation

The balanced evaluation is more honest for the project goal. It shows that the current model still overpredicts ambiguity and has weak recall for clear, not-ambiguous instructions.

Presentation-safe claim:

> When evaluated on a balanced test set, the model's accuracy drops to about 41%, showing that the earlier 64% result was partly helped by class imbalance. The model detects many ambiguous cases but still needs better negative examples, cleaner labels, and stronger image-instruction alignment before it can reliably distinguish clear from ambiguous instructions.
