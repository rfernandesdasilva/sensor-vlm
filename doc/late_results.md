# Sensor-VLM Late-Stage Results Target

## Important Note

These are **projected late-stage results**, not measured results yet.

This file is intended as a realistic target and report scaffold for what we expect after scaling the multimodal experiment beyond the current 300-row subset. Do not present these numbers as final experimental results unless a later run actually achieves them.

## Expected Setup

The late-stage experiment should use a larger DialFRED + ALFRED multimodal subset.

Target configuration:

```text
Rows: 1000-3000
Train: 60-70%
Validation: 15-20%
Test: 15-20%
Labels: balanced or class-weighted
Model: frozen BLIP-2 + sentence-transformer text encoder + MLP classifier
Feature dim: 1538
Trainable parameters: ~428K
```

The input feature vector remains:

```text
pooled BLIP-2 Q-Former image embedding
+ instruction embedding
+ generated caption/VQA embedding
+ caption similarity and ambiguity scalar features
```

## Realistic Target Metrics

The current 300-row run reached:

```text
Test accuracy: 0.52
Macro F1: 0.491
Ambiguous recall: 0.76
```

A realistic target after scaling data and improving frame selection is:

```text
Accuracy: 0.65-0.72
Macro F1: 0.62-0.70
Ambiguous precision: 0.65-0.75
Ambiguous recall: 0.70-0.82
Not-ambiguous recall: 0.55-0.68
```

An optimistic but still plausible result would be:

```text
Accuracy: 0.73-0.76
Macro F1: 0.70-0.74
```

We should not expect `0.85+` multimodal accuracy without substantially stronger visual grounding, more task-specific supervision, or fine-tuning.

## Example Late-Stage Results Table

The following table is a **target format** for the final report.

```text
Model                         Accuracy   Macro F1   Ambiguous Recall   Not-Ambiguous Recall
Text-only MLP baseline         0.78       0.54       0.89               0.18
Multimodal MLP, 300 rows        0.52       0.49       0.76               0.28
Multimodal MLP, larger subset   0.65-0.72  0.62-0.70  0.70-0.82          0.55-0.68
```

The key expected improvement is not necessarily higher weighted F1 than the text-only baseline. The more important improvement is better **balanced performance** and stronger not-ambiguous recognition on a balanced multimodal test set.

## Expected Qualitative Results

For visually grounded ambiguous instructions, the multimodal model should identify missing referents.

Example:

```text
Instruction: Move that away.
Scene: multiple visible objects on a table
Expected prediction: Ambiguous
Expected clarification: Which object do you want moved?
```

For visually grounded specific instructions, the model should be less likely to over-call ambiguity.

Example:

```text
Instruction: Pick up the red mug from the counter.
Scene: one red mug visible on the counter
Expected prediction: Not Ambiguous
```

For visually missing target objects, the model should mark the instruction as ambiguous or ungrounded.

Example:

```text
Instruction: Pick up the apple.
Scene: no apple clearly visible
Expected prediction: Ambiguous
Expected clarification: I do not see the apple. Where is it?
```

## Why These Results Are Realistic

The text-only proof of concept achieved high weighted F1, but the class distribution was heavily skewed toward ambiguous examples. The original report showed weak non-ambiguous performance:

```text
Not Ambiguous F1: 0.20
Ambiguous F1: 0.88
```

Our multimodal subset is balanced, so the metric target should focus on macro F1 and per-class recall rather than weighted F1 alone.

The 300-row multimodal run already shows a useful pattern:

```text
Ambiguous recall is relatively high.
Not-ambiguous recall remains weak.
The model still overpredicts ambiguity.
```

With more rows, the model should see more clear negative examples and more visual variety. This should improve the not-ambiguous class, but only if image-instruction alignment is good enough.

## What Needs To Improve

The most important next improvements are:

1. Use a larger subset, at least 1000 rows if time permits.
2. Improve frame selection by sampling multiple frames from the subgoal instead of one frame near `subgoal_start`.
3. Filter borderline labels, especially rows with `ambiguous_mean` near `0.5`.
4. Add object-aware text features from DialFRED columns such as `verb`, `noun1`, and `noun2`.
5. Compare against the text-only baseline on the same balanced subset.

## Report Framing

A realistic final claim would be:

> We implemented an end-to-end multimodal ambiguity detection pipeline linking DialFRED ambiguity annotations with ALFRED visual observations. Early results show that frozen BLIP-2 features and instruction embeddings can support image-conditioned ambiguity classification, but performance remains limited by frame selection, label noise, and small-scale multimodal supervision.

A stronger claim, if later metrics support it, would be:

> Scaling the multimodal subset and improving image-instruction alignment improves macro F1 and reduces the model's tendency to overpredict ambiguity, especially on clear not-ambiguous instructions.

## Do Not Claim Yet

Do not claim the following unless confirmed by actual experiments:

```text
The multimodal model beats the text-only baseline overall.
The model reliably resolves visual references.
The model generalizes strongly to unseen ALFRED scenes.
The projected 0.65-0.72 accuracy range has been achieved.
```

