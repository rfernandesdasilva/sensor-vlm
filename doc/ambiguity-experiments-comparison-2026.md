# Ambiguity Experiments Comparison 2026

## Status

Updated with measured Ambi3D text and DialFRED/ALFRED target-3000 text results. Ambi3D visual baselines and DialFRED/ALFRED multimodal target-3000 remain blocked by missing views or long-running BLIP-2 extraction failures.

## Experiment Matrix

```text
Experiment                              Dataset          Visual input       Status
clean-balanced-1500                     DialFRED/ALFRED  single frame       measured, actual 290 rows
dialfred-clean-strict-target3000        DialFRED/ALFRED  text only          measured, actual 290 rows
dialfred-clean-relaxed-target3000       DialFRED/ALFRED  text only          measured, actual 290 rows
dialfred-weighted-target3000            DialFRED/ALFRED  text only          measured, actual 3000 rows
ambi3d-text                             Ambi3D           none               measured
ambi3d-evidence-text                    Ambi3D           object metadata    measured, likely label leakage
ambi3d-single-image                     Ambi3D           one scene view     blocked, no local views
ambi3d-multiview                        Ambi3D           pooled views       blocked, no local views
```

## Measured Results

```text
Experiment                         Rows   Accuracy   Macro F1   Balanced Acc.   Not-Ambig Recall   Ambig Recall
Ambi3D text                       22081    0.812      0.811      0.812           0.879              0.746
Ambi3D evidence text              22081    1.000      1.000      1.000           1.000              1.000
DialFRED strict target3000          290    0.471      0.454      0.471           0.294              0.647
DialFRED relaxed target3000         290    0.471      0.421      0.471           0.176              0.765
DialFRED weighted target3000       3000    0.890      0.563      0.605           0.294              0.916
```

The Ambi3D evidence-text result should not be treated as a valid benchmark because the evidence fields include `ambiguity_type`, which is empty for not-ambiguous rows and therefore likely leaks the answer.

The DialFRED weighted target-3000 result reaches 3,000 rows but mostly learns the ambiguous majority class. Its high accuracy is less informative than macro F1 and not-ambiguous recall.

## Metrics To Compare

```text
Rows actually used
Train/valid/test class balance
Accuracy
Macro F1
Balanced accuracy
Not-ambiguous recall
Ambiguous recall
Confusion matrix
```

## Current Measured Anchor

From `doc/clean-balanced-1500-results.md`, the strict clean-balanced target-1500 run produced 290 actual rows:

```text
Text-only:   accuracy 0.471, macro F1 0.46
Multimodal:  accuracy 0.588, macro F1 0.58
```

## How To Interpret The Next Runs

If strict target-3000 still caps near 290 rows, the bottleneck is data availability and image matching, not the target size.

If relaxed target-3000 improves actual row count and macro F1, the current strict thresholds are probably too conservative for scaling.

If weighted target-3000 improves accuracy but not macro F1 or not-ambiguous recall, it should not be presented as a stronger ambiguity detector.

If Ambi3D multi-view beats Ambi3D text-only, that is the cleanest evidence that explicit scene evidence improves ambiguity detection beyond language priors.
