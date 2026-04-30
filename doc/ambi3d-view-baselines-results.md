# Ambi3D View Baselines Results

## Status

Partially measured.

Ambi3D is the best external dataset fit for this project because it directly evaluates visual/3D instruction ambiguity. The normalized manifest maps Ambi3D fields into the existing Sensor-VLM cache format.

## Dataset

```text
Source: jiayuttkx/ambi3d on Hugging Face
Rows: 22,081 total
Fields used: scene_id, instruction_id, question, answer, ambiguity_type, object_id, object_names
Label: answer -> ambiguous
Splits: source test -> valid_unseen; source train is scene-split into train and valid_seen
```

## Manifest Command

```powershell
python -m sensor_vlm.prepare_ambi3d_manifest --output artifacts\features\ambi3d_manifest.csv
```

If local scene renderings are available, pass one or more view roots. The builder records `image_path` for single-view runs and `image_paths` for multi-view runs.

```powershell
python -m sensor_vlm.prepare_ambi3d_manifest --view-root data\ambi3d_views --max-views 4 --output artifacts\features\ambi3d_view_manifest.csv
```

## Baseline Commands

```powershell
python -m sensor_vlm.build_features text-manifest --manifest artifacts\features\ambi3d_manifest.csv --output artifacts\features\ambi3d_text_features.npz
python -m sensor_vlm.build_features text-manifest --manifest artifacts\features\ambi3d_manifest.csv --text-columns instruction evidence_text --output artifacts\features\ambi3d_evidence_text_features.npz
python -m sensor_vlm.build_features multimodal-manifest --manifest artifacts\features\ambi3d_view_manifest.csv --output artifacts\features\ambi3d_single_image_features.npz
python -m sensor_vlm.build_features multiview-manifest --manifest artifacts\features\ambi3d_view_manifest.csv --max-views 4 --output artifacts\features\ambi3d_multiview_features.npz
```

## Training Commands

```powershell
python -m sensor_vlm.train train-cache --features artifacts\features\ambi3d_text_features.npz --checkpoint artifacts\checkpoints\ambi3d_text_mlp.pt --report artifacts\reports\ambi3d-view-text-results.txt --epochs 50 --batch-size 64
python -m sensor_vlm.train train-cache --features artifacts\features\ambi3d_single_image_features.npz --checkpoint artifacts\checkpoints\ambi3d_single_image_mlp.pt --report artifacts\reports\ambi3d-view-single-image-results.txt --epochs 50 --batch-size 16
python -m sensor_vlm.train train-cache --features artifacts\features\ambi3d_multiview_features.npz --checkpoint artifacts\checkpoints\ambi3d_multiview_mlp.pt --report artifacts\reports\ambi3d-view-multiview-results.txt --epochs 50 --batch-size 8
```

## Results

```text
Model                  Accuracy   Macro F1   Balanced Acc.   Not-Ambig Recall   Ambig Recall
Instruction only        0.812      0.811      0.812           0.879              0.746
Evidence text           1.000      1.000      1.000           1.000              1.000
Single-view image       not run    not run    not run         not run            not run
Multi-view pooled       not run    not run    not run         not run            not run
```

Instruction-only artifacts:

```text
Manifest:       artifacts/features/ambi3d_manifest.csv
Feature cache:  artifacts/features/ambi3d_text_features.npz
Checkpoint:     artifacts/checkpoints/ambi3d_text_mlp.pt
Report:         artifacts/reports/ambi3d-view-text-results.txt
```

Evidence-text artifacts:

```text
Feature cache:  artifacts/features/ambi3d_evidence_text_features.npz
Checkpoint:     artifacts/checkpoints/ambi3d_evidence_text_mlp.pt
Report:         artifacts/reports/ambi3d-view-evidence-text-results.txt
```

The evidence-text result is diagnostic only. It likely leaks the label because `ambiguity_type` is populated for ambiguous rows and empty for not-ambiguous rows. The valid measured Ambi3D baseline is therefore the instruction-only model.

## Caveat

The Hugging Face Ambi3D table provides text/object metadata. Visual baselines require local scene renderings or exported views whose paths can be discovered from `scene_id`.

No local `data/ambi3d_views` directory was available during this run, so the single-view and multi-view BLIP-2 baselines could not be executed.

Image preparation instructions are in `doc/ambi3d-image-prep.md`. The scene list needed for ScanNet export can be generated with:

```powershell
python -m sensor_vlm.prepare_ambi3d_manifest --output artifacts\features\ambi3d_manifest.csv --scene-list-output artifacts\features\ambi3d_required_scenes.txt
```
