# Ambi3D Image Preparation

## Summary

Ambi3D annotations are public on Hugging Face, but the visual scenes are not bundled with the JSON files. The paper states that Ambi3D is built on ScanNet scenes, so visual baselines require local ScanNet RGB frames or rendered scene views.

## What Was Downloaded

The Ambi3D annotation files are downloaded by:

```powershell
python -m sensor_vlm.prepare_ambi3d_manifest --output artifacts\features\ambi3d_manifest.csv --scene-list-output artifacts\features\ambi3d_required_scenes.txt
```

This produces:

```text
data/ambi3d/ambi3d_train.json
data/ambi3d/ambi3d_test.json
artifacts/features/ambi3d_manifest.csv
artifacts/features/ambi3d_required_scenes.txt
```

## Required Images

ScanNet access is controlled by ScanNet's terms of use. After access is approved, download/export RGB frames for the scene IDs listed in:

```text
artifacts/features/ambi3d_required_scenes.txt
```

The view discovery code supports folders such as:

```text
data/ambi3d_views/scene0000_00/*.jpg
data/scannet/scans/scene0000_00/color/*.jpg
data/scannet/scans/scene0000_00/frames/color/*.jpg
```

## Build A View Manifest

After local views exist, run:

```powershell
python -m sensor_vlm.prepare_ambi3d_manifest --view-root data\ambi3d_views --max-views 4 --output artifacts\features\ambi3d_view_manifest.csv
```

If using a ScanNet export root:

```powershell
python -m sensor_vlm.prepare_ambi3d_manifest --view-root data\scannet\scans --max-views 4 --output artifacts\features\ambi3d_view_manifest.csv
```

The command reports how many Ambi3D rows have at least one scene view.

## Run Visual Baselines

```powershell
python -m sensor_vlm.build_features multimodal-manifest --manifest artifacts\features\ambi3d_view_manifest.csv --output artifacts\features\ambi3d_single_image_features.npz
python -m sensor_vlm.build_features multiview-manifest --manifest artifacts\features\ambi3d_view_manifest.csv --max-views 4 --output artifacts\features\ambi3d_multiview_features.npz
```

Only run these once the manifest reports nonzero rows with scene views.
