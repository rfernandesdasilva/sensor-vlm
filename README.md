# Sensor-VLM

Sensor-VLM detects ambiguous vision-language instructions from an image and a natural-language command. It is based on the project POCs in `doc/`: a DialFRED ambiguity MLP and a BLIP-2 image feature extraction notebook.

For the full start-to-finish workflow, see `doc/process.md`.

## Setup

From Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The BLIP-2 extractor uses `Salesforce/blip2-opt-2.7b` in fp16 on CUDA when available. Your RTX 4080 Super should be suitable for frozen feature extraction. The first BLIP-2 run downloads large Hugging Face model weights.

Hugging Face authentication is optional for public models, but setting a token can improve rate limits:

```powershell
$env:HF_TOKEN = "your_huggingface_token"
```

The code disables the Windows symlink cache warning by default. If you want the most disk-efficient Hugging Face cache, enable Windows Developer Mode or run the shell as administrator.

## Dataset Setup

Sensor-VLM uses three dataset sources:

1. **DialFRED** for language instructions and ambiguity labels.
2. **ALFRED** for RGB trajectory frames paired with DialFRED rows.
3. **Ambi3D** for an external ambiguity benchmark. Ambi3D annotations are public, but visual scene images require separate ScanNet access.

Dataset links:

```text
DialFRED repository:     https://github.com/xfgao/DialFRED
DialFRED CSV used here:  https://raw.githubusercontent.com/xfgao/DialFRED/main/data/dialfred_human_qa.csv
ALFRED website:          https://askforalfred.com/
ALFRED repository:       https://github.com/askforalfred/alfred
ALFRED full images:      https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/full_2.1.0.7z
ALFRED JSON metadata:    https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_2.1.0.7z
Ambi3D Hugging Face:     https://huggingface.co/datasets/jiayuttkx/ambi3d
Ambi3D train JSON:       https://huggingface.co/datasets/jiayuttkx/ambi3d/resolve/main/ambi3d_train.json
Ambi3D test JSON:        https://huggingface.co/datasets/jiayuttkx/ambi3d/resolve/main/ambi3d_test.json
ScanNet website:         http://www.scan-net.org/
```

Expected local layout:

```text
sensor-vlm/
  data/
    alfred_subset/              small extracted ALFRED subset
    alfred_subset_1500/         larger extracted ALFRED subset
    full_2.1.0/                 full ALFRED raw-image dataset, if unpacked
    json_2.1.0/                 ALFRED trajectory JSON files
    ambi3d/                     downloaded Ambi3D annotation JSON files
    ambi3d_views/               optional exported Ambi3D/ScanNet scene views
    scannet/scans/              optional ScanNet export root
  artifacts/
    features/                   generated manifests and feature caches
    checkpoints/                trained MLP checkpoints
    reports/                    measured experiment reports
```

### DialFRED

DialFRED is downloaded automatically by the project data loader when no local CSV is provided. The text baseline can therefore be run directly:

```powershell
python -m sensor_vlm.train baseline --epochs 50
```

The loader normalizes DialFRED rows into instruction-level ambiguity examples with `instruction`, `ambiguous`, and `split` fields.

No manual DialFRED folder is required unless you want to use a custom CSV. If you do, place it anywhere convenient and pass it with `--csv`.

Direct source used by the loader:

```text
https://raw.githubusercontent.com/xfgao/DialFRED/main/data/dialfred_human_qa.csv
```

### ALFRED

ALFRED provides the raw images used for multimodal DialFRED experiments. To download and prepare the full ALFRED data, run:

```powershell
python -m sensor_vlm.download_alfred full --remove-archive
```

By default, this project expects the full ALFRED raw-image folder at:

```text
data/full_2.1.0/
```

The downloader uses these ALFRED archive links:

```text
full raw images:  https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/full_2.1.0.7z
trajectory JSON:  https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_2.1.0.7z
prebuilt feats:   https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_feat_2.1.0.7z
```

Then link DialFRED rows to ALFRED trajectories:

```powershell
python -m sensor_vlm.build_features link-alfred --alfred-data data\full_2.1.0
```

For faster experiments, you can extract only a subset of ALFRED raw images from the archive instead of unpacking the full dataset:

```powershell
python -m sensor_vlm.extract_alfred_subset --max-rows 60
python -m sensor_vlm.build_features multimodal-manifest --manifest artifacts\features\dialfred_alfred_subset_manifest.csv
```

Subset extraction writes image folders under `data/`, such as:

```text
data/alfred_subset/
data/alfred_subset_300/
data/alfred_subset_1500/
```

The same subset path was used for the 60-row, 300-row, 1500-row, balanced 1500-row, and clean-balanced DialFRED/ALFRED experiments.

### Ambi3D and ScanNet

Ambi3D annotations can be downloaded and normalized into the same manifest format:

```powershell
python -m sensor_vlm.prepare_ambi3d_manifest --output artifacts\features\ambi3d_manifest.csv --scene-list-output artifacts\features\ambi3d_required_scenes.txt
python -m sensor_vlm.build_features text-manifest --manifest artifacts\features\ambi3d_manifest.csv --output artifacts\features\ambi3d_text_features.npz
```

This stores Ambi3D annotation JSON files under:

```text
data/ambi3d/
```

The Ambi3D annotation files come from:

```text
https://huggingface.co/datasets/jiayuttkx/ambi3d
```

This supports Ambi3D text-only baselines immediately. Ambi3D visual baselines require local ScanNet scene images or rendered views, because the public Ambi3D release does not bundle the images. After obtaining ScanNet access and exporting views, place them under a supported folder such as:

```text
data/ambi3d_views/scene0000_00/*.jpg
data/scannet/scans/scene0000_00/color/*.jpg
data/scannet/scans/scene0000_00/frames/color/*.jpg
```

Then build a view manifest and run visual baselines:

```powershell
python -m sensor_vlm.prepare_ambi3d_manifest --view-root data\ambi3d_views --max-views 4 --output artifacts\features\ambi3d_view_manifest.csv
python -m sensor_vlm.build_features multimodal-manifest --manifest artifacts\features\ambi3d_view_manifest.csv --output artifacts\features\ambi3d_single_image_features.npz
python -m sensor_vlm.build_features multiview-manifest --manifest artifacts\features\ambi3d_view_manifest.csv --max-views 4 --output artifacts\features\ambi3d_multiview_features.npz
```

See `doc/ambi3d-image-prep.md` for more details.

## Artifacts and Reports

Generated experiment outputs are written under `artifacts/`:

```text
artifacts/features/      manifests, feature caches, and metadata
artifacts/checkpoints/   trained MLP checkpoints
artifacts/reports/       machine-readable evaluation reports
```

The `artifacts/reports/` folder is the source of truth for measured metrics. Human-readable summaries and writeups live in `doc/`, but the final reported numbers should be checked against the corresponding `.txt` report in `artifacts/reports/`.

## Project Layout

- `src/sensor_vlm/data.py`: DialFRED download, cleaning, and instruction-level ambiguity labels.
- `src/sensor_vlm/blip2_features.py`: BLIP-2 captions, VQA probes, caption variance, and Q-Former embeddings.
- `src/sensor_vlm/features.py`: text baseline and multimodal feature-cache builders.
- `src/sensor_vlm/model.py`: MLP classifier, training loop, evaluation, and checkpoint loading.
- `src/sensor_vlm/train.py`: command-line training workflows.
- `src/sensor_vlm/build_features.py`: ALFRED linking and multimodal cache creation.
- `src/sensor_vlm/prepare_ambi3d_manifest.py`: Ambi3D download and manifest normalization.
- `src/sensor_vlm/infer.py`: image + instruction inference with a template clarification question.
- `notebooks/`: notebook entry points for the baseline, BLIP-2 feature inspection, and multimodal MLP training.

## Text Baseline

This reproduces the POC path using DialFRED labels and `all-MiniLM-L6-v2` instruction embeddings:

```powershell
python -m sensor_vlm.train baseline --epochs 50
```

Outputs are written under `artifacts/features`, `artifacts/checkpoints`, and `artifacts/reports`.

## BLIP-2 Image Feature Demo

Run the BLIP-2 notebook or use the module directly:

```python
from sensor_vlm.blip2_features import Blip2FeatureExtractor

extractor = Blip2FeatureExtractor()
features = extractor.extract_features("samples/example.jpg")
print(features.caption)
print(features.caption_ambiguity_score)
```

## Multimodal Training

For a manifest containing `image_path`, `instruction`, `ambiguous`, and `split` columns:

```powershell
python -m sensor_vlm.build_features multimodal-manifest --manifest data/my_manifest.csv
python -m sensor_vlm.train train-cache --features artifacts/features/multimodal_features.npz --checkpoint artifacts/checkpoints/best_multimodal_mlp.pt
```

For text-only training from any normalized manifest:

```powershell
python -m sensor_vlm.build_features text-manifest --manifest artifacts\features\my_manifest.csv --output artifacts\features\my_text_features.npz
```

The feature vector is:

```text
pooled BLIP-2 Q-Former image embedding
+ instruction text embedding
+ generated caption/VQA text embedding
+ caption similarity and caption ambiguity scalar features
```

## ALFRED/DialFRED Image Linkage

DialFRED gives the ambiguity labels. ALFRED provides visual trajectories. After downloading ALFRED data, link them with:

```powershell
python -m sensor_vlm.download_alfred full --remove-archive
python -m sensor_vlm.build_features link-alfred --alfred-data C:\path\to\alfred\data\full_2.1.0
```

If raw ALFRED images are present, the linker adds `image_path` values that can be used to build a multimodal cache. If only `json_feat` is present, the manifest still records trajectory metadata, but BLIP-2 raw-image extraction needs the full raw image files.

To avoid extracting the full ALFRED archive, create a small image subset directly from `full_2.1.0.7z`:

```powershell
python -m sensor_vlm.extract_alfred_subset --max-rows 60
python -m sensor_vlm.build_features multimodal-manifest --manifest artifacts/features/dialfred_alfred_subset_manifest.csv
```

## Inference

Use a multimodal checkpoint whose input dimension matches the multimodal feature cache:

```powershell
python -m sensor_vlm.infer --image samples/example.jpg --instruction "Move that away" --checkpoint artifacts/checkpoints/best_multimodal_mlp.pt
```

The result includes the ambiguity label, probability, BLIP-2 scene caption, caption-variance score, and a clarification question when the instruction is ambiguous.

## Ambi3D Baselines

Ambi3D can be normalized into the same `instruction`, `ambiguous`, and `split` cache format:

```powershell
python -m sensor_vlm.prepare_ambi3d_manifest --output artifacts\features\ambi3d_manifest.csv
python -m sensor_vlm.build_features text-manifest --manifest artifacts\features\ambi3d_manifest.csv --output artifacts\features\ambi3d_text_features.npz
```

If local scene renders are available, add `--view-root data\ambi3d_views` and use `multimodal-manifest` or `multiview-manifest` for visual baselines.
See `doc/ambi3d-image-prep.md` for the ScanNet scene list and image-folder conventions.

