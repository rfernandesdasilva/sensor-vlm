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

## Project Layout

- `src/sensor_vlm/data.py`: DialFRED download, cleaning, and instruction-level ambiguity labels.
- `src/sensor_vlm/blip2_features.py`: BLIP-2 captions, VQA probes, caption variance, and Q-Former embeddings.
- `src/sensor_vlm/features.py`: text baseline and multimodal feature-cache builders.
- `src/sensor_vlm/model.py`: MLP classifier, training loop, evaluation, and checkpoint loading.
- `src/sensor_vlm/train.py`: command-line training workflows.
- `src/sensor_vlm/build_features.py`: ALFRED linking and multimodal cache creation.
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

