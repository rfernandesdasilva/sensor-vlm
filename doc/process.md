# Sensor-VLM End-to-End Process

This document explains the full Sensor-VLM workflow from a fresh setup through inference. The project detects whether a natural-language instruction is ambiguous in the context of an image, then produces a simple clarification question when the instruction is predicted to be ambiguous.

## 1. Project Goal

Sensor-VLM combines language supervision from DialFRED with visual observations from ALFRED. DialFRED provides instructions, ambiguity labels, question types, and task metadata. ALFRED provides matching embodied-agent trajectories and raw image frames.

The final model is a small MLP classifier trained on frozen feature vectors:

```text
BLIP-2 pooled Q-Former image embedding
+ instruction text embedding
+ generated caption/VQA text embedding
+ caption similarity and caption ambiguity scalar features
```

At inference time, the model receives:

```text
image + instruction -> feature vector -> ambiguity probability -> label + clarification
```

## 2. Setup

From Windows PowerShell, create or activate the virtual environment and install the package:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The BLIP-2 feature extractor uses `Salesforce/blip2-opt-2.7b`. The first run downloads large Hugging Face model weights. CUDA is used when available; otherwise the code falls back to CPU.

Optional Hugging Face token:

```powershell
$env:HF_TOKEN = "your_huggingface_token"
```

The main local folders are created automatically by the scripts:

```text
data/
artifacts/features/
artifacts/checkpoints/
artifacts/reports/
samples/
```

## 3. DialFRED Label Preparation

DialFRED is the supervision source. The loader downloads `dialfred_human_qa.csv` when needed, cleans split names, removes empty instructions, and converts the `necessary` column into the binary ambiguity label:

```text
necessary = true/yes/1 -> ambiguous = 1
otherwise              -> ambiguous = 0
```

Multiple annotation rows are aggregated into one instruction-level row using:

```text
instruction
split
ambiguous_mean
ambiguous
task_id
trial_id
room_type
task_type
subgoal_start
subgoal_end
subgoal_idx
question_types
questions
answers
```

The DialFRED splits are used consistently:

```text
train        -> training
valid_seen   -> validation
valid_unseen -> test
```

To build and train the text-only baseline in one command:

```powershell
python -m sensor_vlm.train baseline --epochs 50
```

This produces:

```text
artifacts/features/dialfred_text_baseline.npz
artifacts/checkpoints/best_text_mlp.pt
artifacts/reports/text_baseline_report.txt
```

## 4. ALFRED Image Data

ALFRED provides the images needed for multimodal training. For raw-image BLIP-2 extraction, use the full ALFRED archive because `json` and `json_feat` do not contain the raw frames needed by this pipeline.

To download and extract the full dataset:

```powershell
python -m sensor_vlm.download_alfred full --remove-archive
```

This downloads `full_2.1.0.7z`, extracts it under `data/`, and returns a `data/full_2.1.0` directory.

If disk space or time is limited, extract a smaller subset directly from the archive:

```powershell
python -m sensor_vlm.extract_alfred_subset --max-rows 300 --output-dir data\alfred_subset_300 --manifest artifacts\features\dialfred_alfred_subset_300_manifest.csv
```

The subset extractor:

1. Loads DialFRED instruction labels.
2. Samples rows across `train`, `valid_seen`, and `valid_unseen`.
3. Finds likely ALFRED raw frames near the DialFRED subgoal.
4. Extracts only those image files from `full_2.1.0.7z`.
5. Writes a manifest with `image_path`, `instruction`, `ambiguous`, split, and task metadata.

Use `--sampling balanced` for a 50/50 ambiguous/not-ambiguous subset or `--sampling natural` to preserve the DialFRED label skew.

## 5. Linking Full ALFRED Data

When the full ALFRED dataset is already extracted, link DialFRED labels to ALFRED trajectories:

```powershell
python -m sensor_vlm.build_features link-alfred --alfred-data data\full_2.1.0
```

The linker indexes `traj_data.json` files by `task_id` and `trial_id`, merges them with DialFRED labels, and chooses a raw image frame near the matching subgoal. The output manifest defaults to:

```text
artifacts/features/dialfred_alfred_manifest.csv
```

Rows with a populated `image_path` can be used for BLIP-2 feature extraction.

## 6. Multimodal Feature Extraction

Build a multimodal `.npz` feature cache from any manifest that includes at least:

```text
image_path
instruction
ambiguous
split
```

Example using a subset manifest:

```powershell
python -m sensor_vlm.build_features multimodal-manifest --manifest artifacts\features\dialfred_alfred_subset_300_manifest.csv --output artifacts\features\alfred_subset_300_multimodal_features.npz
```

For each row, the extractor:

1. Loads the image from `image_path`.
2. Uses BLIP-2 to generate multiple captions.
3. Computes caption similarity and caption ambiguity score.
4. Asks fixed VQA-style questions about the scene.
5. Extracts pooled Q-Former image embeddings.
6. Encodes the instruction with `all-MiniLM-L6-v2`.
7. Encodes the generated caption plus VQA answers with `all-MiniLM-L6-v2`.
8. Concatenates image, text, caption/VQA, and scalar features.

The feature cache contains:

```text
features -> float32 feature matrix
labels   -> ambiguity labels, if present
splits   -> DialFRED split names, if present
```

A sidecar metadata file is also written next to the cache:

```text
<feature-cache>.metadata.json
```

## 7. Model Training

Train the MLP from a saved feature cache:

```powershell
python -m sensor_vlm.train train-cache --features artifacts\features\alfred_subset_300_multimodal_features.npz --checkpoint artifacts\checkpoints\alfred_subset_300_multimodal_mlp.pt --report artifacts\reports\alfred_subset_300_multimodal_report.txt --epochs 50 --batch-size 16
```

Training uses:

```text
train        -> fit the model
valid_seen   -> select the best checkpoint by validation F1
valid_unseen -> final test report
```

The classifier is an MLP with batch normalization, dropout, class-weighted binary cross entropy, learning-rate reduction on validation F1, and early stopping.

Training writes:

```text
artifacts/checkpoints/*.pt -> best model checkpoint
artifacts/reports/*.txt   -> evaluation report
```

The report includes test accuracy, F1, precision, recall, and a class-level classification report for `Not Ambiguous` and `Ambiguous`.

## 8. Inference

Run inference with an image, an instruction, and a multimodal checkpoint whose input dimension matches the feature cache used for training:

```powershell
python -m sensor_vlm.infer --image samples\example.jpg --instruction "Move that away" --checkpoint artifacts\checkpoints\alfred_subset_300_multimodal_mlp.pt
```

Inference performs the same feature-building steps as training for one image/instruction pair:

```text
image -> BLIP-2 captions, VQA answers, Q-Former embedding
instruction -> sentence-transformer embedding
caption/VQA text -> sentence-transformer embedding
combined vector -> MLP probability
```

The CLI prints:

```text
Instruction
Label
Ambiguity probability
BLIP-2 caption
Caption ambiguity score
Clarification question
```

The clarification is template-based. Pronoun-like instructions such as "that", "it", "there", "this", "those", or "them" ask which object or location the user means. Directional instructions ask whose perspective should be used.

## 9. Notebook Workflow

The notebooks provide exploratory entry points for the same process:

```text
notebooks/01_dialfred_baseline.ipynb    -> DialFRED text baseline
notebooks/02_blip2_image_features.ipynb -> BLIP-2 caption/VQA/image feature inspection
notebooks/03_multimodal_mlp.ipynb       -> multimodal cache and MLP training exploration
```

Use the scripts for repeatable runs and the notebooks for inspection, debugging, and presentation.

## 10. Recommended Start-to-Finish Run

For a manageable multimodal experiment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .

python -m sensor_vlm.train baseline --epochs 50

python -m sensor_vlm.extract_alfred_subset --max-rows 300 --output-dir data\alfred_subset_300 --manifest artifacts\features\dialfred_alfred_subset_300_manifest.csv

python -m sensor_vlm.build_features multimodal-manifest --manifest artifacts\features\dialfred_alfred_subset_300_manifest.csv --output artifacts\features\alfred_subset_300_multimodal_features.npz

python -m sensor_vlm.train train-cache --features artifacts\features\alfred_subset_300_multimodal_features.npz --checkpoint artifacts\checkpoints\alfred_subset_300_multimodal_mlp.pt --report artifacts\reports\alfred_subset_300_multimodal_report.txt --epochs 50 --batch-size 16

python -m sensor_vlm.infer --image samples\example.jpg --instruction "Move that away" --checkpoint artifacts\checkpoints\alfred_subset_300_multimodal_mlp.pt
```

## 11. Current Interpretation

The pipeline is operational end to end. Early multimodal runs validate that DialFRED labels can be paired with ALFRED images, BLIP-2 can produce usable scene features, and the MLP can train on fused image/text vectors.

The main limitation is data quality and scale. Small subsets are useful for smoke testing, but final conclusions should use larger subsets, better frame selection, and class-balanced reporting. The existing result notes are in:

```text
doc/early_results.md
doc/late_results.md
```

