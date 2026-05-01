import pandas as pd
import shutil
from pathlib import Path

DATA_DIR = Path("data")
EXTRACTED_DIR = DATA_DIR / "full_2.1.0"
OUTPUT_DIR = DATA_DIR / "alfred_subset_imbalanced"
LABELED_CSV = DATA_DIR / "dialfred_with_ambiguity_labels.csv"

print("Loading labeled data...")
df = pd.read_csv(LABELED_CSV)
print(f"Total samples: {len(df)}")
print(f"Ambiguous: {(df['ambiguous'] == 1).sum()}")
print(f"Not ambiguous: {(df['ambiguous'] == 0).sum()}")

print("\nExtracting all samples...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

extracted = 0
manifest_rows = []

for idx, row in df.iterrows():
    split = row['split']
    task_id = row['task_id']
    trial_id = row['trial_id']
    
    split_dir = EXTRACTED_DIR / split
    
    found = False
    for task_folder in split_dir.glob("*"):
        pattern = "*" + trial_id + "*"
        for trial_folder in task_folder.glob(pattern):
            image_dir = trial_folder / "raw_images"
            if image_dir.exists():
                images = sorted(image_dir.glob("*.jpg"))
                if images:
                    filename = task_id + "_" + trial_id + ".jpg"
                    dest_file = OUTPUT_DIR / filename
                    shutil.copy2(images[0], dest_file)
                    
                    manifest_rows.append({
                        'task_id': task_id + "_" + trial_id,
                        'split': split,
                        'instruction': row['instruction'],
                        'ambiguous': row['ambiguous'],
                        'image_path': str(dest_file)
                    })
                    
                    extracted += 1
                    if extracted % 200 == 0:
                        print(f"Extracted {extracted}")
                    found = True
                    break
        if found:
            break

print(f"\nExtracted {extracted} files")

manifest_df = pd.DataFrame(manifest_rows)
manifest_path = Path("artifacts/features/dialfred_alfred_imbalanced_manifest.csv")
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_df.to_csv(manifest_path, index=False)

print(f"\nManifest saved to: {manifest_path}")
print("\nSplit distribution:")
print(manifest_df['split'].value_counts())
print(f"\nAmbiguous: {manifest_df['ambiguous'].sum()}")
print(f"Not ambiguous: {(~manifest_df['ambiguous'].astype(bool)).sum()}")
