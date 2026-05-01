import pandas as pd
from pathlib import Path
from sensor_vlm.build_features_improved import buildImprovedMultimodalCache

manifest_path = Path("artifacts/features/dialfred_alfred_imbalanced_manifest.csv")
output_path = Path("artifacts/features/multimodal_4574_features.npz")

print("Loading manifest...")
manifest = pd.read_csv(manifest_path)
print(f"Samples: {len(manifest)}")

print("Building BLIP-2 multimodal features...")
print("This will take approximately 3 hours")

buildImprovedMultimodalCache(
    manifest=manifest,
    outputPath=output_path,
    imageColumn="image_path",
    instructionColumn="instruction",
    labelColumn="ambiguous",
    splitColumn="split"
)

print(f"\nFeatures saved to: {output_path}")
