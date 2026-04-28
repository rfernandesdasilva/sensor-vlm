from __future__ import annotations

import argparse

import pandas as pd

from .alfred_linker import link_dialfred_to_alfred
from .data import load_instruction_labels
from .features import build_multimodal_cache_from_manifest
from .paths import FEATURES_DIR, ensure_project_dirs


def command_multimodal_manifest(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    manifest = pd.read_csv(args.manifest)
    if args.max_rows:
        manifest = manifest.head(args.max_rows).copy()
    output = build_multimodal_cache_from_manifest(
        manifest,
        args.output,
        image_column=args.image_column,
        instruction_column=args.instruction_column,
    )
    print(f"Saved multimodal feature cache: {output}")


def command_link_alfred(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    labels = load_instruction_labels(args.csv, download=not args.no_download)
    if args.max_rows:
        labels = labels.head(args.max_rows).copy()
    manifest = link_dialfred_to_alfred(labels, args.alfred_data, output_manifest=args.output)
    linked = int(manifest["image_path"].notna().sum()) if "image_path" in manifest else 0
    print(f"Saved ALFRED manifest: {args.output}")
    print(f"Rows with raw image paths: {linked:,} / {len(manifest):,}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Sensor-VLM feature caches.")
    subparsers = parser.add_subparsers(required=True)

    link = subparsers.add_parser("link-alfred", help="Link DialFRED labels to local ALFRED data.")
    link.add_argument("--alfred-data", required=True, help="Path to ALFRED data root.")
    link.add_argument("--csv", default=None, help="Optional local DialFRED CSV path.")
    link.add_argument("--output", default=FEATURES_DIR / "dialfred_alfred_manifest.csv")
    link.add_argument("--max-rows", type=int, default=None)
    link.add_argument("--no-download", action="store_true")
    link.set_defaults(func=command_link_alfred)

    multimodal = subparsers.add_parser(
        "multimodal-manifest",
        help="Build multimodal .npz features from a manifest with image_path and instruction.",
    )
    multimodal.add_argument("--manifest", required=True)
    multimodal.add_argument("--output", default=FEATURES_DIR / "multimodal_features.npz")
    multimodal.add_argument("--image-column", default="image_path")
    multimodal.add_argument("--instruction-column", default="instruction")
    multimodal.add_argument("--max-rows", type=int, default=None)
    multimodal.set_defaults(func=command_multimodal_manifest)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

