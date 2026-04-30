from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_instruction_labels, print_dataset_summary
from .features import build_text_baseline_cache, load_feature_cache
from .model import (
    TrainingConfig,
    binary_report,
    evaluate_binary,
    split_by_dialfred,
    train_binary_classifier,
)
from .paths import CHECKPOINTS_DIR, FEATURES_DIR, REPORTS_DIR, ensure_project_dirs


def train_from_cache(
    feature_cache: str | Path,
    *,
    checkpoint_path: str | Path,
    report_path: str | Path | None = None,
    epochs: int = 50,
    batch_size: int = 64,
) -> dict[str, object]:
    cache = load_feature_cache(feature_cache)
    features = cache["features"].astype(np.float32)
    labels = cache["labels"].astype(np.int64)
    splits = cache["splits"].astype(str)
    x_train, y_train, x_val, y_val, x_test, y_test = split_by_dialfred(features, labels, splits)

    config = TrainingConfig(epochs=epochs, batch_size=batch_size)
    model, history, best_state = train_binary_classifier(
        x_train,
        y_train,
        x_val,
        y_val,
        config=config,
        checkpoint_path=checkpoint_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(x_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    num_pos = int(y_train.sum())
    num_neg = int(len(y_train) - num_pos)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32).to(device)
    )
    metrics = evaluate_binary(model, test_loader, criterion, device)
    report = binary_report(metrics["labels"], metrics["preds"])

    if report_path:
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "\n".join(
                [
                    "Sensor-VLM Evaluation Report",
                    f"Feature cache: {feature_cache}",
                    f"Best validation F1: {best_state.get('val_f1')}",
                    "",
                    f"Test accuracy: {metrics['accuracy']:.4f}",
                    f"Test F1: {metrics['f1']:.4f}",
                    f"Test macro F1: {metrics['macro_f1']:.4f}",
                    f"Test balanced accuracy: {metrics['balanced_accuracy']:.4f}",
                    f"Test precision: {metrics['precision']:.4f}",
                    f"Test recall: {metrics['recall']:.4f}",
                    f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix'].tolist()}",
                    "",
                    report,
                ]
            ),
            encoding="utf-8",
        )

    print(report)
    return {"history": history, "best_state": best_state, "test_metrics": metrics}


def command_prepare_text(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    labels = load_instruction_labels(args.csv, download=not args.no_download)
    if args.max_rows:
        labels = labels.head(args.max_rows).copy()
    print_dataset_summary(labels)
    output = build_text_baseline_cache(labels, args.output, batch_size=args.batch_size)
    print(f"Saved text baseline cache: {output}")


def command_train(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    train_from_cache(
        args.features,
        checkpoint_path=args.checkpoint,
        report_path=args.report,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


def command_baseline(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    labels = load_instruction_labels(args.csv, download=not args.no_download)
    if args.max_rows:
        labels = labels.head(args.max_rows).copy()
    print_dataset_summary(labels)
    cache_path = Path(args.cache)
    build_text_baseline_cache(labels, cache_path, batch_size=args.batch_size)
    train_from_cache(
        cache_path,
        checkpoint_path=args.checkpoint,
        report_path=args.report,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Sensor-VLM ambiguity classifiers.")
    subparsers = parser.add_subparsers(required=True)

    prepare = subparsers.add_parser("prepare-text", help="Cache text-only DialFRED embeddings.")
    prepare.add_argument("--csv", default=None, help="Optional local DialFRED CSV path.")
    prepare.add_argument("--output", default=FEATURES_DIR / "dialfred_text_baseline.npz")
    prepare.add_argument("--batch-size", type=int, default=128)
    prepare.add_argument("--max-rows", type=int, default=None)
    prepare.add_argument("--no-download", action="store_true")
    prepare.set_defaults(func=command_prepare_text)

    train = subparsers.add_parser("train-cache", help="Train from a saved .npz feature cache.")
    train.add_argument("--features", required=True)
    train.add_argument("--checkpoint", default=CHECKPOINTS_DIR / "best_ambiguity_mlp.pt")
    train.add_argument("--report", default=REPORTS_DIR / "evaluation_report.txt")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=64)
    train.set_defaults(func=command_train)

    baseline = subparsers.add_parser("baseline", help="Prepare text baseline and train it.")
    baseline.add_argument("--csv", default=None, help="Optional local DialFRED CSV path.")
    baseline.add_argument("--cache", default=FEATURES_DIR / "dialfred_text_baseline.npz")
    baseline.add_argument("--checkpoint", default=CHECKPOINTS_DIR / "best_text_mlp.pt")
    baseline.add_argument("--report", default=REPORTS_DIR / "text_baseline_report.txt")
    baseline.add_argument("--epochs", type=int, default=50)
    baseline.add_argument("--batch-size", type=int, default=64)
    baseline.add_argument("--max-rows", type=int, default=None)
    baseline.add_argument("--no-download", action="store_true")
    baseline.set_defaults(func=command_baseline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

