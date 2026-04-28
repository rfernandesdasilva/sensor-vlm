from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset


class AmbiguityDataset(Dataset):
    """Dataset for precomputed embeddings and binary ambiguity labels."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


class AmbiguityMLP(nn.Module):
    """MLP head for binary or multi-class ambiguity classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits


@dataclass
class TrainingConfig:
    batch_size: int = 64
    hidden_dim: int = 256
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 10
    threshold: float = 0.5


def split_by_dialfred(
    embeddings: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask = splits == "train"
    val_mask = splits == "valid_seen"
    test_mask = splits == "valid_unseen"
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError(
            "Expected DialFRED splits train, valid_seen, and valid_unseen. "
            f"Found: {sorted(set(splits.tolist()))}"
        )
    return (
        embeddings[train_mask],
        labels[train_mask],
        embeddings[val_mask],
        labels[val_mask],
        embeddings[test_mask],
        labels[test_mask],
    )


def evaluate_binary(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, object]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0

    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            logits = model(embeddings)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) >= threshold).long()
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.long().cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    return {
        "loss": total_loss / max(len(y_true), 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "preds": y_pred,
        "labels": y_true,
    }


def train_binary_classifier(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    *,
    config: TrainingConfig | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[AmbiguityMLP, dict[str, list[float]], dict[str, object]]:
    config = config or TrainingConfig()
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    drop_last = len(train_embeddings) > config.batch_size and len(train_embeddings) % config.batch_size == 1
    train_loader = DataLoader(
        AmbiguityDataset(train_embeddings, train_labels),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        AmbiguityDataset(val_embeddings, val_labels),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = AmbiguityMLP(
        input_dim=train_embeddings.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(torch_device)

    num_pos = int(train_labels.sum())
    num_neg = int(len(train_labels) - num_pos)
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32).to(torch_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_f1": [],
        "val_loss": [],
        "val_f1": [],
        "val_accuracy": [],
    }
    best_state: dict[str, object] | None = None
    best_val_f1 = -1.0
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_preds: list[int] = []
        train_true: list[int] = []

        for embeddings, labels in train_loader:
            embeddings = embeddings.to(torch_device)
            labels = labels.to(torch_device)
            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(labels)
            train_preds.extend((torch.sigmoid(logits) >= config.threshold).long().cpu().tolist())
            train_true.extend(labels.long().cpu().tolist())

        val_metrics = evaluate_binary(model, val_loader, criterion, torch_device, config.threshold)
        scheduler.step(float(val_metrics["f1"]))

        train_loss = train_loss_sum / max(len(train_true), 1)
        train_f1 = f1_score(train_true, train_preds, zero_division=0)
        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_f1"].append(float(val_metrics["f1"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))

        if float(val_metrics["f1"]) > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            stale_epochs = 0
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": best_val_f1,
                "val_accuracy": float(val_metrics["accuracy"]),
                "input_dim": train_embeddings.shape[1],
                "config": config.__dict__,
            }
            if checkpoint_path:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
    return model, history, best_state or {}


def load_binary_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> tuple[AmbiguityMLP, dict[str, object]]:
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    config = checkpoint.get("config", {})
    model = AmbiguityMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.3)),
    ).to(torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_probability(
    model: nn.Module,
    features: np.ndarray,
    *,
    device: str | torch.device | None = None,
) -> float:
    torch_device = torch.device(device or next(model.parameters()).device)
    model.eval()
    tensor = torch.tensor(features, dtype=torch.float32).reshape(1, -1).to(torch_device)
    with torch.no_grad():
        return float(torch.sigmoid(model(tensor)).item())


def binary_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(
        y_true,
        y_pred,
        target_names=["Not Ambiguous", "Ambiguous"],
        zero_division=0,
    )

