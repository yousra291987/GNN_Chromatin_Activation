"""
train.py — Training loop for GNN node classification.

Handles:
  - Class-weighted loss (critical: ~818 positives vs ~8000+ negatives)
  - Early stopping on validation AUPRC
  - Learning rate scheduling
  - Logging and checkpointing
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score
)


def compute_class_weights(data):
    """Compute inverse-frequency weights for the labeled nodes."""
    mask = data.y >= 0
    labels = data.y[mask]
    counts = torch.bincount(labels)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum() * len(weights)
    return weights


def train_epoch(model, data, optimizer, class_weights=None, device="cpu"):
    """Run one training epoch."""
    model.train()
    data = data.to(device)

    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)

    # Only compute loss on labeled training nodes
    mask = data.train_mask & (data.y >= 0)
    loss = F.cross_entropy(out[mask], data.y[mask], weight=class_weights)

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask_name="val_mask", device="cpu"):
    """
    Evaluate model on a given mask.
    Returns dict of metrics.
    """
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_index, data.edge_attr)

    mask = getattr(data, mask_name) & (data.y >= 0)
    if mask.sum() == 0:
        return {}

    probs = F.softmax(out[mask], dim=1)
    preds = out[mask].argmax(dim=1)
    labels = data.y[mask]

    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()

    metrics = {
        "loss": F.cross_entropy(out[mask], labels).item(),
        "accuracy": (preds == labels).float().mean().item(),
        "f1": f1_score(labels_np, preds_np, average="binary", zero_division=0),
        "precision": precision_score(labels_np, preds_np, zero_division=0),
        "recall": recall_score(labels_np, preds_np, zero_division=0),
    }

    # AUC metrics (need probability of positive class)
    if len(np.unique(labels_np)) > 1:
        probs_pos = probs[:, 1].cpu().numpy()
        metrics["auroc"] = roc_auc_score(labels_np, probs_pos)
        metrics["auprc"] = average_precision_score(labels_np, probs_pos)
    else:
        metrics["auroc"] = 0.0
        metrics["auprc"] = 0.0

    return metrics


def train(
    model,
    data,
    epochs: int = 200,
    lr: float = 0.001,
    weight_decay: float = 5e-4,
    patience: int = 30,
    save_dir: str = "models",
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Full training loop with early stopping.

    Returns
    -------
    dict with training history and best metrics
    """
    data = data.to(device)
    model = model.to(device)

    class_weights = compute_class_weights(data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5, min_lr=1e-6
    )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    best_val_auprc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_auprc": [],
               "val_f1": [], "val_precision": [], "val_recall": []}

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, data, optimizer, class_weights, device)

        # Validate
        val_metrics = evaluate(model, data, "val_mask", device)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics.get("loss", 0))
        history["val_auroc"].append(val_metrics.get("auroc", 0))
        history["val_auprc"].append(val_metrics.get("auprc", 0))
        history["val_f1"].append(val_metrics.get("f1", 0))
        history["val_precision"].append(val_metrics.get("precision", 0))
        history["val_recall"].append(val_metrics.get("recall", 0))

        # LR scheduling
        scheduler.step(val_metrics.get("auprc", 0))

        # Early stopping on AUPRC (better than AUROC for imbalanced data)
        if val_metrics.get("auprc", 0) > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path / "best_model.pt")
        else:
            patience_counter += 1

        if verbose and epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val AUROC: {val_metrics.get('auroc', 0):.4f} | "
                f"Val AUPRC: {val_metrics.get('auprc', 0):.4f} | "
                f"Val F1: {val_metrics.get('f1', 0):.4f} | "
                f"Val Recall: {val_metrics.get('recall', 0):.4f}"
            )

        if patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(save_path / "best_model.pt", weights_only=True))
    test_metrics = evaluate(model, data, "test_mask", device)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Best model (epoch {best_epoch}):")
        print(f"  Test AUROC:     {test_metrics.get('auroc', 0):.4f}")
        print(f"  Test AUPRC:     {test_metrics.get('auprc', 0):.4f}")
        print(f"  Test F1:        {test_metrics.get('f1', 0):.4f}")
        print(f"  Test Precision: {test_metrics.get('precision', 0):.4f}")
        print(f"  Test Recall:    {test_metrics.get('recall', 0):.4f}")

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_auprc": best_val_auprc,
        "test_metrics": test_metrics,
    }
