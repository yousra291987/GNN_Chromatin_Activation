"""
evaluate.py — Visualization and analysis for broad activation GNN.

Labels: non-active → active (1) vs non-active → non-active (0)
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
from pathlib import Path


def plot_training_curves(history: dict, save_dir: str = "figures"):
    """Plot training loss and validation metrics over epochs."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train", alpha=0.8)
    axes[0].plot(history["val_loss"], label="Val", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].legend()

    axes[1].plot(history["val_auroc"], label="AUROC", alpha=0.8)
    axes[1].plot(history["val_auprc"], label="AUPRC", alpha=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation AUC Metrics")
    axes[1].legend()

    axes[2].plot(history["val_f1"], label="F1", alpha=0.8)
    axes[2].plot(history["val_precision"], label="Precision", alpha=0.8)
    axes[2].plot(history["val_recall"], label="Recall", alpha=0.8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Validation Classification Metrics")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_path / 'training_curves.png'}")


def plot_confusion_matrix(model, data, mask_name="test_mask", save_dir="figures", device="cpu"):
    """Plot confusion matrix for test set."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)

    mask = getattr(data, mask_name) & (data.y >= 0)
    preds = out[mask].argmax(dim=1).cpu().numpy()
    labels = data.y[mask].cpu().numpy()

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    tick_labels = ["Stayed Non-Active", "Activated"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix \u2014 Broad Activation Prediction")
    plt.tight_layout()
    plt.savefig(save_path / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(classification_report(labels, preds,
                                target_names=tick_labels))


def plot_roc_pr_curves(model, data, mask_name="test_mask", save_dir="figures", device="cpu"):
    """Plot ROC and Precision-Recall curves."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)

    mask = getattr(data, mask_name) & (data.y >= 0)
    probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
    labels = data.y[mask].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    baseline = labels.sum() / len(labels)
    axes[1].plot(recall, precision, lw=2, label=f"AUPRC = {pr_auc:.3f}")
    axes[1].axhline(y=baseline, color="k", linestyle="--", alpha=0.3,
                    label=f"Baseline = {baseline:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_embeddings_umap(model, data, nodes_df, save_dir="figures", device="cpu"):
    """UMAP of learned node embeddings — 6 panels across 2 figures."""
    try:
        from umap import UMAP
    except ImportError:
        print("Install umap-learn for embedding visualization: pip install umap-learn")
        return

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    data = data.to(device)
    with torch.no_grad():
        embeddings = model.get_embeddings(data.x, data.edge_index, data.edge_attr)
    embeddings = embeddings.cpu().numpy()

    # Only plot labeled nodes
    mask = (data.y >= 0).cpu().numpy()
    emb_sub = embeddings[mask]
    labels = data.y.cpu().numpy()[mask]
    states = np.array(nodes_df["category_E8.5_WT"].tolist())[mask]
    node_types = np.array(data.node_types)[mask]
    nodes_sub = nodes_df.iloc[np.where(mask)[0]]
    h3k27ac = pd.to_numeric(nodes_sub["H3K27ac_progenitors_E8.5_WT"], errors="coerce").fillna(0).values
    h3k27me3 = pd.to_numeric(nodes_sub["H3K27me3_progenitors_E8.5_WT"], errors="coerce").fillna(0).values
    h3k4me2 = pd.to_numeric(nodes_sub["H3K4me2_progenitors_E8.5_WT"], errors="coerce").fillna(0).values
    expression = pd.to_numeric(nodes_sub["expression.progenitors_E8.5_WT"], errors="coerce").fillna(0).values
    expr_log = np.log2(expression + 1)

    # GAT predicted activation probability
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        pred_probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()[mask]

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    coords = reducer.fit_transform(emb_sub)

    STATE_COLORS = {
        "bivalent":  "#E63946",
        "poised":    "#F4A261",
        "Pc-only":   "#457B9D",
        "primed":    "#2A9D8F",
        "active":    "#06D6A0",
        "negative":  "#ADB5BD",
        "unknown":   "#DEE2E6",
    }

    # Helper for percentile clipping
    def clip_q(vals, lo=1, hi=99):
        return np.clip(vals, np.percentile(vals, lo), np.percentile(vals, hi))

    # === Figure 1: Activation, State, Node Type ===
    fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6))

    neg_mask = labels == 0
    pos_mask = labels == 1
    axes1[0].scatter(coords[neg_mask, 0], coords[neg_mask, 1],
                     c="#457B9D", s=5, alpha=0.4, label="Stayed Non-Active")
    axes1[0].scatter(coords[pos_mask, 0], coords[pos_mask, 1],
                     c="#E63946", s=8, alpha=0.8, label="Activated")
    axes1[0].set_title("Activation Label", fontsize=13, fontweight="bold")
    axes1[0].legend(fontsize=9, loc="best", markerscale=2)

    unique_states = sorted(set(states))
    for state in unique_states:
        s_mask = states == state
        color = STATE_COLORS.get(state, "#ADB5BD")
        axes1[1].scatter(coords[s_mask, 0], coords[s_mask, 1],
                         c=color, s=5, alpha=0.5, label=state)
    axes1[1].set_title("E8.5 Epigenetic State", fontsize=13, fontweight="bold")
    axes1[1].legend(fontsize=9, loc="best", markerscale=2)

    prom_mask = node_types == "promoter"
    enh_mask = node_types == "Enhancer"
    axes1[2].scatter(coords[enh_mask, 0], coords[enh_mask, 1],
                     c="#2A9D8F", s=5, alpha=0.4, label="Enhancer")
    axes1[2].scatter(coords[prom_mask, 0], coords[prom_mask, 1],
                     c="#E76F51", s=5, alpha=0.4, label="Promoter")
    axes1[2].set_title("Node Type", fontsize=13, fontweight="bold")
    axes1[2].legend(fontsize=9, loc="best", markerscale=2)

    fig1.suptitle("GNN Node Embeddings \u2014 Broad Activation (UMAP)", fontsize=15, fontweight="bold", y=1.02)
    fig1.tight_layout()
    fig1.savefig(save_path / "embeddings_umap.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved UMAP (3-panel) to {save_path / 'embeddings_umap.png'}")

    # === Figure 2: ChIP-seq signals ===
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))

    vals = clip_q(h3k27ac)
    sc1 = axes2[0].scatter(coords[:, 0], coords[:, 1], c=vals, cmap="YlOrRd", s=5, alpha=0.6)
    axes2[0].set_title("H3K27ac (active mark)", fontsize=13, fontweight="bold")
    plt.colorbar(sc1, ax=axes2[0], shrink=0.8, label="ChIP signal")

    vals = clip_q(h3k27me3)
    sc2 = axes2[1].scatter(coords[:, 0], coords[:, 1], c=vals, cmap="PuBu", s=5, alpha=0.6)
    axes2[1].set_title("H3K27me3 (Polycomb mark)", fontsize=13, fontweight="bold")
    plt.colorbar(sc2, ax=axes2[1], shrink=0.8, label="ChIP signal")

    vals = clip_q(h3k4me2)
    sc3 = axes2[2].scatter(coords[:, 0], coords[:, 1], c=vals, cmap="YlGn", s=5, alpha=0.6)
    axes2[2].set_title("H3K4me2 (Trithorax mark)", fontsize=13, fontweight="bold")
    plt.colorbar(sc3, ax=axes2[2], shrink=0.8, label="ChIP signal")

    fig2.suptitle("GNN Embeddings Colored by Histone Marks (E8.5)", fontsize=15, fontweight="bold", y=1.02)
    fig2.tight_layout()
    fig2.savefig(save_path / "embeddings_umap_chipseq.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved UMAP (ChIP-seq) to {save_path / 'embeddings_umap_chipseq.png'}")

    # === Figure 3: Expression + GAT prediction probability ===
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

    vals = clip_q(expr_log)
    sc4 = axes3[0].scatter(coords[:, 0], coords[:, 1], c=vals,
                           cmap="magma", s=5, alpha=0.6)
    axes3[0].set_title("Expression log2(RPKM+1) at E8.5", fontsize=13, fontweight="bold")
    plt.colorbar(sc4, ax=axes3[0], shrink=0.8, label="log2(RPKM+1)")

    sc5 = axes3[1].scatter(coords[:, 0], coords[:, 1], c=pred_probs,
                           cmap="RdYlBu_r", s=5, alpha=0.6, vmin=0, vmax=1)
    axes3[1].set_title("GAT Predicted P(Activation)", fontsize=13, fontweight="bold")
    plt.colorbar(sc5, ax=axes3[1], shrink=0.8, label="P(activated)")

    fig3.suptitle("Expression & Model Prediction", fontsize=15, fontweight="bold", y=1.02)
    fig3.tight_layout()
    fig3.savefig(save_path / "embeddings_umap_expr_pred.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved UMAP (expression + prediction) to {save_path / 'embeddings_umap_expr_pred.png'}")


def get_top_predictions(model, data, nodes_df, k=50, save_dir="figures", device="cpu"):
    """Rank genes by predicted activation probability."""
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()

    mask = (data.y >= 0).cpu().numpy().astype(bool)
    results = pd.DataFrame({
        "node_id": np.array(data.node_ids)[mask],
        "gene_name": nodes_df["gene_name"].values[mask],
        "node_type": np.array(data.node_types)[mask],
        "state_E8.5": nodes_df["category_E8.5_WT"].values[mask],
        "state_E10.5": nodes_df["category_FNP_E10.5_WT"].values[mask],
        "true_label": data.y.cpu().numpy()[mask],
        "pred_prob": probs[mask],
    })

    results = results.sort_values("pred_prob", ascending=False)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    results.to_csv(save_path / "activation_predictions.csv", index=False)

    print(f"\nTop {k} predicted gene activations:")
    top_k = results.head(k)
    print(top_k[["gene_name", "node_type", "state_E8.5", "state_E10.5",
                  "true_label", "pred_prob"]].to_string(index=False))

    return results


def plot_graph_statistics(data, nodes_df, save_dir="figures"):
    """Plot degree distribution and other graph statistics."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    edge_index = data.edge_index.cpu().numpy()
    degrees = np.bincount(edge_index[0], minlength=data.num_nodes)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(degrees, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Node Degree")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Degree Distribution (mean={degrees.mean():.1f})")
    axes[0].set_yscale("log")

    prom_mask = np.array(nodes_df["node_type"] == "promoter")
    enh_mask = ~prom_mask
    axes[1].hist(degrees[prom_mask], bins=30, alpha=0.6, label="Promoter", color="coral")
    axes[1].hist(degrees[enh_mask], bins=30, alpha=0.6, label="Enhancer", color="steelblue")
    axes[1].set_xlabel("Node Degree")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Degree by Node Type")
    axes[1].legend()
    axes[1].set_yscale("log")

    mask = (data.y >= 0).cpu().numpy()
    labels = data.y.cpu().numpy()[mask]
    degs_labeled = degrees[mask]
    axes[2].boxplot([degs_labeled[labels == 0], degs_labeled[labels == 1]],
                    labels=["Stayed Non-Active", "Activated"])
    axes[2].set_ylabel("Node Degree")
    axes[2].set_title("Degree vs Activation")

    plt.tight_layout()
    plt.savefig(save_path / "graph_statistics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved graph statistics to {save_path / 'graph_statistics.png'}")
