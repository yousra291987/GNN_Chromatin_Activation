#!/usr/bin/env python3
"""
run.py — Project 3e: Enriched Feature GNN

Builds on project3d (TransformerConv + continuous edges) but with
12 engineered node features instead of 3 raw ChIP signals.

Feature groups:
  - 3 raw ChIP-seq marks (H3K27ac, H3K4me2, H3K27me3)
  - 3 ratio features (ac/me3, me2/me3, ac/me2)
  - 1 region width (from genomic coordinates)
  - 1 is_promoter (binary)
  - 3 graph topology (degree, mean neighbor H3K27ac, mean neighbor H3K27me3)

Usage:
    # Step 1: Extract coordinates (run once)
    Rscript scripts/extract_coordinates.R

    # Step 2: Train
    python run.py --config configs/transformer_enriched.yaml
    python run.py --config configs/transformer_no_coords.yaml   # without coords
    python run.py --compare
"""

import argparse
import yaml
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import build_graph, create_masks, load_nodes
from src.model import get_model
from src.train import train
from src.evaluate import (
    plot_training_curves, plot_confusion_matrix, plot_roc_pr_curves,
    plot_embeddings_umap, get_top_predictions, plot_graph_statistics
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single(cfg: dict, label: str = ""):
    """Train and evaluate one configuration."""
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    print(f"\n{'='*60}")
    print(f"  Project 3e: Enriched Feature GNN — {label}")
    print(f"  Model: {model_cfg['name']}")
    print(f"  Ratios: {data_cfg.get('use_ratios', True)}")
    print(f"  Coordinates: {data_cfg.get('use_coordinates', True)}")
    print(f"  Graph features: {data_cfg.get('use_graph_features', True)}")
    print(f"{'='*60}\n")

    # ---- Load data ----
    print("Building graph...")
    data = build_graph(
        nodes_path=data_cfg["nodes_path"],
        edges_path=data_cfg["edges_path"],
        coordinates_path=data_cfg.get("coordinates_path"),
        min_score=data_cfg.get("min_score", 0.0),
        timepoint=data_cfg["timepoint"],
        add_self_loops=data_cfg.get("add_self_loops", True),
        connected_only=data_cfg.get("connected_only", False),
        edge_transform=data_cfg.get("edge_transform", "log"),
        use_ratios=data_cfg.get("use_ratios", True),
        use_coordinates=data_cfg.get("use_coordinates", True),
        use_graph_features=data_cfg.get("use_graph_features", True),
        use_node_type=data_cfg.get("use_node_type", True),
    )
    data = create_masks(
        data,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=data_cfg["seed"],
    )
    nodes_df = load_nodes(data_cfg["nodes_path"])

    if data_cfg.get("connected_only", False):
        id_to_row = {nid: i for i, nid in enumerate(nodes_df["ID"])}
        keep_indices = [id_to_row[nid] for nid in data.node_ids if nid in id_to_row]
        nodes_df = nodes_df.iloc[keep_indices].reset_index(drop=True)

    # ---- Plot graph stats ----
    plot_graph_statistics(data, nodes_df, save_dir=out_cfg["figures_dir"])

    # ---- Build model ----
    model_kwargs = {
        "in_channels": data.num_node_features,
        "hidden_channels": model_cfg["hidden_channels"],
        "num_layers": model_cfg["num_layers"],
        "num_classes": 2,
        "dropout": model_cfg["dropout"],
    }
    if model_cfg["name"] in ("transformer", "gat"):
        model_kwargs["heads"] = model_cfg.get("heads", 4)
    if model_cfg["name"] == "transformer":
        model_kwargs["edge_dim"] = model_cfg.get("edge_dim", 1)

    model = get_model(model_cfg["name"], **model_kwargs)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_cfg['name']} | Features: {data.num_node_features} | Params: {total_params:,}")

    # ---- Train ----
    device = train_cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    results = train(
        model=model,
        data=data,
        epochs=train_cfg["epochs"],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        patience=train_cfg["patience"],
        save_dir=out_cfg["save_dir"],
        device=device,
    )

    # ---- Evaluate and visualize ----
    print("\nGenerating figures...")
    plot_training_curves(results["history"], save_dir=out_cfg["figures_dir"])
    plot_confusion_matrix(model, data, save_dir=out_cfg["figures_dir"], device=device)
    plot_roc_pr_curves(model, data, save_dir=out_cfg["figures_dir"], device=device)
    plot_embeddings_umap(model, data, nodes_df, save_dir=out_cfg["figures_dir"], device=device)
    predictions = get_top_predictions(model, data, nodes_df,
                                      save_dir=out_cfg["figures_dir"], device=device)

    # Save results
    summary = {
        "model": model_cfg["name"],
        "label": label,
        "num_features": data.num_node_features,
        "feature_names": data.feature_names,
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "params": total_params,
        "best_epoch": results["best_epoch"],
        "test_metrics": results["test_metrics"],
    }
    Path(out_cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
    with open(Path(out_cfg["save_dir"]) / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_comparison(device_override=None):
    """Run available configs and compare."""
    configs = []

    # Check which configs to run
    if Path("data/input/coordinates.tsv").exists():
        configs.append(("configs/transformer_enriched.yaml", "Enriched (12 features)"))
    configs.append(("configs/transformer_no_coords.yaml", "Enriched no coords (11 features)"))

    all_results = []
    for config_path, label in configs:
        cfg = load_config(config_path)
        if device_override:
            cfg["training"]["device"] = device_override
        summary = run_single(cfg, label=label)
        all_results.append(summary)

    # ---- Comparison table ----
    print(f"\n{'='*85}")
    print(f"  COMPARISON: Enriched Features vs Previous Best")
    print(f"{'='*85}")
    header = f"{'Experiment':<30} {'Feat':>5} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}"
    print(header)
    print(f"{'-'*85}")

    # Baselines
    print(f"{'project3b (3 feat, GAT)':<30} {'3':>5} {'0.7506':>8} {'0.3823':>8} {'0.423':>8} {'0.279':>8} {'0.876':>8}")
    print(f"{'project3d (3 feat, Transf.)':<30} {'3':>5} {'0.7659':>8} {'0.3993':>8} {'0.445':>8} {'0.306':>8} {'0.816':>8}")

    for r in all_results:
        tm = r["test_metrics"]
        print(f"{r['label']:<30} {r['num_features']:>5} "
              f"{tm.get('auroc', 0):>8.4f} {tm.get('auprc', 0):>8.4f} "
              f"{tm.get('f1', 0):>8.4f} {tm.get('precision', 0):>8.4f} "
              f"{tm.get('recall', 0):>8.4f}")

    # Save comparison
    Path("figures").mkdir(parents=True, exist_ok=True)
    with open("figures/comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ---- Comparison bar chart ----
    labels_plot = ["project3b\n(3 feat)", "project3d\n(3 feat)"] + \
                  [r["label"].replace(" (", "\n(") for r in all_results]
    aurocs = [0.7506, 0.7659] + [r["test_metrics"].get("auroc", 0) for r in all_results]
    auprcs = [0.3823, 0.3993] + [r["test_metrics"].get("auprc", 0) for r in all_results]
    f1s = [0.423, 0.445] + [r["test_metrics"].get("f1", 0) for r in all_results]
    precs = [0.279, 0.306] + [r["test_metrics"].get("precision", 0) for r in all_results]

    x = np.arange(len(labels_plot))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 1.5*width, aurocs, width, label="AUROC", color="#457B9D")
    ax.bar(x - 0.5*width, auprcs, width, label="AUPRC", color="#E63946")
    ax.bar(x + 0.5*width, f1s, width, label="F1", color="#2A9D8F")
    ax.bar(x + 1.5*width, precs, width, label="Precision", color="#F4A261")
    ax.set_ylabel("Score")
    ax.set_title("Feature Enrichment Impact — Broad Activation Prediction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/comparison_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved comparison chart to figures/comparison_chart.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project 3e: Enriched Feature GNN")
    parser.add_argument("--config", default=None, help="Config YAML file")
    parser.add_argument("--compare", action="store_true", help="Run configs and compare")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    args = parser.parse_args()

    if args.compare:
        run_comparison(device_override=args.device)
    elif args.config:
        cfg = load_config(args.config)
        if args.device:
            cfg["training"]["device"] = args.device
        label = Path(args.config).stem
        run_single(cfg, label=label)
    else:
        print("Usage:")
        print("  python run.py --config configs/transformer_enriched.yaml")
        print("  python run.py --config configs/transformer_no_coords.yaml")
        print("  python run.py --compare")
