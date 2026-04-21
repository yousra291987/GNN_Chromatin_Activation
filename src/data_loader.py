"""
data_loader.py — Build graph with ENRICHED node features.

Feature engineering strategy:
  From existing ChIP-seq (3 marks):
    1. H3K27ac, H3K4me2, H3K27me3 (raw, standardized)
    2. H3K27ac / H3K27me3 ratio  — active vs repressive balance
    3. H3K4me2 / H3K27me3 ratio  — trithorax vs polycomb balance
    4. H3K27ac / H3K4me2 ratio   — active enhancer vs poised mark

  From genomic coordinates:
    5. log2(region_width)         — regulatory element size
    6. is_promoter                — binary node type

  From graph topology (precomputed):
    7. log2(degree + 1)           — connectivity in interaction network
    8. mean neighbor H3K27ac      — neighborhood active signal
    9. mean neighbor H3K27me3     — neighborhood repressive signal

  Total: 12 features (vs 3 in project3b/3d)

Edge weights: continuous CHiCAGO scores (log-transformed), same as project3d.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from pathlib import Path


# ---- Column names ----
FEATURE_COLS_E85 = [
    "H3K27ac_progenitors_E8.5_WT",
    "H3K4me2_progenitors_E8.5_WT",
    "H3K27me3_progenitors_E8.5_WT",
]
FEATURE_COLS_FNP = [
    "H3K27ac_FNP_E10.5_WT",
    "H3K4me2_FNP_E10.5_WT",
    "H3K27me3_FNP_E10.5_WT",
]
EXPRESSION_E85 = "expression.progenitors_E8.5_WT"
EXPRESSION_FNP = "expression.FNP_E10.5_WT"

NON_ACTIVE_STATES = {"poised", "bivalent", "Pc-only", "primed", "negative"}
ACTIVE_STATE = "active"


def load_nodes(nodes_path: str) -> pd.DataFrame:
    df = pd.read_csv(nodes_path, sep="\t")
    df["category_E8.5_WT"] = df["category_E8.5_WT"].fillna("unknown")
    df["category_FNP_E10.5_WT"] = df["category_FNP_E10.5_WT"].fillna("unknown")
    for col in FEATURE_COLS_E85 + FEATURE_COLS_FNP + [EXPRESSION_E85, EXPRESSION_FNP]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def load_edges(edges_path: str) -> pd.DataFrame:
    df = pd.read_csv(edges_path, sep="\t")
    df["progenitors_E8.5"] = pd.to_numeric(df["progenitors_E8.5"], errors="coerce").fillna(0.0)
    df["FNP_E10.5"] = pd.to_numeric(df["FNP_E10.5"], errors="coerce").fillna(0.0)
    return df


def safe_ratio(a, b, eps=0.01):
    """Compute log2 ratio: log2((a + eps) / (|b| + eps)). Handles negatives."""
    return np.log2((np.abs(a) + eps) / (np.abs(b) + eps))


def build_graph(
    nodes_path: str,
    edges_path: str,
    coordinates_path: str = None,
    min_score: float = 0.0,
    timepoint: str = "E8.5",
    add_self_loops: bool = True,
    connected_only: bool = False,
    edge_transform: str = "log",
    use_ratios: bool = True,
    use_coordinates: bool = True,
    use_graph_features: bool = True,
    use_node_type: bool = True,
    **kwargs,
) -> Data:
    """
    Build graph with enriched node features.

    Feature groups (all standardized to mean=0, std=1):
      - ChIP-seq: 3 raw marks
      - Ratios: 3 log-ratios between marks
      - Coordinates: log2(width)
      - Node type: is_promoter binary
      - Graph: degree, mean neighbor signals
    """
    nodes_df = load_nodes(nodes_path)
    edges_df = load_edges(edges_path)

    # --- Node ID mapping ---
    node_ids = nodes_df["ID"].tolist()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n_nodes = len(nodes_df)

    # --- Select timepoint columns ---
    if timepoint == "E8.5":
        feat_cols = FEATURE_COLS_E85
        score_col = "progenitors_E8.5"
    else:
        feat_cols = FEATURE_COLS_FNP
        score_col = "FNP_E10.5"

    h3k27ac_col, h3k4me2_col, h3k27me3_col = feat_cols

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    feature_list = []
    feature_names = []

    # --- 1. Raw ChIP-seq signals (3 features) ---
    chip_raw = nodes_df[feat_cols].values.astype(np.float32)
    feature_list.append(chip_raw)
    feature_names.extend(["H3K27ac", "H3K4me2", "H3K27me3"])
    print(f"  [features] 3 ChIP-seq signals")

    # --- 2. ChIP-seq ratio features (3 features) ---
    if use_ratios:
        h3k27ac = nodes_df[h3k27ac_col].values.astype(np.float32)
        h3k4me2 = nodes_df[h3k4me2_col].values.astype(np.float32)
        h3k27me3 = nodes_df[h3k27me3_col].values.astype(np.float32)

        ratio_ac_me3 = safe_ratio(h3k27ac, h3k27me3)   # active vs repressive
        ratio_me2_me3 = safe_ratio(h3k4me2, h3k27me3)   # trithorax vs polycomb
        ratio_ac_me2 = safe_ratio(h3k27ac, h3k4me2)     # active enhancer signal

        ratios = np.column_stack([ratio_ac_me3, ratio_me2_me3, ratio_ac_me2])
        feature_list.append(ratios)
        feature_names.extend(["ratio_ac/me3", "ratio_me2/me3", "ratio_ac/me2"])
        print(f"  [features] 3 ChIP-seq ratio features")

    # --- 3. Genomic coordinates (1 feature: region width) ---
    if use_coordinates and coordinates_path and Path(coordinates_path).exists():
        coords_df = pd.read_csv(coordinates_path, sep="\t")
        # Merge by ID
        coords_map = dict(zip(coords_df["ID"], coords_df["width"]))
        widths = np.array([coords_map.get(nid, 1000) for nid in node_ids], dtype=np.float32)
        log_widths = np.log2(widths + 1).reshape(-1, 1)
        feature_list.append(log_widths)
        feature_names.append("log2_width")
        print(f"  [features] 1 coordinate feature (log2 region width)")
    elif use_coordinates:
        print(f"  [features] coordinates_path not found, skipping width feature")

    # --- 4. Node type: is_promoter (1 feature) ---
    if use_node_type:
        is_promoter = (nodes_df["node_type"] == "promoter").values.astype(np.float32).reshape(-1, 1)
        feature_list.append(is_promoter)
        feature_names.append("is_promoter")
        print(f"  [features] 1 node type feature (is_promoter)")

    # =========================================================================
    # BUILD EDGES (continuous weights, same as project3d)
    # =========================================================================
    edges_df_filtered = edges_df[edges_df[score_col] > min_score].copy()

    src_indices = []
    dst_indices = []
    edge_weights = []

    for _, row in edges_df_filtered.iterrows():
        bait = row["bait_id"]
        other = row["other_end_id"]
        score = row[score_col]
        if bait in id_to_idx and other in id_to_idx:
            src_indices.append(id_to_idx[bait])
            dst_indices.append(id_to_idx[other])
            edge_weights.append(score)
            src_indices.append(id_to_idx[other])
            dst_indices.append(id_to_idx[bait])
            edge_weights.append(score)

    edge_weights = np.array(edge_weights, dtype=np.float32)
    n_interaction_edges = len(edge_weights)

    # Transform edge weights
    if edge_transform == "log":
        edge_weights = np.log2(edge_weights + 1)

    if len(edge_weights) > 0 and edge_transform != "sigmoid":
        ew_mean = edge_weights.mean()
        ew_std = edge_weights.std()
        if ew_std > 0:
            edge_weights = (edge_weights - ew_mean) / ew_std

    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    # --- 5. Graph-derived features (3 features) ---
    if use_graph_features and len(src_indices) > 0:
        # Degree (from real edges, not self-loops)
        degrees = np.zeros(n_nodes, dtype=np.float32)
        for s in src_indices:
            degrees[s] += 1
        log_degrees = np.log2(degrees + 1).reshape(-1, 1)
        feature_list.append(log_degrees)
        feature_names.append("log2_degree")

        # Mean neighbor ChIP signals
        h3k27ac_vals = nodes_df[h3k27ac_col].values.astype(np.float32)
        h3k27me3_vals = nodes_df[h3k27me3_col].values.astype(np.float32)

        neighbor_ac = np.zeros(n_nodes, dtype=np.float32)
        neighbor_me3 = np.zeros(n_nodes, dtype=np.float32)
        neighbor_count = np.zeros(n_nodes, dtype=np.float32)

        for s, d in zip(src_indices, dst_indices):
            neighbor_ac[s] += h3k27ac_vals[d]
            neighbor_me3[s] += h3k27me3_vals[d]
            neighbor_count[s] += 1

        # Avoid division by zero
        neighbor_count[neighbor_count == 0] = 1
        mean_neighbor_ac = (neighbor_ac / neighbor_count).reshape(-1, 1)
        mean_neighbor_me3 = (neighbor_me3 / neighbor_count).reshape(-1, 1)

        feature_list.append(mean_neighbor_ac)
        feature_list.append(mean_neighbor_me3)
        feature_names.extend(["mean_neighbor_H3K27ac", "mean_neighbor_H3K27me3"])
        print(f"  [features] 3 graph-topology features (degree, neighbor signals)")

    # =========================================================================
    # COMBINE AND STANDARDIZE ALL FEATURES
    # =========================================================================
    features = np.hstack(feature_list).astype(np.float32)

    # Standardize each feature to mean=0, std=1
    feat_mean = features.mean(axis=0, keepdims=True)
    feat_std = features.std(axis=0, keepdims=True)
    feat_std[feat_std == 0] = 1.0
    features = (features - feat_mean) / feat_std

    x = torch.tensor(features, dtype=torch.float)
    print(f"  [features] Total: {features.shape[1]} features (all standardized)")

    # =========================================================================
    # SELF-LOOPS
    # =========================================================================
    if add_self_loops:
        self_loops = torch.arange(n_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
        self_weights = torch.zeros(n_nodes, 1, dtype=torch.float)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        edge_attr = torch.cat([edge_attr, self_weights], dim=0)

    # =========================================================================
    # LABELS
    # =========================================================================
    labels = np.full(n_nodes, -1, dtype=np.int64)
    for i, row in nodes_df.iterrows():
        e85_state = row["category_E8.5_WT"]
        fnp_state = row["category_FNP_E10.5_WT"]
        if e85_state in NON_ACTIVE_STATES:
            if fnp_state == ACTIVE_STATE:
                labels[i] = 1
            else:
                labels[i] = 0
    y = torch.tensor(labels, dtype=torch.long)

    # =========================================================================
    # OPTIONAL: CONNECTED ONLY
    # =========================================================================
    if connected_only:
        connected = set()
        ei = edge_index.numpy()
        for j in range(ei.shape[1]):
            if ei[0, j] != ei[1, j]:
                connected.add(ei[0, j])
                connected.add(ei[1, j])
        keep = sorted(connected)
        old_to_new = {old: new for new, old in enumerate(keep)}

        x = x[keep]
        y = y[keep]
        node_ids = [node_ids[i] for i in keep]
        node_types_list = [nodes_df["node_type"].tolist()[i] for i in keep]

        new_src, new_dst, new_w = [], [], []
        for j in range(edge_index.shape[1]):
            s, d = edge_index[0, j].item(), edge_index[1, j].item()
            if s in old_to_new and d in old_to_new:
                new_src.append(old_to_new[s])
                new_dst.append(old_to_new[d])
                new_w.append(edge_attr[j, 0].item())
        edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)
        edge_attr = torch.tensor(new_w, dtype=torch.float).unsqueeze(1)
        print(f"  Filtered to connected subgraph: {len(keep)} nodes")
    else:
        node_types_list = nodes_df["node_type"].tolist()

    # =========================================================================
    # BUILD DATA OBJECT
    # =========================================================================
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.node_ids = node_ids
    data.node_types = node_types_list
    data.feature_names = feature_names

    mask = y >= 0
    print(f"\nGraph built (enriched features):")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges} (interaction: {n_interaction_edges})")
    print(f"  Node features: {data.num_node_features} → {feature_names}")
    print(f"  Labeled: {mask.sum().item()} (pos={(y==1).sum().item()}, neg={(y==0).sum().item()})")
    print(f"  Masked: {(y==-1).sum().item()}")
    return data


def create_masks(data: Data, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create train/val/test masks for labeled nodes only."""
    labeled_mask = data.y >= 0
    labeled_indices = labeled_mask.nonzero(as_tuple=True)[0].numpy()
    labels = data.y[labeled_indices].numpy()

    train_idx, temp_idx = train_test_split(
        labeled_indices, test_size=(1 - train_ratio),
        stratify=labels, random_state=seed
    )
    temp_labels = data.y[temp_idx].numpy()
    val_size = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_size),
        stratify=temp_labels, random_state=seed
    )

    n = data.num_nodes
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(f"  Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, "
          f"Test: {test_mask.sum().item()}")
    return data
