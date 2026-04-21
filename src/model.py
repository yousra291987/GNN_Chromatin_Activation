"""
model.py — GNN architectures with continuous edge weight support.

Implements:
  - WeightedGCN: GCN where messages are scaled by edge weights
  - ChromatinTransformer: TransformerConv that natively uses edge features
  - ChromatinGAT: Standard GAT (baseline, ignores edge weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, TransformerConv, BatchNorm,
    MessagePassing
)
from torch_geometric.utils import add_self_loops, softmax
from torch.nn import Parameter


# =============================================================================
# 1. Weighted GCN — scales neighbor messages by CHiCAGO edge weight
# =============================================================================

class WeightedGCNConv(MessagePassing):
    """
    GCN convolution where each message is multiplied by the edge weight.

    Standard GCN: h_i = sum_j (1/sqrt(d_i * d_j)) * h_j * W
    Weighted GCN: h_i = sum_j (w_ij / sqrt(d_i * d_j)) * h_j * W

    This way, strong CHiCAGO interactions contribute more to the
    aggregated neighborhood representation.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr=None):
        # Transform features
        x = self.lin(x)

        # Compute normalization (degree-based)
        row, col = edge_index
        deg = torch.zeros(x.size(0), device=x.device)
        if edge_attr is not None:
            # Weight-aware degree
            deg.scatter_add_(0, row, edge_attr.squeeze().abs())
        else:
            deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Propagate with edge weights
        out = self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_attr)
        return out + self.bias

    def message(self, x_j, norm, edge_attr):
        # Scale message by normalization AND edge weight
        msg = norm.unsqueeze(-1) * x_j
        if edge_attr is not None:
            msg = msg * edge_attr
        return msg


class WeightedGCN(nn.Module):
    """
    Multi-layer Weighted GCN for node classification.
    Each message-passing step scales neighbor contributions by CHiCAGO score.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(WeightedGCNConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(WeightedGCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        self.convs.append(WeightedGCNConv(hidden_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def get_embeddings(self, x, edge_index, edge_attr=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
        return x


# =============================================================================
# 2. Transformer Conv — attention + edge features natively
# =============================================================================

class ChromatinTransformer(nn.Module):
    """
    Graph Transformer using TransformerConv.

    TransformerConv computes attention as:
        alpha_ij = softmax( (W_q h_i)^T (W_k h_j + W_e e_ij) / sqrt(d) )
        h_i' = sum_j alpha_ij * (W_v h_j + W_e e_ij)

    The edge feature e_ij (CHiCAGO score) directly modulates both
    the attention weights AND the value messages. This means the model
    learns to weight interactions by their strength.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_classes: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        edge_dim: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(
            TransformerConv(in_channels, hidden_channels, heads=heads,
                            edge_dim=edge_dim, dropout=dropout)
        )
        self.bns.append(BatchNorm(hidden_channels * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                TransformerConv(hidden_channels * heads, hidden_channels,
                                heads=heads, edge_dim=edge_dim, dropout=dropout)
            )
            self.bns.append(BatchNorm(hidden_channels * heads))

        # Final layer (single head)
        self.convs.append(
            TransformerConv(hidden_channels * heads, hidden_channels,
                            heads=1, concat=False, edge_dim=edge_dim,
                            dropout=dropout)
        )
        self.bns.append(BatchNorm(hidden_channels))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.elu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def get_embeddings(self, x, edge_index, edge_attr=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.elu(x)
        return x


# =============================================================================
# 3. Standard GAT — baseline (ignores edge weights)
# =============================================================================

class ChromatinGAT(nn.Module):
    """Standard GAT that ignores edge features. Kept as baseline."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_classes: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels * heads))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
            self.bns.append(BatchNorm(hidden_channels * heads))

        self.convs.append(
            GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        )
        self.bns.append(BatchNorm(hidden_channels))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)  # GAT ignores edge_attr
            x = bn(x)
            x = F.elu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def get_embeddings(self, x, edge_index, edge_attr=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
        return x


# =============================================================================
# Factory
# =============================================================================

def get_model(name: str, **kwargs) -> nn.Module:
    """Factory function to get model by name."""
    models = {
        "weighted_gcn": WeightedGCN,
        "transformer": ChromatinTransformer,
        "gat": ChromatinGAT,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from {list(models.keys())}")
    return models[name](**kwargs)
