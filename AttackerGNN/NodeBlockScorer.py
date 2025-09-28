import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class NodeBlockScorer(nn.Module):
    """
    Predicts per-node decision margins. Smaller = closer to boundary = more attackable.
    """
    def __init__(self, in_channels: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.node_head = nn.Linear(hidden_dim, 1)  # predicts (approx) margin

    def embed(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight)
        h = self.norm1(F.relu(h))
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_weight)
        h = self.norm2(F.relu(h))
        return h

    def forward(self, x, edge_index, edge_weight=None):
        """
        Returns predicted margins per node: shape (N,).
        """
        h = self.embed(x, edge_index, edge_weight)
        margin_pred = self.node_head(h).squeeze(-1)
        # margins are non-negative; clamp just in case
        return margin_pred.clamp_min(0.0)

    def topk_closest_to_margin(self, x, edge_index, edge_weight=None, k: int = 64):
        """
        Returns indices of k nodes with *smallest* predicted margins.
        """
        margins = self.forward(x, edge_index, edge_weight)  # (N,)
        k = min(k, margins.numel())
        # smallest margins ⇒ most vulnerable
        vals, idx = torch.topk(-margins, k, largest=True, sorted=False)
        return idx, margins[idx]