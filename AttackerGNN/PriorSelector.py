# AttackerGNN/PreEdgeSelector.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class PriorSelector(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pair_proj = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = self.conv2(h, edge_index, edge_weight)
        row, col = edge_index
        pair_emb = torch.cat([h[row], h[col]], dim=1)  # (E, 2*hidden)
        logits = self.pair_proj(pair_emb).squeeze(-1)  # (E,)
        return logits
