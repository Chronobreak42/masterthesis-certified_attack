

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class PRBCDSelectorGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_pairs: int,
                 tau: float = 1.0):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pair_proj = nn.Linear(2 * hidden_dim, 1)
        self.num_pairs = num_pairs
        self.tau = tau

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor):
        # 1) embed nodes
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = self.conv2(h, edge_index, edge_weight)
        # 2) compute logits for each existing edge
        row, col = edge_index  # each of shape (n_edges,)
        pair_emb = torch.cat([h[row], h[col]], dim=1)  # (n_edges, 2*hidden_dim)
        logits = self.pair_proj(pair_emb).squeeze()    # (n_edges,)

        # 3) sample num_pairs distinct edges via Gumbel-Softmax
        logits_copy = logits.clone()
        selected_idx = []
        gumbel_sample = F.gumbel_softmax(logits_copy, tau=self.tau, hard=True)
        '''
        for _ in range(self.num_pairs):
            gumbel_sample = F.gumbel_softmax(logits_copy, tau=self.tau, hard=True)
            idx = int(gumbel_sample.argmax())
            selected_idx.append(idx)
            logits_copy[idx] = float('-inf')  # mask so we don't pick the same edge again
        '''

        # 4) return just the selected edges (2 x num_pairs)asdasdhzaubauwbfuabwubfjajsbfasdbbavb
        #new_edge_index = edge_index[:, selected_idx]
        return gumbel_sample


'''
# AttackerGNN/PreEdgeSelector.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class PreEdgeSelector(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_pairs: int, tau: float = 1.0):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pair_proj = nn.Linear(2 * hidden_dim, 1)
        self.num_pairs = num_pairs
        self.tau = tau

    # ----- new: reusable pieces -----
    def encode(self, x, edge_index, edge_weight):
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = self.conv2(h, edge_index, edge_weight)
        return h  # (N, hidden)

    def score_pairs(self, h, edge_index_like):
        row, col = edge_index_like
        pair_emb = torch.cat([h[row], h[col]], dim=1)  # (E, 2*hidden)
        return self.pair_proj(pair_emb).squeeze(-1)    # (E,)

    # logits for the *given* edge_index (used in pretraining)
    def forward_logits(self, x, edge_index, edge_weight):
        h = self.encode(x, edge_index, edge_weight)
        return self.score_pairs(h, edge_index)

    # keep your sampling API for inference/attack
    def sample_k_gumbel(self, x, edge_index, edge_weight):
        h = self.encode(x, edge_index, edge_weight)
        logits = self.score_pairs(h, edge_index)       # (E,)

        logits_copy = logits.clone()
        selected_idx = []
        for _ in range(self.num_pairs):
            y = F.gumbel_softmax(logits_copy, tau=self.tau, hard=True, dim=0)  # (E,)
            idx = int(y.argmax())
            selected_idx.append(idx)
            logits_copy[idx] = float('-inf')  # mask

        return edge_index[:, selected_idx]    # (2, num_pairs)

'''
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class PreEdgeSelector(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_pairs: int,
                 tau: float = 1.0):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pair_proj = nn.Linear(2 * hidden_dim, 1)
        self.num_pairs = num_pairs
        self.tau = tau

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor):
        # 1) embed nodes
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = self.conv2(h, edge_index, edge_weight)
        # 2) compute logits for each existing edge
        row, col = edge_index  # each of shape (n_edges,)
        pair_emb = torch.cat([h[row], h[col]], dim=1)  # (n_edges, 2*hidden_dim)
        logits = self.pair_proj(pair_emb).squeeze()    # (n_edges,)

        # 3) sample num_pairs distinct edges via Gumbel-Softmax
        logits_copy = logits.clone()
        selected_idx = []
        for _ in range(self.num_pairs):
            gumbel_sample = F.gumbel_softmax(logits_copy, tau=self.tau, hard=True)
            idx = int(gumbel_sample.argmax())
            selected_idx.append(idx)
            logits_copy[idx] = float('-inf')  # mask so we don't pick the same edge again

        # 4) return just the selected edges (2 x num_pairs)asdasdhzaubauwbfuabwubfjajsbfasdbbavb
        new_edge_index = edge_index[:, selected_idx]
        return new_edge_index
'''