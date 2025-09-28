import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class PreEdgeSelector(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int,
                 num_pairs: int):
        """
        in_channels  – dim. of input features X
        hidden_dim   – dim. of the intermediate GCN
        num_pairs    – how many candidate edges to select
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.num_pairs = num_pairs

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None
               ) -> torch.Tensor:
        # 1) single GCN layer: H = σ(Â X W)
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)                                   # (N × hidden_dim)

        # 2) full pairwise scores S = H Hᵀ
        #    (be careful: this is O(N²) in memory!)
        S = h @ h.t()                                   # (N × N)
        N = S.size(0)

        # 3) mask out the diagonal (no self-loops)
        S.fill_diagonal_(-1e9)

        # 4) flatten, pick top-k entries
        vals, idxs = torch.topk(S.view(-1), self.num_pairs, sorted=False)
        row = idxs // N
        col = idxs %  N

        # return as edge_index: shape (2, num_pairs)
        return torch.stack([row, col], dim=0)