import torch
import torch.nn as nn
import torch.nn.functional as F

# ------- utils -------
def coo_norm_with_self_loops(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if A.layout != torch.sparse_coo:
        raise ValueError("A must be sparse COO")
    A = A.coalesce()
    N, dev, dt = A.size(0), A.device, A.dtype
    I = torch.arange(N, device=dev, dtype=torch.long).repeat(2, 1)
    V = torch.ones(N, device=dev, dtype=dt)
    idx = torch.cat([A.indices(), I], dim=1)
    val = torch.cat([A.values(), V], dim=0)
    A = torch.sparse_coo_tensor(idx, val, size=(N, N), device=dev, dtype=dt).coalesce()
    row, col = A.indices()
    deg = torch.zeros(N, device=dev, dtype=dt).index_add_(0, row, A.values())
    dinv = deg.clamp_min(eps).pow(-0.5)
    val = A.values() * dinv[row] * dinv[col]
    return torch.sparse_coo_tensor(A.indices(), val, size=A.size(), device=dev, dtype=dt).coalesce()

# ------- model -------
class AllPairsLinkPredictor(nn.Module):
    """
    Trains on *all* unordered pairs (u<v) at once (no negative sampling, no batching).
    Uses a 2-layer GCN encoder and a bilinear decoder S = (H W) H^T.
    """
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden, bias=True)
        self.lin2 = nn.Linear(hidden, out_dim, bias=True)
        self.dropout = dropout
        self.W = nn.Parameter(torch.empty(out_dim, out_dim))
        nn.init.xavier_uniform_(self.W)

    def encode(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        Ahat = coo_norm_with_self_loops(A)
        H = torch.sparse.mm(Ahat, X)
        H = F.relu(self.lin1(H))
        H = F.dropout(H, p=self.dropout, training=self.training)
        H = torch.sparse.mm(Ahat, H)
        H = self.lin2(H)  # (N, d)
        return H

    def score_matrix(self, H: torch.Tensor) -> torch.Tensor:
        return (H @ self.W) @ H.T  # logits (N,N)

    def loss_all_pairs(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        BCE over ALL unordered pairs (u<v). Labels are taken from A (treated as undirected, binary).
        """
        H = self.encode(X, A)
        S = self.score_matrix(H)                     # (N,N) logits
        N = S.size(0)
        tri = torch.triu(torch.ones(N, N, dtype=torch.bool, device=S.device), diagonal=1)

        # build symmetric dense labels from sparse A
        A = A.coalesce()
        Y = torch.zeros(N, N, device=S.device, dtype=S.dtype)
        u, v = A.indices()
        Y[u, v] = 1.0
        Y[v, u] = 1.0
        y = Y[tri]                                   # (E_all,)
        logits = S[tri]                              # (E_all,)

        return F.binary_cross_entropy_with_logits(logits, y)

    @torch.no_grad()
    def predict_all(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Returns full probability matrix for ALL pairs (N×N). Mask diagonal/tri as you wish.
        """
        H = self.encode(X, A)
        S = self.score_matrix(H)
        return torch.sigmoid(S)

    @torch.no_grad()
    def bottomk_pairs(self, X: torch.Tensor, A: torch.Tensor, k: int):
        """
        Rank ALL unordered pairs (u<v) by uncertainty |p - 0.5| and return
        the k pairs with the smallest uncertainty (most ambiguous links).
        """
        P = self.predict_all(X, A)
        N = P.size(0)
        tri = torch.triu(torch.ones(N, N, dtype=torch.bool, device=P.device), diagonal=1)
        vals = P[tri]
        k = int(min(k, vals.numel()))
        # smaller |p-0.5| = more uncertain
        uncertainty = torch.abs(vals - 0.5)
        bottomv, sel = torch.topk(uncertainty, k=k, largest=False)
        ui, vj = tri.nonzero(as_tuple=True)
        return torch.stack([ui[sel].cpu(), vj[sel].cpu()], 0), vals[sel].cpu()