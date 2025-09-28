"""
Pretraining a *learned* global edge scorer by distilling PGD topology gradients
===============================================================================

Goal
----
Train a fast edge-scoring module that predicts the global importance of an edge
(add/remove) for an *untargeted* attack, so you can sample strong PR-BCD blocks
without recomputing dense gradients every time.

Approach
--------
1) Use your existing node encoder (NodeBlockScorer.embed) to get node embeddings h.
2) Build PGD-style labels by backpropagating a global attack loss wrt dense A:
   G = dL_global/dA (symmetrized, zero diag). Use |G_ij| (or signed variants)
   as pseudo-labels for (i,j) pairs.
3) Train an *edge scorer* f(h_i, h_j) → s_ij to match these labels using
   regression (MSE on normalized |G|) or pairwise ranking (hinge/softmax).
4) At sampling time, score many pairs quickly with the learned f and pick Top-K
   to form the PR-BCD block.

This file provides:
- EdgeBlockScorer: small MLP/bilinear scorer on node embeddings.
- compute_pgd_edge_labels: builds (u,v) and labels y_ij from a surrogate model.
- pretrain_pgd_edge_scorer: a training loop (regression + optional pairwise rank).
- sample_block_from_pgdlearned: use the trained edge scorer to create a block.

Drop-in usage
-------------
Within your PRBCD object, set:
  self.selector  -> node encoder (already present in your code)
  self.edge_scorer = EdgeBlockScorer(d=self.selector.hidden_dim)
Then call pretrain with your trained surrogate and dataset. Afterwards, use
sample_block_from_pgdlearned(...) to initialize a block.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Small edge scorer
# ------------------------------
class EdgeBlockScorer(nn.Module):
    """Score an undirected pair (i,j) from node embeddings h.

    score(i,j) = MLP([h_i, h_j, |h_i-h_j|, h_i*h_j]) or a bilinear form.
    """
    def __init__(self, d: int, hidden: int = 128, use_bilinear: bool = True):
        super().__init__()
        self.use_bilinear = use_bilinear
        if use_bilinear:
            self.B = nn.Parameter(torch.empty(d, d))
            nn.init.xavier_uniform_(self.B)
        self.mlp = nn.Sequential(
            nn.Linear(4 * d + (1 if use_bilinear else 0), hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    @torch.no_grad()
    def pair_features(self, h: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        hu, hv = h[u], h[v]
        feats = [hu, hv, (hu - hv).abs(), hu * hv]
        if self.use_bilinear:
            bil = (hu @ self.B @ hv.T).diagonal().unsqueeze(-1)
            feats.append(bil)
        return torch.cat(feats, dim=-1)

    def forward(self, h: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        # pairs: (M,2) with pairs[:,0]=u, pairs[:,1]=v
        u, v = pairs[:, 0].long(), pairs[:, 1].long()
        hu, hv = h[u], h[v]
        feats = [hu, hv, (hu - hv).abs(), hu * hv]
        if self.use_bilinear:
            bil = (hu @ self.B @ hv.T).diagonal().unsqueeze(-1)
            feats.append(bil)
        x = torch.cat(feats, dim=-1)
        return self.mlp(x).squeeze(-1)


# ------------------------------
# Dense adjacency helpers
# ------------------------------

def dense_from_edge_index(edge_index: torch.Tensor, n: int, device: torch.device | str) -> torch.Tensor:
    A = torch.zeros((n, n), dtype=torch.float32, device=device)
    A[edge_index[0].long(), edge_index[1].long()] = 1.0
    A[edge_index[1].long(), edge_index[0].long()] = 1.0
    A.fill_diagonal_(0.0)
    return A


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    N = A.size(0)
    A_tilde = A + torch.eye(N, dtype=A.dtype, device=A.device)
    deg = A_tilde.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt


# ------------------------------
# Build PGD-style labels (one dense backward)
# ------------------------------
@torch.no_grad()
def _upper_tri(N: int, device: torch.device | str):
    return torch.triu_indices(N, N, offset=1, device=device)


def compute_pgd_edge_labels(
    surrogate,                 # callable(X, A_bar)->logits
    X: torch.Tensor,           # (N,F)
    edge_index: torch.Tensor,  # (2,E)
    y: torch.Tensor,           # (N,)
    idx_loss: Optional[torch.Tensor] = None,
    loss_type: Literal['ce','cw'] = 'ce',
    kappa: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (pairs, labels, sign_pref) for upper-tri pairs.

    labels = |dL/dA_ij| (z-scored), where L is a global attack loss.
    sign_pref in {-1,0,+1} indicates delete/add preference from the raw gradient.
    """
    device = X.device
    N = X.size(0)
    A0 = dense_from_edge_index(edge_index, N, device)

    A = A0.clone().detach().requires_grad_(True)
    A_bar = normalize_adjacency(A)
    logits = surrogate(X, A_bar)

    if idx_loss is None:
        idx_loss = torch.arange(N, device=device)
    if loss_type == 'ce':
        loss = -F.cross_entropy(logits[idx_loss], y[idx_loss], reduction='mean')
    else:
        Z = logits[idx_loss]
        yy = y[idx_loss]
        true = Z[torch.arange(Z.size(0), device=device), yy]
        Zm = Z.clone(); Zm[torch.arange(Z.size(0), device=device), yy] = -1e9
        other = Zm.max(dim=1).values
        margin = true - other
        loss = torch.maximum(margin, torch.tensor(-float(kappa), device=device)).mean()

    loss.backward()
    G = A.grad
    G = 0.5*(G + G.t()); G.fill_diagonal_(0.0)

    u, v = _upper_tri(N, device)
    g = G[u, v]
    labels = g.abs()
    # z-score to stabilize regression
    labels = (labels - labels.mean()) / (labels.std() + 1e-6)

    sign_pref = torch.sign(g)  # + add, - delete, 0 neutral
    pairs = torch.stack([u, v], dim=1)
    return pairs, labels, sign_pref


# ------------------------------
# Pretraining loop (regression + optional pairwise ranking)
# ------------------------------
@dataclass
class EdgePretrainCfg:
    epochs: int = 10
    batch_size: int = 262144
    lr: float = 1e-3
    wd: float = 5e-4
    rank_weight: float = 0.0   # >0 to add pairwise ranking loss
    top_frac: float = 0.1      # for ranking: sample positives from top-10%


def pretrain_pgd_edge_scorer(
    node_encoder,              # module with .embed(X, edge_index, edge_weight)
    edge_scorer: EdgeBlockScorer,
    surrogate,                 # callable(X, A_bar)->logits
    X: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    idx_loss: Optional[torch.Tensor] = None,
    cfg: EdgePretrainCfg = EdgePretrainCfg(),
):
    device = X.device
    node_encoder.train(); edge_scorer.train()

    # 1) Labels from a single dense backward (can repeat over epochs if desired)
    with torch.no_grad():
        pairs, labels, _ = compute_pgd_edge_labels(surrogate, X, edge_index, y, idx_loss, 'ce')

    # 2) Node embeddings (sparse forward allowed)
    with torch.no_grad():
        h = node_encoder.embed(X, edge_index, None)  # (N,d)

    N_pairs = pairs.size(0)
    idx_all = torch.arange(N_pairs, device=device)
    opt = torch.optim.Adam(list(edge_scorer.parameters()), lr=cfg.lr, weight_decay=cfg.wd)
    mse = nn.MSELoss()

    def ranking_loss(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y are z-scored; pick top p% as pos, bottom p% as neg and do hinge
        p = max(1, int(cfg.top_frac * y.numel()))
        order = torch.argsort(y, descending=True)
        pos_idx = order[:p]
        neg_idx = order[-p:]
        pos = scores[pos_idx]
        neg = scores[neg_idx]
        # max(0, 1 - pos + neg)
        diff = 1.0 - pos.view(-1,1) + neg.view(1,-1)
        return F.relu(diff).mean()

    for e in range(cfg.epochs):
        perm = idx_all[torch.randperm(N_pairs, device=device)]
        for start in range(0, N_pairs, cfg.batch_size):
            end = min(N_pairs, start + cfg.batch_size)
            sel = perm[start:end]
            batch_pairs = pairs[sel]
            yb = labels[sel]

            scores = edge_scorer(h, batch_pairs)
            loss = mse(scores, yb)
            if cfg.rank_weight > 0:
                loss = loss + cfg.rank_weight * ranking_loss(scores.detach(), yb.detach())

            opt.zero_grad(); loss.backward(); opt.step()

    node_encoder.eval(); edge_scorer.eval()


# ------------------------------
# Sampling with learned scorer
# ------------------------------
@torch.no_grad()
def sample_block_from_pgdlearned(
    node_encoder,
    edge_scorer: EdgeBlockScorer,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    K: int,                            # block size
    prefer: Literal['abs','add','del'] = 'abs',
):
    device = X.device
    N = X.size(0)
    # 1) Embeddings
    h = node_encoder.embed(X, edge_index, None)

    # 2) All upper-tri pairs (C(N,2)) — fine for Cora-ML; for bigger graphs, sample
    u, v = _upper_tri(N, device)
    pairs = torch.stack([u, v], dim=1)

    # 3) Score
    scores = edge_scorer(h, pairs)

    # 4) Preference: if 'add'/'del', we can lightly bias by current adjacency
    A0 = dense_from_edge_index(edge_index, N, device)
    A_up = A0[u, v]
    if prefer == 'add':
        scores = scores * (1.0 - A_up)  # zero out existing edges
    elif prefer == 'del':
        scores = scores * (A_up)        # zero out non-edges

    K = int(min(K, scores.numel()))
    sel = torch.topk(scores, k=K, largest=True).indices
    sel_u, sel_v = u[sel], v[sel]

    # return as (2, K) edge_index (upper-tri)
    return torch.stack([sel_u, sel_v], dim=0)
