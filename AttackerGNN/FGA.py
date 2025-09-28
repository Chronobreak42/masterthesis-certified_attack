"""
Fast Gradient Attack (FGA) implementation for graph data (node classification / embeddings).

This file provides:
  • Simple 2-layer GCN that matches Eq. (2) in the paper via explicit normalized adjacency.
  • A dense-adjacency FGA attacker that, for a given target node, iteratively
    computes dL_t/dA (gradient wrt adjacency), symmetrizes it, and at each step
    adds/removes the edge with maximum valid |gradient| (Algorithm 1; Eq. (5–8)).
  • Variants: 'unlimited', 'direct', and 'indirect' attacks, early stopping, and
    optional misclassification check.

Notes
-----
• This implementation works with dense adjacency. For Cora/Citeseer sizes this is fine.
  For larger graphs, replace dense ops by sparse equivalents. The gradient pipeline
  relies on autograd through normalization A_bar = D^(-1/2) (A + I) D^(-1/2).
• The attacker modifies an *undirected*, unweighted adjacency (0/1 off-diagonal).
• The gradients are taken wrt the *unweighted* A prior to normalization; the
  normalization remains differentiable (through degree). If you want to attack
  other embedding models (DeepWalk/node2vec/LINE), you can still use this attacker
  to produce an adversarial graph and then re-embed with the target method.

Author: (you) — 2025
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Utilities
# ------------------------------

def to_dense_adjacency(edge_index: torch.Tensor, num_nodes: int, device: Optional[str] = None) -> torch.Tensor:
    """Build a dense, symmetric 0/1 adjacency (no self-loops) from edge_index.

    Parameters
    ----------
    edge_index : (2, E) LongTensor
        COO indices, may be directed or undirected.
    num_nodes : int
        Number of nodes.
    device : Optional[str]
        Target device.

    Returns
    -------
    A : (N, N) float32 Tensor
        Symmetric 0/1 adjacency, zeros on the diagonal.
    """
    if device is None:
        device = edge_index.device
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    u, v = edge_index[0].long(), edge_index[1].long()
    A[u, v] = 1.0
    A[v, u] = 1.0
    A.fill_diagonal_(0.0)
    return A


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """Compute A_bar = D^(-1/2) (A + I) D^(-1/2) with autograd support.

    A is assumed symmetric with zero diag (undirected simple graph). We add I inside.
    """
    N = A.size(0)
    A_tilde = A + torch.eye(N, dtype=A.dtype, device=A.device)
    deg = A_tilde.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt


# ------------------------------
# Model
# ------------------------------

class SimpleGCN(nn.Module):
    """Two-layer GCN that exactly follows Y' = softmax(Â ReLU(Â X W0) W1).

    This avoids GCNConv normalization differences by using explicit Â.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.W0 = nn.Parameter(torch.empty(in_dim, hidden_dim))
        self.W1 = nn.Parameter(torch.empty(hidden_dim, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W0)
        nn.init.xavier_uniform_(self.W1)

    def forward(self, X: torch.Tensor, A_bar: torch.Tensor) -> torch.Tensor:
        H = A_bar @ (X @ self.W0)
        H = F.relu(H)
        Z = A_bar @ (H @ self.W1)
        return Z  # logits (no softmax)


# ------------------------------
# FGA Attacker
# ------------------------------

AttackMode = Literal['unlimited', 'direct', 'indirect']

@dataclass
class FGAResult:
    A_adv: torch.Tensor  # (N,N) float, final 0/1 adjacency
    steps: List[Tuple[int, int, int, float]]  # (u, v, action(+1 add / -1 del), abs_grad)
    num_steps: int
    stopped_early: bool
    misclassified_after: Optional[int]


class FGA:
    """Fast Gradient Attack (FGA) on graphs (white-box against the GCN surrogate).

    At each iteration h:
      1) Compute gradient g = dL_t/dA on current adversarial A^(h-1) for target node t.
      2) Symmetrize g -> g_hat = (g + g^T)/2, zero diag.
      3) Among VALID pairs (add if g>0 & A=0; delete if g<0 & A=1), pick (i,j) with max |g_hat_ij|.
      4) Update A by A_ij <- 1 if adding else 0 (and symmetrize).

    Variants: 'unlimited' (any pair), 'direct' (edges incident to target), 'indirect' (neither endpoint is target).
    """

    def __init__(
        self,
        model: SimpleGCN,
        X: torch.Tensor,         # (N,F)
        A: torch.Tensor,         # (N,N) 0/1, symmetric, zero diag
        y: torch.Tensor,         # (N,) Long labels
        device: Optional[str] = None,
    ):
        if device is None:
            device = X.device.type if isinstance(X.device, torch.device) else 'cpu'
        self.device = device
        self.model = model.to(device)
        self.X = X.to(device)
        self.y = y.to(device)
        self.N = A.size(0)
        assert A.shape == (self.N, self.N)
        self.A0 = A.to(device).detach().clone()

        self.model.eval()

    @torch.no_grad()
    def predict(self, A: torch.Tensor) -> torch.Tensor:
        A_bar = normalize_adjacency(A)
        logits = self.model(self.X, A_bar)
        return logits.argmax(dim=1)

    def _target_loss(self, logits: torch.Tensor, target: int) -> torch.Tensor:
        """Cross-entropy loss on the true label of target node (maximize this)."""
        return F.cross_entropy(logits[target:target+1], self.y[target:target+1])

    def _mask_candidates(self, A: torch.Tensor, ghat: torch.Tensor, target: int, mode: AttackMode) -> torch.Tensor:
        """Return a score matrix S with |ghat| on valid positions, else 0; upper triangle only."""
        # Validity by sign and current adjacency state
        can_add = (A == 0) & (ghat > 0)
        can_del = (A == 1) & (ghat < 0)
        valid = can_add | can_del

        if mode == 'direct':
            tgt_mask = torch.zeros_like(valid, dtype=torch.bool)
            tgt_mask[target, :] = True
            tgt_mask[:, target] = True
            valid &= tgt_mask
        elif mode == 'indirect':
            # neither endpoint is target
            not_tgt = torch.ones(self.N, dtype=torch.bool, device=A.device)
            not_tgt[target] = False
            valid &= (not_tgt[:, None] & not_tgt[None, :])
        # else unlimited: no extra restriction

        # upper triangle only (i<j) and zero diag
        tri = torch.triu(torch.ones_like(valid, dtype=torch.bool), diagonal=1)
        valid &= tri

        scores = torch.zeros_like(ghat)
        scores[valid] = ghat[valid].abs()
        return scores

    def _one_step(self, A: torch.Tensor, target: int, mode: AttackMode) -> Tuple[torch.Tensor, Tuple[int,int,int,float]]:
        """Compute gradient and apply the single best add/delete.

        Returns updated A and a tuple (u,v,action,abs_grad).
        """
        A_var = A.clone().detach().requires_grad_(True)
        A_bar = normalize_adjacency(A_var)
        logits = self.model(self.X, A_bar)
        Lt = self._target_loss(logits, target)
        Lt.backward()
        g = A_var.grad  # (N,N)
        # symmetrize & zero diag
        ghat = 0.5 * (g + g.T)
        ghat.fill_diagonal_(0.0)

        S = self._mask_candidates(A, ghat, target, mode)
        if S.max() <= 0:
            # Fallback: ignore sign constraints, pick largest |grad| with change impact
            S = torch.triu(ghat.abs(), diagonal=1)

        # argmax in upper triangle
        flat_idx = S.view(-1).argmax()
        u = int(flat_idx // self.N)
        v = int(flat_idx % self.N)
        if u == v:
            # extremely unlikely due to triu, guard anyway
            return A, (u, v, 0, 0.0)

        action = 1 if ghat[u, v] > 0 and A[u, v].item() == 0 else -1

        A_new = A.clone()
        if action == 1:
            A_new[u, v] = 1.0
            A_new[v, u] = 1.0
        else:
            A_new[u, v] = 0.0
            A_new[v, u] = 0.0

        return A_new, (u, v, action, float(ghat[u, v].abs().item()))

    def attack(
        self,
        target: int,
        K: int,
        mode: AttackMode = 'unlimited',
        stop_on_miscls: bool = True,
        return_intermediate: bool = False,
    ) -> FGAResult:
        """Run K FGA steps against the target node.

        Parameters
        ----------
        target : int
            Target node index whose loss we maximize.
        K : int
            Number of edge modifications (budget).
        mode : {'unlimited','direct','indirect'}
        stop_on_miscls : bool
            If True, stop when the target gets misclassified.
        return_intermediate : bool
            If True, keeps intermediate As (disabled by default to save memory).

        Returns
        -------
        FGAResult
        """
        A = self.A0.clone()
        steps: List[Tuple[int,int,int,float]] = []
        mis_after: Optional[int] = None

        # Initial prediction
        with torch.no_grad():
            pred0 = self.predict(A)[target].item()

        for h in range(1, K + 1):
            A, info = self._one_step(A, target, mode)
            steps.append(info)

            if stop_on_miscls:
                with torch.no_grad():
                    pred = self.predict(A)[target].item()
                if pred != pred0:
                    mis_after = h
                    break

        return FGAResult(A_adv=A, steps=steps, num_steps=len(steps), stopped_early=(mis_after is not None), misclassified_after=mis_after)


# ------------------------------
# Example usage (Cora-like tensors)
# ------------------------------
if __name__ == "__main__":
    # Minimal synthetic example (replace with real data for Cora/Citeseer)
    torch.manual_seed(0)

    N, F_in, H, C = 100, 16, 32, 7
    X = torch.randn(N, F_in)

    # Build a random sparse graph (undirected)
    prob = 0.02
    A = (torch.rand(N, N) < prob).float()
    A.fill_diagonal_(0.0)
    A = torch.triu(A, diagonal=1)
    A = A + A.T

    # Fake labels
    y = torch.randint(0, C, (N,))

    # Train a tiny GCN briefly on random train split
    model = SimpleGCN(F_in, H, C)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    train_idx = torch.arange(int(0.6 * N))
    A_bar = normalize_adjacency(A)
    for epoch in range(50):
        model.train()
        logits = model(X, A_bar)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    # Pick a target node that is currently correctly classified
    with torch.no_grad():
        pred = model(X, A_bar).argmax(dim=1)
    correct_nodes = (pred == y).nonzero(as_tuple=False).flatten()
    target = int(correct_nodes[0].item()) if correct_nodes.numel() > 0 else 0

    attacker = FGA(model, X, A, y)
    result = attacker.attack(target=target, K=10, mode='unlimited', stop_on_miscls=True)

    print(f"Target node: {target}; early stop: {result.stopped_early}; steps taken: {result.num_steps}; misclassified_after={result.misclassified_after}")
    for (u, v, action, gabs) in result.steps:
        print(f"  step: ({u},{v}) action={'add' if action==1 else 'del'} |grad|={gabs:.4e}")
