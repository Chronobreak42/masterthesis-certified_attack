import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ---------- sparse helpers ----------
def _ensure_coalesced(A: torch.Tensor) -> torch.Tensor:
    if A.layout != torch.sparse_coo:
        raise ValueError("Expected sparse COO tensor.")
    return A.coalesce()

def add_self_loops_sparse(A: torch.Tensor, fill_value: float = 1.0) -> torch.Tensor:
    A = _ensure_coalesced(A)
    N = A.size(0); dev, dt = A.device, A.dtype
    I = torch.arange(N, device=dev, dtype=torch.long).repeat(2, 1)  # (2,N)
    V = torch.full((N,), fill_value, device=dev, dtype=dt)
    return torch.sparse_coo_tensor(
        torch.cat([A.indices(), I], dim=1),
        torch.cat([A.values(), V], dim=0),
        size=A.size(), device=dev, dtype=dt
    ).coalesce()

def normalize_adj_sparse(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    A_hat = add_self_loops_sparse(A)
    idx, val = A_hat.indices(), A_hat.values()
    N, dev, dt = A_hat.size(0), A_hat.device, A_hat.dtype
    deg = torch.zeros(N, device=dev, dtype=dt)
    deg.index_add_(0, idx[0], val)
    dinv = (deg.clamp(min=eps)).pow(-0.5)
    val = val * dinv[idx[0]] * dinv[idx[1]]
    return torch.sparse_coo_tensor(idx, val, size=A_hat.size(), device=dev, dtype=dt).coalesce()

def sparse_add(A: torch.Tensor, B: Optional[torch.Tensor]) -> torch.Tensor:
    return A.coalesce() if B is None else (A.coalesce() + B.coalesce()).coalesce()

def sparse_clamp_min(A: torch.Tensor, min_val: float = 0.0) -> torch.Tensor:
    A = A.coalesce()
    v = A.values().clone().clamp_(min=min_val)
    return torch.sparse_coo_tensor(A.indices(), v, size=A.size(), device=A.device, dtype=A.dtype).coalesce()

# ---------- layers ----------
class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
    def forward(self, X_sparse: torch.Tensor, A_sparse: torch.Tensor, M_sparse: Optional[torch.Tensor] = None):
        A_eff = sparse_clamp_min(sparse_add(A_sparse, M_sparse), 0.0)
        A_bar = normalize_adj_sparse(A_eff)
        X = X_sparse.to_dense()                # densify once
        H = torch.sparse.mm(A_bar, X)          # (N, in_dim)
        return self.lin(H)                     # (N, out_dim)

class AttnPooling(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1, bias=False)
    def forward(self, H: torch.Tensor):
        e = self.attn(H).squeeze(-1)         # (N,)
        a = F.softmax(e, dim=0)              # (N,)
        z = (a.unsqueeze(1) * H).sum(0)      # (d_out,)
        return z, a

class BilinearEdgeScorer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.W)
    def score_pairs(self, H: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # logits for specific pairs (u,v): (H@W)[u] · H[v]
        T = H @ self.W                       # (N, d)
        return (T[u] * H[v]).sum(-1)         # (K,)

# ---------- full model ----------
class GNNLeastLikelyEdges(nn.Module):
    """
    Sparse I/O. Forward returns the b least-likely EXISTING edges directly:
      edges_idx: (2, b)  (u,v) with u<v from A
      edge_probs: (b,)   sigmoid(logits) for those edges (smallest first)
      z: (out_dim,)      pooled graph embedding (for downstream use)
    """
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.gcn1 = DenseGCNLayer(in_dim, hidden)
        self.gcn2 = DenseGCNLayer(hidden, out_dim)
        self.pool = AttnPooling(out_dim)
        self.edge_scorer = BilinearEdgeScorer(out_dim)
        self.dropout = dropout

    # Encode once
    def encode(self, X_sparse: torch.Tensor, A_sparse: torch.Tensor, M_sparse: Optional[torch.Tensor] = None):
        H = F.relu(self.gcn1(X_sparse, A_sparse, M_sparse))   # (N, hidden)
        H = F.dropout(H, p=self.dropout, training=self.training)
        H_sparse = H.to_sparse_coo()
        H2 = self.gcn2(H_sparse, A_sparse, M_sparse)          # (N, out_dim)
        return H2

    # Utility: upper-tri existing edges (u<v)
    @staticmethod
    def _upper_triangle_edges_from_sparse(A_sparse: torch.Tensor, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        A = A_sparse.coalesce()
        u, v = A.indices()
        mask = u < v
        u, v = u[mask], v[mask]
        if device is not None:
            u, v = u.to(device), v.to(device)
        return u, v

    # ---- BCE LOSS over pos edges vs sampled non-edges ----
    def bce_loss(self,
                 X_sparse: torch.Tensor,
                 A_sparse: torch.Tensor,
                 M_sparse: Optional[torch.Tensor] = None,
                 neg_ratio: int = 1,
                 max_pos_per_batch: int = 64000) -> torch.Tensor:
        device = X_sparse.device
        N = A_sparse.size(0)

        # Encode once per call (requires grad through model params)
        H = self.encode(X_sparse, A_sparse, M_sparse)  # (N, d_out)

        # Positive pairs (i<j)
        pos_u_all, pos_v_all = self._upper_triangle_edges_from_sparse(A_sparse, device=device)
        Epos = pos_u_all.numel()
        if Epos == 0:
            raise ValueError("No positive edges found in A.")

        # Dense forbid mask for negative sampling
        forbid = A_sparse.to_dense().bool().to(device)
        forbid.fill_diagonal_(True)

        # IMPORTANT: keep a Tensor accumulator (not float)
        total_loss = torch.zeros((), device=device, dtype=H.dtype)
        ptr = 0
        while ptr < Epos:
            take = min(max_pos_per_batch, Epos - ptr)
            u_pos = pos_u_all[ptr:ptr + take]
            v_pos = pos_v_all[ptr:ptr + take]

            # Negatives
            Kneg = take * max(1, int(neg_ratio))
            out_u, out_v, need = [], [], Kneg
            while need > 0:
                batch = int(need * 1.6) + 64
                u = torch.randint(0, N, (batch,), device=device)
                v = torch.randint(0, N, (batch,), device=device)
                keep = (u != v) & (~forbid[u, v])
                u, v = u[keep], v[keep]
                if u.numel() == 0:
                    continue
                take_neg = min(need, u.numel())
                out_u.append(u[:take_neg]);
                out_v.append(v[:take_neg])
                need -= take_neg
            u_neg = torch.cat(out_u);
            v_neg = torch.cat(out_v)

            # Logits
            T = H @ self.edge_scorer.W  # (N, d_out)
            pos_logits = (T[u_pos] * H[v_pos]).sum(-1)  # (take,)
            neg_logits = (T[u_neg] * H[v_neg]).sum(-1)  # (Kneg,)

            logits = torch.cat([pos_logits, neg_logits], 0)
            labels = torch.cat([torch.ones_like(pos_logits),
                                torch.zeros_like(neg_logits)], 0)

            batch_loss = F.binary_cross_entropy_with_logits(logits, labels)

            # Accumulate as tensor; weight by fraction to get epoch mean
            total_loss = total_loss + batch_loss * (take / Epos)

            ptr += take

        # This is a scalar tensor with grad_fn
        return total_loss

    # ---- FORWARD: directly return least-likely existing edges ----
    @torch.no_grad()
    def forward(
            self,
            X_sparse: torch.Tensor,
            A_sparse: torch.Tensor,
            M_sparse: Optional[torch.Tensor],
            b: int,
            existing_only: bool = False,  # <- NEW: set True to keep old behavior
            dense_limit: int = 7500,  # auto-switch threshold (N <= dense_limit -> dense path)
            block_size: int = 2000  # block size for the blockwise path
    ):
        """
        Returns (for u < v):
          edges_idx:  (2, b)  node indices (u; v)
          edge_probs: (b,)    smallest sigmoid(logit) across the chosen candidate set
          z:          (out_dim,)

        If existing_only=False (default): consider ALL pairs (complete graph, u<v).
        If existing_only=True:           consider ONLY existing edges from A (previous behavior).
        """
        device = X_sparse.device
        H = self.encode(X_sparse, A_sparse, M_sparse)  # (N, d_out)
        z, _ = self.pool(H)
        T = H @ self.edge_scorer.W  # (N, d_out)
        N = H.size(0)

        if existing_only:
            # ---- old behavior: only score existing edges (u<v) ----
            u_exist, v_exist = self._upper_triangle_edges_from_sparse(A_sparse, device=device)
            if u_exist.numel() == 0:
                raise ValueError("No edges in A (upper triangle).")
            logits = (T[u_exist] * H[v_exist]).sum(-1)
            probs = torch.sigmoid(logits)
            k = int(min(b, probs.numel()))
            vals, sel = torch.topk(probs, k=k, largest=False)
            edges_idx = torch.stack([u_exist[sel].cpu(), v_exist[sel].cpu()], dim=0)
            return edges_idx, vals.cpu(), z.cpu()

        # ---- new behavior: consider ALL pairs (complete graph upper triangle) ----
        # Fast dense path if feasible
        if N <= dense_limit:
            # S = (H W) H^T
            S = T @ H.T  # (N, N) logits
            tri = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)
            if not tri.any():
                raise ValueError("Graph too small to form (u<v) pairs.")
            u_all, v_all = tri.nonzero(as_tuple=True)  # (E_all,)
            probs = torch.sigmoid(S[tri])  # (E_all,)
            k = int(min(b, probs.numel()))
            vals, sel = torch.topk(probs, k=k, largest=False)
            edges_idx = torch.stack([u_all[sel].cpu(), v_all[sel].cpu()], dim=0)
            return edges_idx, vals.cpu(), z.cpu()

        # Memory-friendly blockwise path
        best_vals = None
        best_u = None
        best_v = None
        k_target = int(b)

        for i0 in range(0, N, block_size):
            i1 = min(N, i0 + block_size)
            Ti = T[i0:i1]  # (bi, d)
            for j0 in range(i0, N, block_size):
                j1 = min(N, j0 + block_size)
                Hj = H[j0:j1]  # (bj, d)

                # logits for block
                Sblk = Ti @ Hj.T  # (bi, bj)

                # mask to keep only upper-triangle pairs (u<v)
                if i0 == j0:
                    mask = torch.triu(
                        torch.ones(i1 - i0, j1 - j0, dtype=torch.bool, device=device),
                        diagonal=1
                    )
                else:
                    mask = torch.ones_like(Sblk, dtype=torch.bool)

                if not mask.any():
                    continue

                probs_blk = torch.sigmoid(Sblk[mask])  # (m,)

                # absolute indices for (u,v)
                ui, vj = mask.nonzero(as_tuple=True)
                u_abs = ui + i0
                v_abs = vj + j0

                if best_vals is None:
                    take = min(k_target, probs_blk.numel())
                    vals, sel = torch.topk(probs_blk, k=take, largest=False)
                    best_vals = vals
                    best_u = u_abs[sel]
                    best_v = v_abs[sel]
                else:
                    all_vals = torch.cat([best_vals, probs_blk], dim=0)
                    all_u = torch.cat([best_u, u_abs], dim=0)
                    all_v = torch.cat([best_v, v_abs], dim=0)
                    take = min(k_target, all_vals.numel())
                    vals, sel = torch.topk(all_vals, k=take, largest=False)
                    best_vals = vals
                    best_u = all_u[sel]
                    best_v = all_v[sel]

        if best_vals is None or best_vals.numel() == 0:
            raise RuntimeError("No candidate pairs found in blockwise pass.")

        edges_idx = torch.stack([best_u.cpu(), best_v.cpu()], dim=0)
        return edges_idx, best_vals.cpu(), z.cpu()

    @torch.no_grad()
    def init_block_highLP_add_remove(
            self,
            X_sparse: torch.Tensor,
            A_sparse: torch.Tensor,
            M_sparse: Optional[torch.Tensor],
            B: int,  # total block size (adds + removes)
            add_ratio: float = 0.9,  # fraction of B reserved for adds
            p_min: float = 0.6,
            p_max: float = 0.3, # plausibility threshold for both adds & removes
            per_node_cap: int = 64,  # max pairs that can touch the same node
            rank_by: str = "prob",  # "prob" or "slope" (p*(1-p))
            dense_limit: int = 7500,  # threshold for dense path
            block_size: int = 2000  # block size for blockwise path
    ):
        """
        Returns:
          edges_idx:   (2, M) stacked (u,v) with u<v; first adds then removes (M<=B if pool is small)
          edge_probs:  (M,) LP probabilities p(u,v) for those pairs
          is_add_mask: (M,) bool tensor (True -> add, False -> remove)
        Strategy:
          - Adds:    pick NON-EDGES with HIGH LP (p >= p_min)
          - Removes: pick EXISTING EDGES with HIGH LP (p >= p_min)
          - Within each pool, rank by 'prob' (descending) or 'slope' = p*(1-p) (descending)
          - Apply per-node caps to keep diversity
        """
        device = X_sparse.device
        H = self.encode(X_sparse, A_sparse, M_sparse)  # (N, d)
        T = H @ self.edge_scorer.W  # (N, d)
        N = H.size(0)

        def select_with_cap(u_all, v_all, score_all, k_target):
            if u_all.numel() == 0 or k_target <= 0:
                return (torch.empty(0, dtype=torch.long),) * 2 + (torch.empty(0, dtype=score_all.dtype),)

            order = torch.argsort(score_all, descending=True)
            u_all, v_all, score_all = u_all[order], v_all[order], score_all[order]

            taken = torch.zeros(N, dtype=torch.int32, device=device)
            sel = []

            for i in range(u_all.numel()):
                u, v = int(u_all[i]), int(v_all[i])
                if taken[u] < per_node_cap and taken[v] < per_node_cap:
                    sel.append(i)
                    taken[u] += 1
                    taken[v] += 1
                if len(sel) >= k_target:
                    break

            # If not enough selected, just fill from top of remaining list
            if len(sel) < k_target:
                # take remaining ones regardless of node cap
                missing = k_target - len(sel)
                all_indices = set(range(u_all.numel()))
                skipped = list(all_indices - set(sel))
                sel.extend(skipped[:missing])

            sel = torch.tensor(sel, device=device)
            return u_all[sel], v_all[sel], score_all[sel]

        # ---------- identify existing upper-tri edges ----------
        A = A_sparse.coalesce()
        uE, vE = A.indices()
        ut_mask = uE < vE
        uE, vE = uE[ut_mask], vE[ut_mask]

        # ---------- compute P(u,v) over all u<v ----------
        def all_pairs_upper():
            if N <= dense_limit:
                S = T @ H.T
                tri = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)
                ui, vj = tri.nonzero(as_tuple=True)
                P = torch.sigmoid(S[tri])
                # is existing?
                exist_dense = torch.zeros((N, N), dtype=torch.bool, device=device)
                if uE.numel() > 0:
                    exist_dense[uE, vE] = True
                is_exist = exist_dense[ui, vj]
                return ui, vj, P, is_exist
            # blockwise
            # store existing edges as linear indices on CPU for fast membership
            exist_lin = (uE.long().cpu() * N + vE.long().cpu())
            ui_all, vj_all, P_all, E_all = [], [], [], []
            for i0 in range(0, N, block_size):
                i1 = min(N, i0 + block_size)
                Ti = T[i0:i1]
                for j0 in range(i0, N, block_size):
                    j1 = min(N, j0 + block_size)
                    Hj = H[j0:j1]
                    Sblk = Ti @ Hj.T
                    if i0 == j0:
                        mask = torch.triu(torch.ones(i1 - i0, j1 - j0, dtype=torch.bool, device=device), diagonal=1)
                    else:
                        mask = torch.ones_like(Sblk, dtype=torch.bool)
                    if not mask.any():
                        continue
                    Pblk = torch.sigmoid(Sblk[mask])
                    ui, vj = mask.nonzero(as_tuple=True)
                    u_abs = (ui + i0).to(torch.long)
                    v_abs = (vj + j0).to(torch.long)
                    # membership test on CPU
                    cand_lin = (u_abs.cpu() * N + v_abs.cpu())
                    is_exist = torch.isin(cand_lin, exist_lin).to(device)
                    ui_all.append(u_abs);
                    vj_all.append(v_abs);
                    P_all.append(Pblk);
                    E_all.append(is_exist)
            return torch.cat(ui_all), torch.cat(vj_all), torch.cat(P_all), torch.cat(E_all)

        ui, vj, P, is_exist = all_pairs_upper()

        # ---------- split pools ----------
        # Adds: NON-edges with high LP
        add_mask = (~is_exist) & (P >= p_min)
        u_add, v_add, p_add = ui[add_mask], vj[add_mask], P[add_mask]

        # Removes: EXISTING edges with high LP
        rem_mask = is_exist & (P <= p_max)
        u_rem, v_rem, p_rem = ui[rem_mask], vj[rem_mask], P[rem_mask]

        # ---------- scoring within each pool ----------
        if rank_by == "prob":
            score_add = p_add
            score_rem = p_rem
        elif rank_by == "slope":
            score_add = p_add * (1.0 - p_add)  # boundary emphasis
            score_rem = p_rem * (1.0 - p_rem)
        else:
            raise ValueError("rank_by must be 'prob' or 'slope'.")

        # ---------- select with diversity caps ----------
        B_add = int(round(B * add_ratio))
        B_rem = B - B_add
        u_add, v_add, p_add = select_with_cap(u_add, v_add, score_add, B_add)
        u_rem, v_rem, p_rem = select_with_cap(u_rem, v_rem, score_rem, B_rem)
        # ---------- merge & return ----------
        u = torch.cat([u_add, u_rem], 0)
        v = torch.cat([v_add, v_rem], 0)
        probs = torch.cat([p_add, p_rem], 0)
        is_add_mask = torch.cat([
            torch.ones(u_add.numel(), dtype=torch.bool, device=device),
            torch.zeros(u_rem.numel(), dtype=torch.bool, device=device)
        ], 0)

        edges_idx = torch.stack([u.cpu(), v.cpu()], dim=0)
        return edges_idx, probs.cpu(), is_add_mask.cpu()