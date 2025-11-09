import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv  # or SAGEConv, GATConv, etc.
    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False


# ---------------------------
# Utility: all unordered pairs
# ---------------------------
def triu_pairs(N, device):
    iu, ju = torch.triu_indices(N, N, offset=1, device=device)
    return iu.long(), ju.long()  # each (C,), C=N*(N-1)//2


# ---------------------------------------
# Utility: map K-block weights to full-C
# ---------------------------------------
def lift_block_weights_to_fullC(C, current_search_space, perturbed_edge_weight, device):
    """
    current_search_space: (K,) linear triu indices (CPU or device)
    perturbed_edge_weight: (K,) on device
    returns: full_prev_weight: (C,) on device
    """
    full_prev = torch.zeros(C, device=device)
    if current_search_space is None or current_search_space.numel() == 0:
        return full_prev
    lin = current_search_space.to(device=device, dtype=torch.long)
    full_prev[lin] = perturbed_edge_weight.to(device)
    return full_prev


# ---------------------------------------
# Gumbel utilities
# ---------------------------------------
def sample_gumbel(shape, device):
    # Gumbel(0,1): -log(-log(U))
    U = torch.empty(shape, device=device).uniform_(0, 1)
    return -torch.log(-torch.log(U + 1e-20) + 1e-20)

def gumbel_topk_straight_through(logits, K, tau=0.7):
    C = logits.numel()
    device = logits.device

    z_work = logits.clone()
    soft_acc = torch.zeros(C, device=device)
    hard_idxs = []

    for _ in range(K):
        # soft surrogate each step
        soft_t = torch.nn.functional.softmax(z_work / tau, dim=0)
        soft_acc += soft_t

        U = torch.empty_like(z_work).uniform_(0, 1)
        g = -torch.log(-torch.log(U + 1e-20) + 1e-20)

        pick = torch.argmax(z_work + g)
        hard_idxs.append(pick)
        z_work[pick] = -1e9  # Sample without replacement (W.I.P.)

    hard_idx = torch.stack(hard_idxs).long()
    return hard_idx, soft_acc, torch.tensor(0.0, device=device)

# --------------------------
# Node encoder (GCN or MLP)
# --------------------------
class NodeEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=128):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden)
        self.gcn2 = GCNConv(hidden, out_dim)
        self.use_pyg = True

    def forward(self, x, edge_index=None, edge_weight=None):
        if self.use_pyg:
            h = self.gcn1(x, edge_index, edge_weight)
            h = F.relu(h, inplace=True)
            h = self.gcn2(h, edge_index, edge_weight)
            return h


# --------------------------
# Sampler GNN (main module)
# --------------------------
class SamplerGNN(nn.Module):
    """
    Produces:
      - hard modified_edge_index (2, K) for PRBCD forward
      - current_search_space (K,) linear triu indices
      - soft_selection (C,) for differentiable PRBCD training (backprop path)

    Inputs:
      X: (N, F) node features
      base_edge_index, base_edge_weight: for node encoder (PyG style)
      A_base: (N, N) or access to compute sign[u,v]
      block_size (K), tau
      prev (K,) perturbed_edge_weight and (K,) current_search_space for context
    """
    def __init__(self, in_dim, gnn_hidden=128, pair_hidden=128):
        super().__init__()
        self.encoder = NodeEncoder(in_dim, hidden=gnn_hidden, out_dim=gnn_hidden)

    @torch.no_grad()
    def _sign_from_A(self, N, A_base, iu, ju):
        # +1 if insertion (no edge), -1 if deletion (edge exists)
        is_edge = (A_base[iu, ju] > 0).float()
        sign = torch.where(is_edge > 0, torch.tensor(-1.0, device=A_base.device), torch.tensor(+1.0, device=A_base.device))
        return sign

    def forward(
        self,
        X,                          # (N, F)
        base_edge_index,            # (2, E) for encoder
        base_edge_weight,           # (E,)
        A_base,                     # (N, N) float/binary
        block_size,                 # K
        tau=0.7,
        prev_current_search_space=None,  # (K,) linear triu idx (CPU or device) or None
        prev_perturbed_edge_weight=None, # (K,) (device) or None
    ):
        device = X.device
        N = X.shape[0]
        iu, ju = triu_pairs(N, device)  # (C,)
        C = iu.numel()

        H = self.encoder(X, base_edge_index, base_edge_weight)  # (N, D)
        Hu, Hv = H[iu], H[ju]                                   # (C, D)

        sign = self._sign_from_A(N, A_base, iu, ju)             # (C,)
        if (prev_current_search_space is not None) and (prev_perturbed_edge_weight is not None):
            full_prev = lift_block_weights_to_fullC(
                C, prev_current_search_space, prev_perturbed_edge_weight, device)
        else:
            full_prev = torch.zeros(C, device=device)

        hadamard = Hu * Hv                                      # (C, D)
        pair_feat = torch.cat([Hu, Hv, hadamard, sign.unsqueeze(1), full_prev.unsqueeze(1)], dim=1)  # (C, 3D+2)
        logits = self.pair_head(pair_feat)                      # (C,)

        # --- Gumbel Top-K straight-through ---
        hard_idx, soft_acc, log_prob = gumbel_topk_straight_through(logits, K=block_size, tau=tau)  # (K,), (C,), scalar

        # --- Hard outputs for PRBCD bookkeeping ---
        hard_u = iu[hard_idx].detach().cpu()
        hard_v = ju[hard_idx].detach().cpu()
        modified_edge_index = torch.stack([hard_u, hard_v], dim=0)  # (2, K) on CPU (PRBCD-friendly)

        # linear triu indices for current_search_space (compute on CPU)
        lin = hard_u * (2*N - hard_u - 1) // 2 + (hard_v - hard_u - 1)  # fast triu linear index
        current_search_space = lin.long().cpu()  # (K,)

        # Return everything needed
        return {
            "modified_edge_index": modified_edge_index,   # (2, K) CPU long
            "current_search_space": current_search_space, # (K,)  CPU long
            "soft_selection": soft_acc,                   # (C,)  device float (backprop path)
            "pair_logits": logits,                        # (C,)  device (for aux losses/monitoring)
            "gumbel_log_prob": log_prob,                  # scalar (optional)
            "iu": iu, "ju": ju, "sign": sign,             # keep for downstream A_pert build
        }