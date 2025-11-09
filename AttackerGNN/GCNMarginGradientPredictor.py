import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_adj(A: torch.Tensor) -> torch.Tensor:
    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    A_ = A + I
    d = A_.sum(1).clamp(min=1e-12).pow(-0.5)
    D_ = torch.diag(d)
    return D_ @ A_ @ D_

class TinyGCN(nn.Module):
    def __init__(self, in_feats: int, hidden: int, out_feats: int):
        super().__init__()
        self.W1 = nn.Linear(in_feats, hidden, bias=False)
        self.W2 = nn.Linear(hidden, out_feats, bias=False)
    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        Ahat = normalize_adj(A)
        H = F.relu(Ahat @ self.W1(X))
        Z = Ahat @ self.W2(H)
        return Z  # (N, C) logits

def tanh_margin_loss_label_free(logits: torch.Tensor) -> torch.Tensor:
    top2 = logits.topk(2, dim=1).values               # (N, 2)
    margin = top2[:, 0] - top2[:, 1]                  # z_score = z_top1 - z_top2
    return (-torch.tanh(margin)).mean()