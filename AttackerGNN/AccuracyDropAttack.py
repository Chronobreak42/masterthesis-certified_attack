# attacker_only.py
# Minimal attacker labeler. No victim model or graph construction included.
# Requires: numpy
import numpy as np
import random
from typing import List, Tuple, Callable

Random = random.Random

def default_eval_fn(A_clean: np.ndarray, A_pert: np.ndarray) -> float:
    """
    Example placeholder evaluation function.
    Returns a fake 'accuracy drop' (for testing). Replace with your real evaluator.
    """
    # simple heuristic: fraction of degree-sequence change (toy)
    deg_clean = A_clean.sum(axis=1)
    deg_pert = A_pert.sum(axis=1)
    return float(np.abs(deg_clean - deg_pert).sum()) / (2.0 * A_clean.shape[0])

def make_candidate_pool(A: np.ndarray, num_candidates: int, seed: int = 0) -> List[Tuple[int,int,str]]:
    """
    Build a pool of candidate flips (u,v,'add' or 'del').
    - For 'add': (u,v,'add') is a non-edge (u<v)
    - For 'del': (u,v,'del') is an existing edge (u<v)
    We return pairs with u < v to avoid duplicates for undirected graphs.
    """
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    # upper-tri indices
    iu, ju = np.triu_indices(N, k=1)
    pairs = list(zip(iu.tolist(), ju.tolist()))
    existing = [p for p in pairs if A[p] == 1]
    non_existing = [p for p in pairs if A[p] == 0]

    n_exist = min(len(existing), num_candidates // 2)
    n_non = min(len(non_existing), num_candidates - n_exist)

    chosen_exist = rng.choice(len(existing), size=n_exist, replace=False).tolist() if n_exist>0 else []
    chosen_non = rng.choice(len(non_existing), size=n_non, replace=False).tolist() if n_non>0 else []

    candidates = [(existing[i][0], existing[i][1], 'del') for i in chosen_exist] \
                 + [(non_existing[i][0], non_existing[i][1], 'add') for i in chosen_non]
    return candidates

def label_edge_flips(
    A_clean: np.ndarray,
    candidates: List[Tuple[int,int,str]],
    eval_fn: Callable[[np.ndarray, np.ndarray], float],
    p_add: float = 0.5,
    p_del: float = 0.5,
    drop_threshold: float = 0.05,
    rng_seed: int = 0
) -> np.ndarray:
    """
    For each candidate (u,v,action) decide probabilistically to flip it.
    If flipped, create A_pert, call eval_fn(A_clean, A_pert) -> drop (scalar).
    If drop > drop_threshold => y_out[i] = 1 else 0.
    Returns numpy array y_out (0/1) aligned with candidates order.
    """
    rnd = Random(rng_seed)
    N = A_clean.shape[0]
    y_out = np.zeros(len(candidates), dtype=np.int8)

    for i, (u, v, action) in enumerate(candidates):
        prob = p_add if action == 'add' else p_del
        if rnd.random() >= prob:
            # we chose NOT to flip this candidate (label remains 0)
            continue

        # copy adjacency and flip
        A_pert = A_clean.copy()
        if action == 'add':
            A_pert[u, v] = 1
            A_pert[v, u] = 1
        else:  # 'del'
            A_pert[u, v] = 0
            A_pert[v, u] = 0

        drop = float(eval_fn(A_clean, A_pert))
        if drop > drop_threshold:
            y_out[i] = 1

    return y_out

# ----------------- minimal usage example -----------------
if __name__ == "__main__":
    # Suppose you already have adjacency A (N x N numpy 0/1)
    # here's an example random A for demonstration only:
    N = 100
    rng = np.random.default_rng(1)
    A = (rng.random((N, N)) < 0.02).astype(np.int8)
    A = np.triu(A, k=1)
    A = A + A.T  # symmetric
    np.fill_diagonal(A, 0)

    # build candidates (no victim required)
    candidates = make_candidate_pool(A, num_candidates=200, seed=42)

    # get labels using a placeholder eval function (replace with your own)
    y_out = label_edge_flips(
        A_clean=A,
        candidates=candidates,
        eval_fn=default_eval_fn,   # ← replace with your real evaluator
        p_add=0.8,
        p_del=0.8,
        drop_threshold=0.03,
        rng_seed=123
    )

    print("num candidates:", len(candidates))
    print("labels sum (num harmful flips):", int(y_out.sum()))
    # each candidates[i] corresponds to y_out[i]