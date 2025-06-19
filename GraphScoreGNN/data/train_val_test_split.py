import torch
from typing import Dict, Tuple

def train_val_test_split(
    edge_index: torch.Tensor,
    num_nodes: int,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    directed: bool = False,
) -> Tuple[torch.Tensor,
           Dict[str, torch.Tensor],
           Dict[str, torch.Tensor]]:
    """
    Splits a graph’s edges into train / val / test sets and samples an
    equal-sized set of negative edges for each split.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list of shape (2, E) containing **all** positive edges
        in the (optionally undirected) graph.

    num_nodes : int
        Total number of nodes in the graph; needed to sample negatives.

    val_ratio : float, optional (default=0.05)
        Fraction of edges to assign to the validation split.

    test_ratio : float, optional (default=0.10)
        Fraction of edges to assign to the test split.

    directed : bool, optional (default=False)
        If ``False`` the graph is assumed undirected, so an edge (i, j)
        is treated as identical to (j, i) when checking for collisions
        during negative sampling.

    Returns
    -------
    edge_index_train : torch.Tensor
        The **training adjacency** (shape ``(2, E_train)``) –
        the graph you feed into message passing.
        *Validation and test positives are removed*.

    pos_edges : Dict[str, torch.Tensor]
        Dictionary with keys ``'train' | 'val' | 'test'`` mapping
        to the positive edge tensors for each split.

    neg_edges : Dict[str, torch.Tensor]
        Dictionary with keys ``'train' | 'val' | 'test'`` mapping
        to the *same-sized* negative edge tensors for each split.
    """
    # --- 1.  Randomly permute edges and slice -----------------------------
    E = edge_index.size(1)
    perm = torch.randperm(E)
    num_val  = int(E * val_ratio)
    num_test = int(E * test_ratio)

    pos_edges = {
        'test':  edge_index[:, perm[:num_test]],
        'val':   edge_index[:, perm[num_test:num_test + num_val]],
        'train': edge_index[:, perm[num_test + num_val:]],
    }

    # --- 2.  Remove val/test edges from the graph used for message passing -
    edge_index_train = pos_edges['train']

    # --- 3.  Generate negative edges for each split -----------------------
    from GraphScoreGNN.utils.sampler import sample_negative_edges  # your earlier util

    neg_edges = {}
    for split, pos_e in pos_edges.items():
        neg_edges[split] = sample_negative_edges(
            pos_edges['train'],          # we *avoid* test/val edges on purpose
            num_nodes,
            num_samples=pos_e.size(1),
            directed=directed,
        )

    return edge_index_train, pos_edges, neg_edges