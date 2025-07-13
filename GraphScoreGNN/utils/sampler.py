import torch

def sample_binary_perm(edge_index):
    """
    Randomly samples and deletes edges from the edge index

    Parameters:
    -----------
    edge_index: torch.Tensor
        edge index matrix.

    Returns:
    --------
    mod_edge_index: torch.Tensor
        randomly permutated edge index tensor.
    """

    mod_edge_index = edge_index.

def sample_negative_edges(
        edge_index: torch.Tensor,
        num_nodes: int,
        num_samples: int,
        directed: bool = False
) -> torch.Tensor:
    """
    Randomly samples negative edges — node pairs (i, j) that do not
    exist in the given edge_index.

    Parameters
    ----------
    edge_index : torch.Tensor
        Positive edges of shape (2, E). All edges to be avoided.

    num_nodes : int
        Total number of nodes in the graph.

    num_samples : int
        Number of negative edges to sample.

    directed : bool, optional (default=False)
        If False, treats (i, j) as equal to (j, i) and avoids both directions.

    Returns
    -------
    neg_edge_index : torch.Tensor
        Tensor of shape (2, num_samples) containing sampled node pairs
        that are not in the original edge list.
    """
    # --- Create set of forbidden edges (positive edges) ------------------
    E = edge_index.size(1)
    pos_set = set()
    for i in range(E):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        pos_set.add((u, v))
        if not directed:
            pos_set.add((v, u))

    # --- Sample until we have enough valid negatives ---------------------
    neg_edges = set()
    while len(neg_edges) < num_samples:
        i = torch.randint(0, num_nodes, (num_samples,))
        j = torch.randint(0, num_nodes, (num_samples,))
        for u, v in zip(i.tolist(), j.tolist()):
            if u == v:  # skip self-loops
                continue
            if (u, v) in pos_set:
                continue
            if not directed and (v, u) in pos_set:
                continue
            if (u, v) in neg_edges:
                continue
            neg_edges.add((u, v))
            if len(neg_edges) == num_samples:
                break

    # --- Convert to tensor format (2, num_samples) -----------------------
    neg_edges = torch.tensor(list(neg_edges), dtype=torch.long).t()
    return neg_edges