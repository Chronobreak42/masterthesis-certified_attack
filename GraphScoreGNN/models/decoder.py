import torch


class InnerProductDecoder(torch.nn.Module):
    """
    Decoder that reconstructs edges using the inner product of latent embeddings.

    Used in Graph Autoencoders to compute edge existence scores.

    Variables:
    ----------
    (no trainable parameters)

    Methods:
    --------
    forward(z, edge_index):
        Computes dot product between z[i] and z[j] for all (i, j) in edge_index.

    Parameters:
    -----------
    z : torch.Tensor
        Node embeddings of shape (num_nodes, hidden_dim).

    edge_index : torch.Tensor
        Edge index of shape (2, num_edges); each column is a node pair (i, j).

    Returns:
    --------
    torch.Tensor
        Logit scores (real numbers) representing how likely each edge is.
    """

    @staticmethod
    def forward(z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)  # inner product for each edge pair