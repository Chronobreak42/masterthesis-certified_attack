import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GAEEncoder(nn.Module):

    """
       GAEEncoder is a feature-less graph encoder for unsupervised learning.
       It uses learnable node embeddings and a single GCN layer to compute
       latent representations.

       Variables:
       ----------
       num_nodes : int
           Number of nodes in the graph; used to create an embedding table.

       embed_dim : int
           Dimensionality of the learnable input embeddings.

       hidden_dim : int
           Dimensionality of the latent representation (output of the GCN).

       embedding : nn.Embedding
           A learnable table where each node index maps to a d-dimensional vector.

       conv : GCNConv
           A single GCN layer to aggregate and transform the node embeddings.
       """

    def __init__(self, num_nodes, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        self.conv = GCNConv(embed_dim, hidden_dim)

    def forward(self, edge_index):
        x = self.embedding.weight                      # X₀
        z = self.conv(x, edge_index)                   # Z = Â X W
        return F.normalize(z, p=2, dim=-1)             # optional