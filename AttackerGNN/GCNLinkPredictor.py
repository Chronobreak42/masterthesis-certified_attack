import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNLinkPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, out_feats)
        self.dropout = nn.Dropout(p=0.5)
        # bilinear decoder is stronger than raw dot-product
        self.bilinear = nn.Bilinear(out_feats, out_feats, 1, bias=False)
    def encode(self, x, edge_index):
        # GCN encoder: two layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x  # final node embeddings
    '''def decode(self, z, edge_index):
        # Dot-product decoder for link prediction
        src, dst = edge_index  # pair of node indices
        # Compute dot product similarity for each pair
        return (z[src] * z[dst]).sum(dim=-1)  # returns a score for each edge'''

    def decode(self, z, edge_index):
        src, dst = edge_index
        return self.bilinear(z[src], z[dst]).squeeze(-1)  # logits

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        scores = self.decode(z, edge_label_index)
        return scores