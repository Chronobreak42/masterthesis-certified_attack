import torch
import argparse
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected

from GraphScoreGNN.data.train_val_test_split import train_val_test_split
from GraphScoreGNN.utils.seed import set_seed
from GraphScoreGNN.models.encoder import GAEEncoder
from GraphScoreGNN.models.decoder import InnerProductDecoder
from GraphScoreGNN.engine.train import train_epoch, eval_epoch
from sklearn.metrics import roc_auc_score

def main():
    """
    Main entry point for training a feature-less Graph Autoencoder (GAE)
    on a citation graph dataset (e.g., Cora, Citeseer, Pubmed).

    This pipeline uses:
    - Learnable node embeddings (no input features)
    - A single-layer GCN encoder
    - Inner-product decoder
    - Binary cross-entropy loss on edge existence (pos/neg)

    """

    # Argument parser
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='Cora', type=str,
                   help='Name of the dataset (Cora, Citeseer, Pubmed)')
    p.add_argument('--epochs', default=200, type=int,
                   help='Number of training epochs')
    p.add_argument('--embed_dim', default=64, type=int,
                   help='Dimension of node embeddings')
    p.add_argument('--hidden_dim', default=64, type=int,
                   help='Output dimension of GCN encoder')
    p.add_argument('--lr', default=1e-3, type=float,
                   help='Learning rate for optimizer')
    p.add_argument('--seed', default=42, type=int,
                   help='Seed for reproducibility')
    cfg = p.parse_args()

    # Set reproducible seed
    set_seed(cfg.seed)

    # Choose device
    device = torch.device('cpu')

    # Load graph data (no features used)
    data = torch.load(r'E:\Masterarbeit\ProjectCombined\cache\demo\demo_1.pt')
    data.edge_index = to_undirected(data.edge_index)

    # Split edges into train/val/test + negative samples
    pos_edges, neg_edges = train_val_test_split(data.edge_index, data.num_nodes)
    # Each of pos_edges and neg_edges is a dict with keys: 'train', 'val', 'test'

    # Initialize model components
    encoder = GAEEncoder(
        num_nodes=data.num_nodes,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim
    ).to(device)

    decoder = InnerProductDecoder().to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.lr)

    # Training loop
    for epoch in range(1, cfg.epochs + 1):
        loss = train_epoch(
            model=encoder,
            decoder=decoder,
            optimizer=optimizer,
            edge_index=data.edge_index.to(device),
            pos_edge=pos_edges['train'].to(device),
            neg_edge=neg_edges['train'].to(device)
        )

        if epoch % 10 == 0 or epoch == cfg.epochs:
            val_auc = eval_epoch(
                model=encoder,
                decoder=decoder,
                edge_index=data.edge_index.to(device),
                pos_edge=pos_edges['val'].to(device),
                neg_edge=neg_edges['val'].to(device),
                metric_fn=roc_auc_score
            )
            print(f"[Epoch {epoch:03d}] Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")

    # Final evaluation on test set
    test_auc = eval_epoch(
        model=encoder,
        decoder=decoder,
        edge_index=data.edge_index.to(device),
        pos_edge=pos_edges['test'].to(device),
        neg_edge=neg_edges['test'].to(device),
        metric_fn=roc_auc_score
    )
    print(f"\nTest ROC AUC: {test_auc:.4f}")

if __name__ == '__main__':
    main()