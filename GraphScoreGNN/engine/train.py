import torch
import torch.nn.functional as F


def _bce_loss(pos_logits, neg_logits):
    """
    Computes average binary cross-entropy loss for positive and negative edge logits.

    Parameters:
    -----------
    pos_logits : torch.Tensor
        Logits for positive (existing) edges.

    neg_logits : torch.Tensor
        Logits for negative (non-existent) edges.

    Returns:
    --------
    torch.Tensor
        Scalar loss value.
    """
    pos_loss = F.binary_cross_entropy_with_logits(pos_logits,
                                                  torch.ones_like(pos_logits))
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits,
                                                  torch.zeros_like(neg_logits))
    return (pos_loss + neg_loss) / 2


def train_epoch(model, decoder, optimizer, edge_index,
                pos_edge, neg_edge):
    """
    One epoch of training for a GAE.

    Parameters:
    -----------
    model : nn.Module
        GAE encoder model.

    decoder : nn.Module
        Decoder model (e.g., InnerProductDecoder).

    optimizer : torch.optim.Optimizer
        Optimizer for model parameters.

    edge_index : torch.Tensor
        Full graph edge list used for message passing (2, E).

    pos_edge : torch.Tensor
        Positive training edges (2, N_pos).

    neg_edge : torch.Tensor
        Negative sampled edges (2, N_neg).

    Returns:
    --------
    float
        Average loss over all edge pairs.
    """
    model.train()
    optimizer.zero_grad()
    z = model(edge_index)
    pos_logits = decoder(z, pos_edge)
    neg_logits = decoder(z, neg_edge)
    loss = _bce_loss(pos_logits, neg_logits)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_epoch(model, decoder, edge_index,
               pos_edge, neg_edge, metric_fn):
    """
    Evaluates the model on a set of edges using a binary classification metric.

    Parameters:
    -----------
    model : nn.Module
        GAE encoder model.

    decoder : nn.Module
        Decoder model.

    edge_index : torch.Tensor
        Edge list for message passing.

    pos_edge : torch.Tensor
        Positive evaluation edges.

    neg_edge : torch.Tensor
        Negative evaluation edges.

    metric_fn : callable
        Function that computes a metric (e.g. ROC AUC) from predictions and labels.

    Returns:
    --------
    float
        Evaluation score (e.g. AUC).
    """
    model.eval()
    z = model(edge_index)
    logits = torch.cat([decoder(z, pos_edge),
                        decoder(z, neg_edge)])
    labels = torch.cat([torch.ones(pos_edge.size(1)),
                        torch.zeros(neg_edge.size(1))]).to(logits.device)
    return metric_fn(labels.cpu().numpy(), logits.cpu().sigmoid().numpy())