import copy

import numpy as np
import torch
from sacred import Experiment

from rgnn_at_scale.data import prep_graph
from rgnn_at_scale.helper.io import Storage
from sparse_smoothing.models import GCN
from sparse_smoothing.utils import load_and_standardize
from sparse_smoothing.prediction import predict_smooth_gnn

# there are many parameters that are inputs of the function call like seed, other I have already taken from the function call so just change their position in code

model_storage_type = 'demo',
binary_attr = False
make_undirected = True
dataset = 'cora_ml'
model_params = dict(
    label="Vanilla GCN",
    model="GCN",
    do_cache_adj_prep=True,
    n_filters=64,
    dropout=0.5,
    svd_params=None,
    jaccard_params=None,
    gdc_params={"alpha": 0.15, "k": 64})
hyperparams = dict(model_params)
train_params = dict(
    lr=1e-2,
    weight_decay=1e-3,
    patience=300,
    max_epochs=3000)
ppr_cache = None
ppr_cache_params = dict()
if ppr_cache_params is not None:
    ppr_cache = dict(ppr_cache_params)
    ppr_cache.update(dict(
        dataset=dataset,
        make_undirected=make_undirected,
    ))
data_dir = './data'
data_device = "cpu"
graph = prep_graph(dataset, data_device, dataset_root=data_dir, make_undirected=make_undirected,
                   binary_attr=binary_attr, return_original_split=dataset.startswith('ogbn'))
attr, adj, labels = graph[:3]
n_features = attr.shape[1]
n_classes = int(labels[~labels.isnan()].max() + 1)
hyperparams.update({
    'n_features': n_features,
    'n_classes': n_classes,
    'ppr_cache_params': ppr_cache,
    'train_params': train_params
})
params = None


def setParams(seed):
    global params
    params = dict(dataset=dataset, binary_attr=binary_attr, make_undirected=make_undirected,
                  seed=seed, **hyperparams)


def overwriteModel(new_adj):
    # make model as an input aswell
    artifact_dir = 'cache'
    ex = Experiment()
    storage = Storage(artifact_dir, experiment=ex)
    models_and_hyperparams = storage.find_models('demo', params)
    for model, hyperparams in models_and_hyperparams:
        model.adj = new_adj
        storage.save_model('demo', params, model)


def loadModel():
    loaded_from_robustness = torch.load(f'cache/demo/demo_1.pt', map_location='cpu', weights_only=False)
    trained_state_dict = copy.deepcopy(loaded_from_robustness)
    for key in list(trained_state_dict.keys()):
        if 'layers.0.' in key:
            new_key = key.replace('layers.0.gcn_0', 'conv1')
            trained_state_dict[new_key] = trained_state_dict.pop(key)  # Transpose if necessary
    for key in list(trained_state_dict.keys()):
        if 'layers.1.' in key:
            new_key = key.replace('layers.1.gcn_1', 'conv2')
            trained_state_dict[new_key] = trained_state_dict.pop(key)  # Transpose if necessary
    graph_ = load_and_standardize(f"data/{dataset}.npz")
    # graph = load_and_standardize('data/pubmed.npz')
    n, d = graph_.attr_matrix.shape
    nc = graph_.labels.max() + 1
    model_ = GCN(n_features=d, n_classes=nc, n_hidden=64)  # .cuda()
    # Load the modified state dictionary
    model_.load_state_dict(trained_state_dict)
    return model_

def computeNewCertificate():
    pf_plus_adj = 0.01  # 0
    pf_minus_adj = 0.3  # 0
    pf_plus_att = 0
    pf_minus_att = 0
    n_samples_eval = 2500
    sample_config = {
        'n_samples': n_samples_eval,
        'pf_plus_adj': pf_plus_adj,
        'pf_minus_adj': pf_minus_adj,
        'pf_plus_att': pf_plus_att,
        'pf_minus_att': pf_minus_att,
    }
    sample_config_pre_eval = sample_config.copy()
    sample_config_pre_eval['n_samples'] = 50
    batch_size = 25
    conf_alpha = 0.1
    model_ = loadModel()
    graph_ = load_and_standardize(f"data/{dataset}.npz")
    edge_idx = torch.LongTensor(np.stack(graph_.adj_matrix.nonzero()))  # .cuda()
    attr_idx = torch.LongTensor(np.stack(graph_.attr_matrix.nonzero()))  # .cuda()

    n, d = graph_.attr_matrix.shape
    nc = graph_.labels.max() + 1


    # we a small number of samples to estimate the majority class
    pre_votes = predict_smooth_gnn(attr_idx=attr_idx, edge_idx=edge_idx,
                                   sample_config=sample_config_pre_eval,
                                   model=model_, n=n, d=d, nc=nc,
                                   batch_size=batch_size)

    # we use a larger number of samples to estimate a lower bound
    # on the probability of observing the majority class
    votes = predict_smooth_gnn(attr_idx=attr_idx, edge_idx=edge_idx,
                               sample_config=sample_config,
                               model=model_, n=n, d=d, nc=nc,
                               batch_size=batch_size)

    from sparse_smoothing.cert import p_lower_from_votes, binary_certificate_grid

    # conf_alpha = 0.01

    # compute the lower bound on the probability of the majority class
    p_lower = p_lower_from_votes(votes=votes, pre_votes=pre_votes, alpha=conf_alpha, n_samples=n_samples_eval)


    grid_binary_class, *_ = binary_certificate_grid(pf_plus=pf_plus_adj, pf_minus=pf_minus_adj,
                                                                p_emps=p_lower, reverse=False, progress_bar=True)


    grid_threshold = 0.5
    grid_radii = (grid_binary_class > grid_threshold)

    return grid_binary_class, grid_radii
