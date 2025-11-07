import logging

from collections import defaultdict
import math
from typing import Tuple, Optional

import pandas as pd
from torch.nn import BCEWithLogitsLoss

import AttackerGNN.PGDTopologyAttack as pgdtop
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch_sparse
from torch_sparse import SparseTensor
#from AttackerGNN.PreEdgeSelector import PRBCDSelectorGNN
from AttackerGNN.PriorSelector import PriorSelector
from AttackerGNN.NodeBlockScorer import NodeBlockScorer
import AttackerGNN.gnn_least_likely_edge as lle
import AttackerGNN.gnn_score_all as sall
from AttackerGNN.GCNLinkPredictor import GCNLinkPredictor
from AttackerGNN.GCNMarginGradientPredictor import TinyGCN, tanh_margin_loss_label_free

# from rgnn_at_scale.models import MODEL_TYPE
from rgnn_at_scale.helper import utils
from rgnn_at_scale.attacks.base_attack import Attack, SparseAttack


class PRBCD(SparseAttack):
    """Sampled and hence scalable PGD attack for graph data.
    """

    def __init__(self,
                 keep_heuristic: str = 'WeightOnly',
                 lr_factor: float = 100,
                 display_step: int = 20,
                 epochs: int = 400,
                 fine_tune_epochs: int = 100,
                 block_size: int = 1_000_000,
                 with_early_stopping: bool = True,
                 do_synchronize: bool = False,
                 eps: float = 1e-7,
                 max_final_samples: int = 20,
                 pre_hidden: int = 64, # New
                 **kwargs):
        super().__init__(**kwargs)

        # --- Pre-GNN definition ---
        # Map original node features to a smaller hidden space
        '''
        self.pre_gnn = PRBCDSelectorGNN(
                    in_channels = self.attr.shape[1],
                    hidden_dim = pre_hidden,
                    num_pairs = 64
                    )
        '''

        self.pre_gnn_prior = PriorSelector(
            in_channels = self.attr.shape[1],
            hidden_dim = pre_hidden,
        )

        self.selector = NodeBlockScorer(
            in_channels=self.attr.shape[1],
            hidden_dim=pre_hidden,  # reuse your arg
        ).to(self.device)

        self._margin_trained = False

        self.keep_heuristic = keep_heuristic
        self.display_step = display_step
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.epochs_resampling = epochs - fine_tune_epochs
        self.block_size = block_size
        self.with_early_stopping = with_early_stopping
        self.eps = eps
        self.do_synchronize = do_synchronize
        self.max_final_samples = max_final_samples

        self.intra_decay_mode = kwargs.get("intra_decay_mode", "exp")
        self.intra_halflife_epochs = int(max(1, kwargs.get("intra_halflife_epochs", 5)))
        self.initial_max_intra_ratio = 0.2  # start ratio
        self.final_max_intra_ratio = 0.0  # end ratio after all resampling steps

        self.current_search_space: torch.Tensor = None
        self.current_node_search_space: torch.Tensor = None
        self.sample_space: torch.Tensor = None
        self.edges_to_attack_index: torch.Tensor = torch.empty((2, 0), dtype=torch.long)
        self.modified_edge_index: torch.Tensor = None
        self.perturbed_edge_weight: torch.Tensor = None
        self.semi = None

        self.make_pgd_forward_from_victim()

        # --- selector optimizer for online updates at resampling ---
        if getattr(self, "selector", None) is not None:
            self.selector_opt = torch.optim.Adam(
                self.selector.parameters(), lr=1e-3, weight_decay=5e-4
            )
        else:
            self.selector_opt = None

        if self.make_undirected:
            self.n_possible_edges = self.n * (self.n - 1) // 2
        else:
            self.n_possible_edges = self.n ** 2  # We filter self-loops later

        self.lr_factor = lr_factor * max(math.log2(self.n_possible_edges / self.block_size), 1.)

    def _attack(self, graph, n_perturbations, semi=False, use_cert="none", grid_radii: Optional[np.ndarray] = None, grid_binary_class: Optional[np.ndarray] = None, **kwargs):
        """Perform attack (`n_perturbations` is increasing as it was a greedy attack).

        Parameters
        ----------
        n_perturbations : int
            Number of edges to be perturbed (assuming an undirected graph)
        """
        self.semi = semi
        self.use_cert = use_cert

        assert self.block_size > n_perturbations, \
            f'The search space size ({self.block_size}) must be ' \
            + f'greater than the number of permutations ({n_perturbations})'

        # For early stopping (not explicitly covered by pesudo code)
        best_accuracy = float('Inf')
        best_epoch = float('-Inf')

        # For collecting attack statistics
        self.attack_statistics = defaultdict(list)

        # Sample initial search space (Algorithm 1, line 3-4)
        if use_cert in ("sampling_grid_radii", "sampling_grid_radii_alt_11", "both_1", "both_2_random"):
            self.sample_block_from_certificates_radii(grid_radii=grid_radii, n_perturbations=n_perturbations)
        elif use_cert in ("sampling_grid_binary_class", "sampling_grid_binary_class_alt_11", "sampling_grid_binary_class_alt_22", "both_2"):
            print(use_cert, "run sampling_grid_binary_class")
            self.sample_block_from_certificates_binary_class(grid_binary_class=grid_binary_class, n_perturbations=n_perturbations)
        elif use_cert in ("sampling_with_prior",):
            print(use_cert, "run sampling_with_prior")
            self.sample_block_from_prior_gumbel(n_perturbations=n_perturbations, tau=1.0)
        elif use_cert in ("selector_block",):  # here is my selector
            print(use_cert, "-> using selector to build the initial block")
            self.sample_block_from_selector(n_perturbations=n_perturbations, k_nodes=350)
        elif use_cert in ("selector_block_pgd",):  # here is my selector
            print(use_cert, "-> using selector to build the initial block")
            self.sample_block_from_pgdtopk_direct()
        elif use_cert in ("selector_block_linkpred",):
            print(use_cert, "-> using selector link pred")
            self.sample_block_from_linkpred_gnn(graph=graph,n_perturbations=n_perturbations)
        elif use_cert in ("selector_block_linkpred_all",):
            print(use_cert, "-> using selector link pred all")
            self.sample_block_from_linkpred_gnn_all_pairs(graph=graph, b=50000)
        elif use_cert in ("selector_block_linkpred_gcn",):
            print(use_cert, "-> using selector link pred gcn")
            self.sample_block_from_linkpred_gcn(graph=graph)
        elif use_cert in ("selector_block_gradient_gcn",):
            print(use_cert, "-> using selector gradient gcn")
            self.sample_block_from_margin_loss_gcn(graph=graph)
        else:
            print(use_cert, "run sampling with no certificate")
            self.sample_random_block(n_perturbations)

        # Accuracy and attack statistics before the attack even started
        with torch.no_grad():

            '''
            self.perturbed_edge_weight.retain_grad()
            edge_index_dbg, edge_weight_dbg = self.get_modified_adj()
            edge_weight_dbg.retain_grad()
            

            print("grad on perturbed_edge_weight:",
                  None if self.perturbed_edge_weight.grad is None else self.perturbed_edge_weight.grad.abs().sum().item())
            print("grad on fused edge_weight:",
                  None if edge_weight_dbg.grad is None else edge_weight_dbg.grad.abs().sum().item())
            '''


            logits = self._get_logits(self.attr, self.edge_index, self.edge_weight)
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
            accuracy = utils.accuracy(logits, self.labels, self.idx_attack)

            logging.info(f'\nBefore the attack - Loss: {loss.item()} Accuracy: {100 * accuracy:.3f} %\n')

            self._append_attack_statistics(loss.item(), accuracy, 0., 0.)

            del logits, loss

        # Loop over the epochs (Algorithm 1, line 5)
        for epoch in tqdm(range(self.epochs)):
            self.perturbed_edge_weight.requires_grad = True

            # Retreive sparse perturbed adjacency matrix `A \oplus p_{t-1}` (Algorithm 1, line 6)
            edge_index, edge_weight = self.get_modified_adj()

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Calculate logits for each node (Algorithm 1, line 6)
            logits = self._get_logits(self.attr, edge_index, edge_weight)
            # Calculate loss combining all each node (Algorithm 1, line 7)
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack]) #Todo: Hier wird der loss und gradient für perturbed edge weight erzeugt.
            # Retreive gradient towards the current block (Algorithm 1, line 7)
            gradient = utils.grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]
            self.gradient = gradient

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            with torch.no_grad():
                # Gradient update step (Algorithm 1, line 7)
                edge_weight = self.update_edge_weights(n_perturbations, epoch, gradient)[1]
                # For monitoring
                probability_mass_update = self.perturbed_edge_weight.sum().item()
                # Projection to stay within relaxed `L_0` budget (Algorithm 1, line 8)
                self.perturbed_edge_weight = Attack.project(
                    n_perturbations, self.perturbed_edge_weight, self.eps)
                # For monitoring
                probability_mass_projected = self.perturbed_edge_weight.sum().item()

                # Calculate accuracy after the current epoch (overhead for monitoring and early stopping)
                edge_index, edge_weight = self.get_modified_adj()
                logits = self.attacked_model(data=self.attr.to(self.device), adj=(edge_index, edge_weight))
                accuracy = utils.accuracy(logits, self.labels, self.idx_attack)

                del edge_index, edge_weight, logits

                if epoch % self.display_step == 0:
                    logging.info(f'\nEpoch: {epoch} Loss: {loss} Accuracy: {100 * accuracy:.3f} %\n')

                # Save best epoch for early stopping (not explicitly covered by pesudo code)
                if self.with_early_stopping and best_accuracy > accuracy:
                    best_accuracy = accuracy
                    best_epoch = epoch
                    best_search_space = self.current_search_space.clone().cpu()
                    best_edge_index = self.modified_edge_index.clone().cpu()
                    best_edge_weight_diff = self.perturbed_edge_weight.detach().clone().cpu()

                self._append_attack_statistics(loss, accuracy, probability_mass_update, probability_mass_projected)

                # Resampling of search space (Algorithm 1, line 9-14)
                if epoch < self.epochs_resampling - 1:
                    if use_cert in ("resampling_grid_radii", "both_1"):
                        print(use_cert, "run resampling_grid_radii")
                        self.resample_random_block_from_cert_radii(grid_radii=grid_radii,
                                                                   n_perturbations=n_perturbations)
                    elif use_cert in ("resampling_grid_binary_class", "both_2", "both_2_random"):
                        print(use_cert, "run resampling_grid_binary_class")
                        self.resample_random_block_from_cert_binary_class(grid_binary_class=grid_binary_class,
                                                                  n_perturbations=n_perturbations)
                    elif use_cert in ("sampling_with_prior",):
                        print(use_cert, "run resampling_with_prior")
                        self.resample_block_from_prior_gumbel(n_perturbations=n_perturbations, tau=1.0)


                    elif use_cert in ("selector_block", "selector_block_pgd"):

                        # --- Configurable decay mode ---
                        # Choose between: "exp" | "linear" | "none"
                        decay_mode = getattr(self, "intra_decay_mode", "exp")

                        # Initial and final intra ratios (set these somewhere in __init__)
                        init_ratio = float(self.initial_max_intra_ratio)
                        final_ratio = float(getattr(self, "final_max_intra_ratio", 0.0))

                        # --- Compute current intra ratio ---
                        if decay_mode == "exp":
                            # Exponential decay: halve every 10 epochs
                            halving_steps = max(0, int(epoch // 10))
                            current_max_intra = init_ratio * (0.5 ** halving_steps)

                        elif decay_mode == "linear":
                            # Linear decay over all resampling epochs
                            total_resample_steps = max(1, getattr(self, "epochs_resampling", 1) - 1)
                            resample_step = min(epoch, total_resample_steps)
                            t = float(resample_step) / float(total_resample_steps)
                            current_max_intra = init_ratio * (1.0 - t) + final_ratio * t

                        elif decay_mode == "none":
                            # No decay: keep constant
                            current_max_intra = init_ratio

                        else:
                            raise ValueError(f"Unknown intra decay mode: {decay_mode}")

                        # --- Clamp and log ---
                        current_max_intra = max(final_ratio, min(1.0, current_max_intra))
                        logging.info(
                            f"[Resample] epoch={epoch} | mode={decay_mode} | max_intra_ratio={current_max_intra:.4f}"
                        )

                        # --- Perform resampling ---
                        self.resample_block_from_selector(
                            n_perturbations=n_perturbations,
                            blend_alpha=0,
                            prefer_edges="intra",
                            max_intra_ratio=current_max_intra,
                        )
                    else:
                        print(use_cert, "run resampling with no certificate")
                        self.resample_random_block(n_perturbations)
                elif self.with_early_stopping and epoch == self.epochs_resampling - 1:
                    # Retreive best epoch if early stopping is active (not explicitly covered by pesudo code)
                    logging.info(
                        f'Loading search space of epoch {best_epoch} (accuarcy={best_accuracy}) for fine tuning\n')
                    self.current_search_space = best_search_space.to(self.device)
                    self.modified_edge_index = best_edge_index.to(self.device)
                    self.perturbed_edge_weight = best_edge_weight_diff.to(self.device)
                    self.perturbed_edge_weight.requires_grad = True

        #_edge_to_node_transfer = PRBCD.linear_to_triu_idx(self.modified_edge_index)

        #row_idx, col_idx = _edge_to_node_transfer[0], _edge_to_node_transfer[1] TODO: hier könnte man die nodes die noch übrig sind nach dem Angriff evaluaten

        #self.nodes_after_attack = pd.concat([row_idx, col_idx])

        # Retreive best epoch if early stopping is active (not explicitly covered by pesudo code)
        if self.with_early_stopping:
            self.current_search_space = best_search_space.to(self.device)
            self.modified_edge_index = best_edge_index.to(self.device)
            self.perturbed_edge_weight = best_edge_weight_diff.to(self.device)

        # Sample final discrete graph (Algorithm 1, line 16)
        edge_index = self.sample_final_edges(n_perturbations)[0]

        self.adj_adversary = SparseTensor.from_edge_index(
            edge_index,
            torch.ones_like(edge_index[0], dtype=torch.float32),
            (self.n, self.n)
        ).coalesce().detach()
        self.attr_adversary = self.attr

        # TODO: Don't we want to switch to returning things? Haha yeah me too

    def _get_logits(self, x, edge_index, edge_weight):
        return self.attacked_model(
            data=x.to(self.device),
            adj=(edge_index.to(self.device), edge_weight.to(self.device))
        )

    def _get_logits_pre_gnn(self, x, edge_index, edge_weight):
        new_edge_index = self.pre_gnn(x, edge_index, edge_weight)
        return self.attacked_model(
            data=x.to(self.device),
            adj=(new_edge_index.to(self.device),
                 torch.ones(new_edge_index.shape[1], device=self.device))
        )

    @torch.no_grad()
    def sample_final_edges(self, n_perturbations: int) -> Tuple[torch.Tensor, torch.Tensor]:
        best_accuracy = float('Inf')
        perturbed_edge_weight = self.perturbed_edge_weight.detach()
        # TODO: potentially convert to assert
        perturbed_edge_weight[perturbed_edge_weight <= self.eps] = 0

        for i in range(self.max_final_samples):
            if best_accuracy == float('Inf'):
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges = torch.zeros_like(perturbed_edge_weight)
                sampled_edges[torch.topk(perturbed_edge_weight, n_perturbations).indices] = 1
            else:
                sampled_edges = torch.bernoulli(perturbed_edge_weight).float()

            if sampled_edges.sum() > n_perturbations:
                n_samples = sampled_edges.sum()
                logging.info(f'{i}-th sampling: too many samples {n_samples}')
                continue
            self.perturbed_edge_weight = sampled_edges

            edge_index, edge_weight = self.get_modified_adj()
            logits = self._get_logits(self.attr, edge_index, edge_weight)
            accuracy = utils.accuracy(logits, self.labels, self.idx_attack)

            # Save best sample
            if best_accuracy > accuracy:
                best_accuracy = accuracy
                best_edges = self.perturbed_edge_weight.clone().cpu()

        # Recover best sample
        self.perturbed_edge_weight.data.copy_(best_edges.to(self.device))

        edge_index, edge_weight = self.get_modified_adj()
        edge_mask = edge_weight == 1

        allowed_perturbations = 2 * n_perturbations if self.make_undirected else n_perturbations
        edges_after_attack = edge_mask.sum()
        clean_edges = self.edge_index.shape[1]
        assert (edges_after_attack >= clean_edges - allowed_perturbations
                and edges_after_attack <= clean_edges + allowed_perturbations), \
            f'{edges_after_attack} out of range with {clean_edges} clean edges and {n_perturbations} pertutbations'
        return edge_index[:, edge_mask], edge_weight[edge_mask]

    def get_modified_adj(self):
        # --- Fallback: no perturbations yet ---
        if getattr(self, "perturbed_edge_weight", None) is None:
            edge_index = self.edge_index.to(self.device)
            edge_weight = (
                self.edge_weight.to(self.device).float()
                if getattr(self, "edge_weight", None) is not None
                else torch.ones(edge_index.size(1), device=self.device)
            )
            return edge_index, edge_weight

        # --- Original logic below ---
        if (
                not self.perturbed_edge_weight.requires_grad
                or not hasattr(self.attacked_model, 'do_checkpoint')
                or not self.attacked_model.do_checkpoint
        ):
            if self.make_undirected:
                modified_edge_index, modified_edge_weight = utils.to_symmetric(
                    self.modified_edge_index, self.perturbed_edge_weight, self.n
                )
            else:
                modified_edge_index, modified_edge_weight = (
                    self.modified_edge_index,
                    self.perturbed_edge_weight,
                )
            edge_index = torch.cat((self.edge_index.to(self.device), modified_edge_index), dim=-1)
            edge_weight = torch.cat((self.edge_weight.to(self.device), modified_edge_weight))

            edge_index, edge_weight = torch_sparse.coalesce(
                edge_index, edge_weight, m=self.n, n=self.n, op='sum'
            )
        else:
            # TODO: test with pytorch 1.9.0
            # Currently (1.6.0) PyTorch does not support return arguments of `checkpoint` that do not require gradient.
            # For this reason we need this extra code and to execute it twice (due to checkpointing in fact 3 times...)
            from torch.utils import checkpoint

            def fuse_edges_run(perturbed_edge_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                if self.make_undirected:
                    modified_edge_index, modified_edge_weight = utils.to_symmetric(
                        self.modified_edge_index, perturbed_edge_weight, self.n
                    )
                else:
                    modified_edge_index, modified_edge_weight = (
                        self.modified_edge_index,
                        self.perturbed_edge_weight,
                    )
                edge_index = torch.cat((self.edge_index.to(self.device), modified_edge_index), dim=-1)
                edge_weight = torch.cat((self.edge_weight.to(self.device), modified_edge_weight))

                edge_index, edge_weight = torch_sparse.coalesce(
                    edge_index, edge_weight, m=self.n, n=self.n, op='sum'
                )
                return edge_index, edge_weight

            # Hack: for very large graphs the block needs to be added on CPU to save memory
            if len(self.edge_weight) > 100_000_000:
                device = self.device
                self.device = 'cpu'
                self.modified_edge_index = self.modified_edge_index.to(self.device)
                edge_index, edge_weight = fuse_edges_run(self.perturbed_edge_weight.cpu())
                self.device = device
                self.modified_edge_index = self.modified_edge_index.to(self.device)
                return edge_index.to(self.device), edge_weight.to(self.device)

            with torch.no_grad():
                edge_index = fuse_edges_run(self.perturbed_edge_weight)[0]

            edge_weight = checkpoint.checkpoint(
                lambda *input: fuse_edges_run(*input)[1],
                self.perturbed_edge_weight,
            )

        # Allow removal of edges
        edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]

        return edge_index, edge_weight

    def update_edge_weights(self, n_perturbations: int, epoch: int,
                            gradient: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates the edge weights and adaptively, heuristically refined the learning rate such that (1) it is
        independent of the number of perturbations (assuming an undirected adjacency matrix) and (2) to decay learning
        rate during fine-tuning (i.e. fixed search space).

        Parameters
        ----------
        n_perturbations : int
            Number of perturbations.
        epoch : int
            Number of epochs until fine tuning.
        gradient : torch.Tensor
            The current gradient.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated edge indices and weights.
        """
        lr_factor = n_perturbations / self.n / 2 * self.lr_factor
        lr = lr_factor / np.sqrt(max(0, epoch - self.epochs_resampling) + 1)
        self.perturbed_edge_weight.data.add_(lr * gradient)

        # We require for technical reasons that all edges in the block have at least a small positive value
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = self.eps

        return self.get_modified_adj()

    def sample_random_block(self, n_perturbations: int = 0):
        for _ in range(self.max_final_samples):
            self.current_search_space = torch.randint(
                self.n_possible_edges, (self.block_size,), device=self.device)
            self.current_search_space = torch.unique(self.current_search_space, sorted=True)
            if self.make_undirected:
                self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = PRBCD.linear_to_full_idx(self.n, self.current_search_space)
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            if self.current_search_space.size(0) >= n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    def sample_block_from_certificates_radii(self, grid_radii, n_perturbations: int = 0):
        for _ in range(self.max_final_samples):
            # Tried different grid_cells to use for. [1,1] and [2,2] showed best results (determined with 5 examples each)
            if self.use_cert in ("sampling_grid_radii_alt_11",):
                print(self.use_cert, "run sampling_grid_radii_alt_11")
                self.current_node_search_space = np.where(grid_radii[:, 1, 1] == False)[0]
            else:
                print(self.use_cert, "run sampling_grid_radii")
                self.current_node_search_space = np.where(grid_radii[:, 2, 2] == False)[0]
            # draw edges: draw nodes from current_node_search_space and concatenate
            # TODO: The following lines till setup_search_space_undirected could be improved in terms of readability, change method outputs
            if not self.semi:
                edges_idx = self.build_full_idx_matrix(False, self.block_size)
            else:
                self.build_full_idx_matrix_semi(False, self.block_size)
            # bis hierhin wird für die gesamte Blockmatrix gezogen

            if self.make_undirected:
                # make undirected: cut all (x,y) where x >= y
                self.current_search_space = self.edges_to_current_search_space(self.n)
                self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)

                # self.setup_search_space_undirected(self.n) with the two lines above this method is NOT needed anymore
                # TODO: This will cut arbitrary number of entries, I made a function draw_undirected_matrix() to bypass this, but may be slow
                # maybe return later to this idea
            else:
                # TODO: i have not checked if it works for the directed case, I think it will NOT work
                self.modified_edge_index = PRBCD.cut_diagonal_entries(edges_idx)

            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            if self.current_search_space.size(0) >= n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    def sample_block_from_certificates_binary_class(self, grid_binary_class, n_perturbations: int = 0):
        for _ in range(self.max_final_samples):

            if self.use_cert in ("sampling_grid_binary_class_alt_11", "sampling_grid_binary_class_alt_22"):
                print("using alternative current_node_search_space sampling")
                self.sample_current_node_search_space_det(grid_binary_class)
            else:
                sums = np.sum(grid_binary_class, axis=(1, 2))  # shape: (2810,)
                smallest_indices = np.argsort(sums)[:len(sums) // 2]
                self.current_node_search_space = torch.from_numpy(smallest_indices)

            # draw edges: draw nodes from current_node_search_space and concatenate
            if not self.semi:
                edges_idx = self.build_full_idx_matrix(False, self.block_size)
            else:
                self.build_full_idx_matrix_semi(False, self.block_size)
            # bis hierhin wird für die gesamte Blockmatrix gezogen

            if self.make_undirected:
                # make undirected: cut all (x,y) where x >= y
                self.current_search_space = self.edges_to_current_search_space(self.n)
                self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
                # self.setup_search_space_undirected(self.n) this method is not needed anymore
            else:
                # TODO: i have not checked if it works for the directed case, I think it will NOT work
                self.modified_edge_index = PRBCD.cut_diagonal_entries(edges_idx)

            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            if self.current_search_space.size(0) >= n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    def sample_block_from_prior_gumbel(self, n_perturbations: int = 0, tau: float = 1.0):
        """
        Build the initial PR-BCD block by sampling 'block_size' distinct edges
        from the selector's learned prior with straight-through Gumbel-Softmax.
        Also record log-probs for policy-gradient updates.
        """
        # sample_block_from_prior_gumbel / resample_block_from_prior_gumbel

        logits = self.pre_gnn_prior(self.attr, self.edge_index, self.edge_weight)  # (E,)
        logits = logits.to(self.device).float()  # <- ensure float dtype

        k = min(self.block_size, logits.numel())
        selected_idx, logps = [], []
        mask = torch.zeros_like(logits)

        for _ in range(k):
            # p = softmax(logits + mask); y ~ ST GumbelSoftmax(p)
            y = F.gumbel_softmax(logits + mask, tau=tau, hard=True, dim=0)  # (E,)
            j = int(y.argmax())
            selected_idx.append(j)

            # log p_j for PG:
            logp = F.log_softmax(logits + mask, dim=0)[j]
            logps.append(logp)

            mask[j] = float('-inf')  # forbid duplicates

        sel = torch.tensor(selected_idx, device=self.device, dtype=torch.long)

        # build PR-BCD search block (indices are into self.edge_index)
        self.current_search_space = sel
        self.modified_edge_index = self.edge_index[:, sel].to(self.device)
        self.perturbed_edge_weight = torch.full(
            (sel.numel(),), self.eps, dtype=torch.float32, requires_grad=True, device=self.device
        )

        # keep the selector logprobs for PG step this epoch
        self._selector_log_probs = torch.stack(logps)  # (k,)

        if self.current_search_space.size(0) < n_perturbations:
            raise RuntimeError("Not enough edges sampled for the requested budget.")

    def _node_saliency_from_victim(self,
                                   subset: str = "attack",
                                   loss_type: str = "ce") -> torch.Tensor:
        """
        Compute per-node saliency from the victim's classification loss via a
        node-level gate. Returns a length-N tensor of non-negative saliencies.

        subset: "attack" (= self.idx_attack), "all", or a sequence/tensor of node indices.
        """
        # Put victim in eval mode; this does NOT disable grads.
        self.attacked_model.eval()

        # ---- helpers ---------------------------------------------------------
        def _as_long_idx(idx_like):
            if torch.is_tensor(idx_like):
                return idx_like.to(self.device, dtype=torch.long)
            return torch.as_tensor(idx_like, device=self.device, dtype=torch.long)

        def _as_labels(y_like):
            if torch.is_tensor(y_like):
                return y_like.to(self.device, dtype=torch.long)
            return torch.as_tensor(y_like, device=self.device, dtype=torch.long)

        # Choose which nodes contribute to the loss
        if subset == "attack":
            idx = _as_long_idx(self.idx_attack)
        elif subset == "all":
            idx = torch.arange(self.n, device=self.device, dtype=torch.long)
        else:
            idx = _as_long_idx(subset)

        # Handle empty selection gracefully
        if idx.numel() == 0:
            return torch.zeros(self.n, device=self.device)

        # ---- data on the correct device/dtypes -------------------------------
        x = self.attr.to(self.device).float()  # (N, F)

        # IMPORTANT: use the *current* (possibly perturbed) adjacency if available
        if hasattr(self, "get_modified_adj"):
            ei, ew = self.get_modified_adj()
            ei = ei.to(self.device)
            ew = ew.to(self.device).float()
        else:
            ei = self.edge_index.to(self.device)
            ew = (self.edge_weight.to(self.device).float()
                  if getattr(self, "edge_weight", None) is not None
                  else torch.ones(ei.size(1), device=self.device))

        y = _as_labels(self.labels)

        # ---- compute logits with grad enabled -------------------------------
        with torch.enable_grad():
            # node gates (one scalar per node) to measure sensitivity
            g = torch.zeros(self.n, device=self.device).requires_grad_(True)  # (N,)
            x_gated = x * (1.0 + g.unsqueeze(1))  # broadcast to (N, F)

            logits = self.attacked_model(data=x_gated, adj=(ei, ew))  # (N, C)

            if loss_type == "tanhMargin":
                # margin = logit_true - max_{k != true} logit_k ; we *minimize* margin
                logits_sel = logits[idx]  # (M, C)
                y_sel = y[idx]  # (M,)
                true_logit = logits_sel.gather(1, y_sel.view(-1, 1)).squeeze(1)  # (M,)
                masked = logits_sel.clone()
                masked[torch.arange(masked.size(0), device=masked.device), y_sel] = -1e30
                max_other = masked.max(dim=1).values
                margin = true_logit - max_other
                loss = torch.tanh(margin).mean().neg()  # -mean(tanh(margin))
            else:
                # default: cross-entropy over the chosen subset
                loss = F.cross_entropy(logits[idx], y[idx])

            # gradient of loss wrt node gates
            (grad_g,) = torch.autograd.grad(loss, g, retain_graph=False, create_graph=False)

        saliency = grad_g.abs().detach()  # (N,)
        return saliency

    def _gumbel_topk(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        Gumbel-top-k without replacement on a 1D logits tensor.
        Returns indices of the k sampled items (k clipped to len).
        """
        k = int(min(k, logits.numel()))
        if k <= 0 or logits.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=logits.device)
        # Add i.i.d. Gumbel noise: g = -log(-log(U))
        g = -torch.log(-torch.log(torch.clamp(torch.rand_like(logits), 1e-12, 1. - 1e-12)))
        return torch.topk(logits + g, k=k, largest=True).indices

    import torch
    import torch.nn.functional as F

    def _dense_from_edge_index(self):
        N = int(self.n)
        A = torch.zeros((N, N), dtype=torch.float32, device=self.device)
        ei = self.edge_index.to(self.device)
        A[ei[0].long(), ei[1].long()] = 1.0
        A[ei[1].long(), ei[0].long()] = 1.0
        A.fill_diagonal_(0.0)
        return A

    def _normalize_dense(self, A):
        N = A.size(0)
        A_tilde = A + torch.eye(N, dtype=A.dtype, device=A.device)
        deg = A_tilde.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)
        D = torch.diag(deg_inv_sqrt)
        return D @ A_tilde @ D

    def make_pgd_forward_from_victim(self):
        # attach a function that maps (X, Abar_dense) -> logits via attacked_model
        def fwd(X, Abar):
            # convert dense normalized adjacency to sparse edge_index/edge_weight
            idx = Abar.nonzero(as_tuple=False).t()  # (2, E)
            w = Abar[idx[0], idx[1]]  # (E,)
            return self.attacked_model(data=X, adj=(idx, w))

        self.pgd_forward = fwd

    def _pgd_node_labels(self, loss_nodes=None, loss_type="ce", kappa=0.0):
        import torch
        import torch.nn.functional as F

        with torch.enable_grad():
            N = int(self.n)
            X = self.attr.to(self.device).float()
            y = torch.as_tensor(self.labels, device=self.device, dtype=torch.long)

            if loss_nodes is None:
                idx = (torch.as_tensor(self.idx_attack, device=self.device, dtype=torch.long)
                       if getattr(self, "idx_attack", None) is not None
                       else torch.arange(N, device=self.device, dtype=torch.long))
            else:
                idx = torch.as_tensor(loss_nodes, device=self.device, dtype=torch.long)

            # >>> use *current* graph, not the clean one
            ei, ew = self.get_modified_adj()
            ei = ei.to(self.device)
            ew = ew.to(self.device).float()

            g = torch.zeros(N, device=self.device, requires_grad=True)
            Xg = X * (1.0 + g.unsqueeze(1))

            logits = self.attacked_model(data=Xg, adj=(ei, ew))

            if loss_type == "ce":
                loss = F.cross_entropy(logits[idx], y[idx])
            else:
                Z = logits[idx];
                yy = y[idx]
                true = Z[torch.arange(Z.size(0), device=Z.device), yy]
                Zm = Z.clone();
                Zm[torch.arange(Z.size(0), device=Z.device), yy] = -1e9
                other = Zm.max(dim=1).values
                margin = true - other
                if kappa:
                    margin = torch.maximum(margin, torch.tensor(-float(kappa), device=Z.device))
                loss = -margin.mean()

            (grad_g,) = torch.autograd.grad(loss, g, retain_graph=False, create_graph=False)
            s = grad_g.abs()
            return (s - s.mean()) / (s.std() + 1e-6)

    def _pretrain_node_scorer_from_pgd(self, epochs=20, lr=1e-3, wd=5e-4,
                                       loss_nodes=None, loss_type="ce", kappa=0.0, log_every=5):
        """
        Distill PGD-style node saliency into your NodeBlockScorer:
          target: s_i (sum_j |dL/dA_ij|)
          pred:   selector.node_head(selector.embed(...))
        """
        assert hasattr(self, "selector") and (self.selector is not None), \
            "self.selector (NodeBlockScorer) is required."

        self.selector.train()
        X = self.attr.to(self.device).float()
        ei = self.edge_index.to(self.device)
        ew = (self.edge_weight.to(self.device).float() if getattr(self, "edge_weight", None) is not None else None)

        # labels from PGD gradient
        y_node = self._pgd_node_labels(loss_type=loss_type, kappa=kappa)  # (N,)

        opt = torch.optim.Adam(self.selector.parameters(), lr=lr, weight_decay=wd)

        for e in range(1, epochs + 1):
            opt.zero_grad()
            h = self.selector.embed(X, ei, ew)  # (N,d)
            pred = self.selector.node_head(h).squeeze(-1)  # (N,)
            loss = F.smooth_l1_loss(pred, y_node)
            loss.backward()
            opt.step()

            if (e % max(1, log_every) == 0) or e in (1, epochs):
                # you can swap to logging if you prefer
                print(f"[PGD-NodePretrain] epoch {e}/{epochs}  loss={loss.item():.4f}")

        self.selector.eval()
        # flag for downstream checks
        self._pgd_node_trained = True

    def sample_block_from_pgdtopk(self,
                                  k_nodes: int = 350,
                                  prefer_edges: str = "intra",  # 'intra'|'incident'|'mix'
                                  loss_nodes=None,
                                  loss_type="ce",
                                  kappa: float = 0.0,
                                  pretrain_epochs: int = 20,
                                  pretrain_lr: float = 1e-3,
                                  pretrain_wd: float = 5e-4,
                                  max_intra_ratio: float = 0.8):
        """
        1) Pretrains your node scorer on PGD-style node labels.
        2) Extracts Top-K nodes by the trained scorer.
        3) (Optional) Forms a PR-BCD block from those nodes.

        Sets:
          self.current_search_space (linear upper-tri idx)
          self.modified_edge_index  (2, K) upper-tri edges
          self.perturbed_edge_weight (K,)
          self.topk_nodes (for inspection)
        """
        # --- (1) pretrain on PGD node labels ---
        self._pretrain_node_scorer_from_pgd(
            epochs=pretrain_epochs, lr=pretrain_lr, wd=pretrain_wd,
            loss_nodes=loss_nodes, loss_type=loss_type, kappa=kappa
        )

        # --- (2) score nodes and select Top-K ---
        X = self.attr.to(self.device).float()
        ei = self.edge_index.to(self.device)
        ew = (self.edge_weight.to(self.device).float() if getattr(self, "edge_weight", None) is not None else None)

        with torch.no_grad():
            h = self.selector.embed(X, ei, ew)
            node_scores = self.selector.node_head(h).squeeze(-1)
            # z-score (optional)
            node_scores = (node_scores - node_scores.mean()) / (node_scores.std() + 1e-6)
            k = int(min(max(2, k_nodes), self.n))
            topk = torch.topk(node_scores, k=k, largest=True).indices.to(self.device)

        self.topk_nodes = topk

        # --- (3) build a block from Top-K nodes (upper-tri pairs) ---
        import math
        N = int(self.n)
        # candidates within topk
        if topk.numel() >= 2:
            comb = torch.combinations(topk, r=2, with_replacement=False)  # (C(k,2), 2)
            intra_u, intra_v = comb[:, 0], comb[:, 1]
        else:
            intra_u = intra_v = torch.empty(0, dtype=torch.long, device=self.device)

        # incident candidates: connect topk to outside
        all_idx = torch.arange(N, device=self.device)
        is_top = torch.zeros(N, dtype=torch.bool, device=self.device);
        is_top[topk] = True
        outside = all_idx[~is_top]

        # score edges by node scores (sum rule)
        def score_pairs(U, V):
            if U.numel() == 0:
                return torch.empty(0, device=self.device)
            return node_scores[U] + node_scores[V]

        intra_scores = score_pairs(intra_u, intra_v)

        # incident pool (rough sampling to keep it light)
        target_block = int(self.block_size)
        intra_quota = int(round(max_intra_ratio * target_block)) if prefer_edges in ("intra", "mix") else 0
        remaining = max(0, target_block - intra_quota)

        inc_u_list, inc_v_list, inc_s_list = [], [], []
        if prefer_edges in ("incident", "mix") and outside.numel() > 0:
            per_u = max(1, math.ceil(remaining / max(1, topk.numel())))
            for u in topk:
                # sample a small set of partners
                if outside.numel() <= per_u:
                    v = outside
                else:
                    sel = torch.randperm(outside.numel(), device=self.device)[:per_u]
                    v = outside[sel]
                if v.numel() == 0:
                    continue
                inc_u_list.append(u.repeat(v.numel()))
                inc_v_list.append(v)
                inc_s_list.append(score_pairs(u.repeat(v.numel()), v))
        if len(inc_u_list) > 0:
            inc_u = torch.cat(inc_u_list);
            inc_v = torch.cat(inc_v_list);
            inc_scores = torch.cat(inc_s_list)
            # unique
            lin = (torch.minimum(inc_u, inc_v) * N + torch.maximum(inc_u, inc_v))
            uniq_lin, uniq_idx = torch.unique(lin, sorted=False, return_inverse=False, return_counts=False,
                                              return_indices=True)
            inc_u, inc_v, inc_scores = inc_u[uniq_idx], inc_v[uniq_idx], inc_scores[uniq_idx]
        else:
            inc_u = inc_v = torch.empty(0, dtype=torch.long, device=self.device)
            inc_scores = torch.empty(0, device=self.device)

        chosen_u, chosen_v = [], []

        # take intra edges first
        if intra_u.numel() > 0 and intra_quota > 0:
            take = min(intra_quota, intra_u.numel())
            top = torch.topk(intra_scores, k=take, largest=True).indices
            chosen_u.append(intra_u[top]);
            chosen_v.append(intra_v[top])

        # then incident edges to fill the rest
        need = target_block - (0 if len(chosen_u) == 0 else chosen_u[-1].numel())
        if need > 0 and inc_u.numel() > 0:
            take = min(need, inc_u.numel())
            top = torch.topk(inc_scores, k=take, largest=True).indices
            chosen_u.append(inc_u[top]);
            chosen_v.append(inc_v[top])

        if len(chosen_u) == 0:
            # fallback: random upper-tri picks
            tri_u, tri_v = torch.triu_indices(N, N, offset=1, device=self.device)
            perm = torch.randperm(tri_u.numel(), device=self.device)[:target_block]
            final_u, final_v = tri_u[perm], tri_v[perm]
        else:
            final_u = torch.cat(chosen_u);
            final_v = torch.cat(chosen_v)
            # pad if short
            if final_u.numel() < target_block:
                tri_u, tri_v = torch.triu_indices(N, N, offset=1, device=self.device)
                rest = target_block - final_u.numel()
                perm = torch.randperm(tri_u.numel(), device=self.device)[:rest]
                final_u = torch.cat([final_u, tri_u[perm]])
                final_v = torch.cat([final_v, tri_v[perm]])

        # ensure upper-tri (i<j)
        uu = torch.minimum(final_u, final_v)
        vv = torch.maximum(final_u, final_v)
        mask = uu < vv
        uu, vv = uu[mask], vv[mask]

        full_idx = torch.stack([uu, vv], dim=0)  # (2, K')

        # convert to PRBCD's linear indexing
        lin_idx = PRBCD.triu_idx_to_linear_idx(self.n, full_idx)

        self.current_search_space = lin_idx
        self.modified_edge_index = full_idx
        self.perturbed_edge_weight = torch.full(
            (self.current_search_space.numel(),),
            self.eps, dtype=torch.float32, device=self.device, requires_grad=True
        )

    def sample_block_from_linkpred_gcn(self, graph):
        import numpy as np
        import torch
        import torch.optim as optim
        from torch.nn import BCEWithLogitsLoss
        from torch_geometric.data import Data

        device = getattr(self, "device", torch.device("cpu"))

        # ---------- Build PyG Data from provided graph (SciPy -> torch) ----------
        # Features (dense float32)
        X_coo = graph.attr_matrix.tocoo()
        X_idx = torch.tensor(np.vstack([X_coo.row, X_coo.col]), dtype=torch.long)
        X_val = torch.tensor(X_coo.data, dtype=torch.float32)
        X = torch.sparse_coo_tensor(X_idx, X_val, size=graph.attr_matrix.shape).coalesce().to_dense()

        # Adjacency -> undirected edge_index (only once per undirected edge; upper-tri)
        A_coo = graph.adj_matrix.tocoo()
        ei_full = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long)
        # keep (u<v) to get the undirected set without duplicates/self-loops
        mask_upper = ei_full[0] < ei_full[1]
        ei_upper = ei_full[:, mask_upper]  # (2, E_u)

        # for message passing we want both directions of the TRAIN edges:
        def make_bidir(ei):
            return torch.cat([ei, ei.flip(0)], dim=1)  # (2, 2*E_subset)

        data = Data(x=X, edge_index=None)  # will set train_ei later
        data = data.to(device)
        N = data.num_nodes
        F = data.num_features

        # ---------- Manual split on undirected edges ----------
        E_u = ei_upper.size(1)
        perm = torch.randperm(E_u)
        # 5% val, 10% test (same as before)
        n_val = max(1, int(0.05 * E_u))
        n_test = max(1, int(0.10 * E_u))
        n_train = E_u - n_val - n_test
        idx_train = perm[:n_train]
        idx_val = perm[n_train:n_train + n_val]
        idx_test = perm[n_train + n_val:]

        train_pos_u = ei_upper[:, idx_train]  # (2, Etr_u)
        val_pos_u = ei_upper[:, idx_val]
        test_pos_u = ei_upper[:, idx_test]

        # Edge index used for message passing during training = bidirectional train edges
        train_edge_index = make_bidir(train_pos_u).to(device)

        # ---------- Model / opt ----------
        model = GCNLinkPredictor(in_feats=F, hidden_feats=64, out_feats=32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = BCEWithLogitsLoss()

        # ---------- Train link predictor ----------
        from torch_geometric.utils import negative_sampling
        model.train()
        rolling_sum, window = 0.0, 10

        x = data.x.to(device)
        for epoch in range(1, 201):
            # negatives: same count as positive train edges (bidirectional count is double, so use undirected count)
            neg_edge_index = negative_sampling(
                edge_index=train_edge_index,
                num_nodes=N,
                num_neg_samples=train_pos_u.size(1) * 2  # match bidir positives
            )

            pos_edge_index = train_edge_index  # (2, 2*Etr_u)
            combined_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

            pos_labels = torch.ones(pos_edge_index.size(1), device=device, dtype=torch.float32)
            neg_labels = torch.zeros(neg_edge_index.size(1), device=device, dtype=torch.float32)
            combined_labels = torch.cat([pos_labels, neg_labels], dim=0)

            # shuffle
            perm_edges = torch.randperm(combined_edge_index.size(1), device=device)
            combined_edge_index = combined_edge_index[:, perm_edges]
            combined_labels = combined_labels[perm_edges]

            optimizer.zero_grad()
            score_logits = model(x, pos_edge_index, combined_edge_index)
            loss = loss_fn(score_logits.view_as(combined_labels), combined_labels)
            loss.backward()
            optimizer.step()

            rolling_sum += loss.item()
            if epoch % window == 0 or epoch == 1:
                avg = rolling_sum / (window if epoch % window == 0 else 1)
                print(f"Epoch {epoch:03d} | loss: {loss.item():.4f} | avg{window}: {avg:.4f}")
                if epoch % window == 0:
                    rolling_sum = 0.0

        # ---------- Score ALL unordered pairs by entropy ----------
        model.eval()
        with torch.no_grad():
            # all u<v pairs exactly once
            all_pairs = torch.combinations(torch.arange(N, device=device), r=2)  # (M, 2)
            all_edge_index = all_pairs.t().contiguous()  # (2, M)
            edge_logits_all = model(x, train_edge_index, all_edge_index)  # (M,)
            p = torch.sigmoid(edge_logits_all)
            eps = 1e-12
            edge_uncertainty_all = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))  # (M,)

        # ---------- Select top-K & store for PR-BCD ----------
        K = min(50_000, all_edge_index.size(1))
        topk_idx = torch.topk(edge_uncertainty_all, k=K, largest=True).indices

        block_pairs = all_edge_index[:, topk_idx].detach().cpu()  # (2, K)
        block_lin = PRBCD.triu_idx_to_linear_idx(N, block_pairs)  # (K,), CPU

        # IMPORTANT: match PR-BCD expectations
        self.current_search_space = block_lin.detach().to(device)  # (K,)
        self.modified_edge_index = block_pairs.detach().to(device)  # (2, K)
        self.perturbed_edge_weight = torch.full(
            (block_lin.numel(),),
            getattr(self, "eps", 1e-7),
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )

        print(f"Built ALL-PAIRS uncertainty block with {block_lin.numel()} edges.")

    def sample_block_from_linkpred_gnn_all_pairs(self, graph, b):

        # Suppose you already have: graph.attr_matrix (N x d, scipy COO/CSR) and graph.adj_matrix (N x N)
        N, d = graph.attr_matrix.shape

        # Convert attributes to torch sparse COO
        X_coo = graph.attr_matrix.tocoo()
        X_idx = torch.tensor(np.vstack([X_coo.row, X_coo.col]), dtype=torch.long)
        X_val = torch.tensor(X_coo.data, dtype=torch.float32)
        X_sparse = torch.sparse_coo_tensor(X_idx, X_val, size=(N, d)).coalesce()

        # Convert adjacency to torch sparse COO (ensure symmetric for undirected graphs)
        A_coo = graph.adj_matrix.tocoo()
        A_idx = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long)
        A_val = torch.tensor(A_coo.data, dtype=torch.float32)
        A_sparse = torch.sparse_coo_tensor(A_idx, A_val, size=(N, N)).coalesce()

        # Optional modified edges M as sparse COO (or set to None)

        N, d = X_sparse.size()

        model = sall.AllPairsLinkPredictor(in_dim=d, hidden=64, out_dim=64, dropout=0.1).to(X_sparse.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

        # train a few epochs on ALL pairs
        for ep in range(10):
            opt.zero_grad(set_to_none=True)
            loss = model.loss_all_pairs(X_sparse, A_sparse)
            loss.backward()
            opt.step()
            print(f"epoch {ep + 1:02d} | loss {loss.item():.4f}")

        # inference: top-20 pairs overall
        model.eval()
        pairs, probs = model.bottomk_pairs(X_sparse, A_sparse, k=b)
        print("Top pairs:", pairs[:, :5].T.tolist(), "…")
        print(pairs.shape)
        print("Probs:", probs[:5].tolist(), "…")

        # --- 4) Map to linear index space and trim/unique like sample_random_block ---
        # edges_idx are already u < v (upper-triangle). Convert to linear upper-tri indices.
        lin = self.pairs_to_linear_uppertri(pairs.to(self.device), N)  # (b_sel,)
        # keep unique + sorted (should already be unique, but mirror random_block behavior)
        lin = torch.unique(lin, sorted=True)

        # Cap to self.block_size to honor the contract
        if lin.numel() > self.block_size:
            # because edges came sorted by ascending prob, keep front part
            lin = lin[: self.block_size]

        self.current_search_space = lin  # (m,)

        # --- 5) Set modified_edge_index exactly like sample_random_block ---
        if self.make_undirected:
            self.modified_edge_index = PRBCD.linear_to_triu_idx(N, self.current_search_space)
        else:
            self.modified_edge_index = PRBCD.linear_to_full_idx(N, self.current_search_space)
            # drop self-loops in the directed/full case
            is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
            if not torch.all(is_not_self_loop):
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

        # --- 6) Initialize perturbed_edge_weight like sample_random_block ---
        self.perturbed_edge_weight = torch.full(
            (self.current_search_space.size(0),),
            self.eps, dtype=torch.float32, device=self.device, requires_grad=True
        )

    def sample_block_from_margin_loss_gcn(self, graph, alpha_Sigmoid_APert=0):
        import numpy as np
        import torch

        device = getattr(self, "device", torch.device("cpu"))

        # ----- 1) Pull graph from argument (SciPy -> torch) -----
        # X: (N, F)
        X_coo = graph.attr_matrix.tocoo()
        X_idx = torch.tensor(np.vstack([X_coo.row, X_coo.col]), dtype=torch.long, device=device)
        X_val = torch.tensor(X_coo.data, dtype=torch.float32, device=device)
        X_sparse = torch.sparse_coo_tensor(X_idx, X_val, size=graph.attr_matrix.shape, device=device).coalesce()
        X = X_sparse.to_dense()

        # A_base: (N, N) dense, symmetric, binary
        A_coo = graph.adj_matrix.tocoo()
        A_idx = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long, device=device)
        A_val = torch.tensor(A_coo.data, dtype=torch.float32, device=device)
        A_sparse = torch.sparse_coo_tensor(A_idx, A_val, size=graph.adj_matrix.shape, device=device).coalesce()
        A_base = A_sparse.to_dense()
        A_base = (A_base + A_base.t()).clamp(max=1.0)

        N, F = X.shape

        # classes (try to infer, else default)
        if hasattr(graph, "labels") and graph.labels is not None:
            num_classes = int(np.max(graph.labels)) + 1
        else:
            num_classes = 7  # safe default for Cora-like graphs

        # ----- 2) Victim surrogate (TinyGCN) + label-free masked margin -----
        gcn = TinyGCN(in_feats=F, hidden=64, out_feats=num_classes).to(device)
        gcn.eval()

        # ----- 3) Relaxed p over ALL undirected pairs (u < v) -----
        iu, ju = torch.triu_indices(N, N, offset=1, device=device)  # all pairs once
        M = iu.numel()

        # sign: +1 insertion if no edge; -1 deletion if edge exists
        is_existing = (A_base[iu, ju] > 0)
        sign = torch.where(is_existing,
                           torch.tensor(-1.0, device=device),
                           torch.tensor(+1.0, device=device))  # (M,)

        p = torch.zeros(M, device=device, requires_grad=True)

        if alpha_Sigmoid_APert != 0:
            # Smooth mapping from p → perturbation strength in [0, 1]
            pert_strength = torch.sigmoid(alpha_Sigmoid_APert * p)  # (M,)

            # Build symmetric perturbation matrix
            Pmat = torch.zeros((N, N), device=device, dtype=torch.float32)
            Pmat[iu, ju] = sign * pert_strength
            Pmat[ju, iu] = sign * pert_strength
        else:
            Pmat = torch.zeros((N, N), device=device, dtype=torch.float32)
            Pmat[iu, ju] = sign * p
            Pmat[ju, iu] = sign * p

        A_pert = A_base + Pmat

        # ----- 4) Forward/backward to get |∂L'/∂p| -----
        logits = gcn(X, A_pert)
        loss = tanh_margin_loss_label_free(logits)
        gcn.zero_grad(set_to_none=True)
        loss.backward()  # populates p.grad

        scores = p.grad.abs().detach()  # (M,)

        # ----- 5) Top-K edges & store for PR-BCD -----
        # use self.block_size if present, else cap at 50k
        K_cap = 50_000
        K = min(getattr(self, "block_size", K_cap), K_cap, M)
        topk = torch.topk(scores, k=K, largest=True)
        chosen = topk.indices  # indices into iu/ju

        block_pairs = torch.stack([iu[chosen].cpu(), ju[chosen].cpu()], dim=0).contiguous()  # (2, K)
        block_lin = PRBCD.triu_idx_to_linear_idx(N, block_pairs)  # (K,), CPU

        self.current_search_space = block_lin.detach().to(device)  # (K,)
        self.modified_edge_index = block_pairs.detach().to(device)  # (2, K)
        self.perturbed_edge_weight = torch.full(
            (block_lin.numel(),),
            getattr(self, "eps", 1e-7),
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )

        print(f"Built p-grad block over ALL pairs with {block_lin.numel()} edges (top-{K} by |dL'/dp|).")
        print(loss.item())

    def sample_block_from_linkpred_gnn_old(self, graph, n_perturbations):
        # Suppose you already have: graph.attr_matrix (N x d, scipy COO/CSR) and graph.adj_matrix (N x N)
        N, d = graph.attr_matrix.shape

        # Convert attributes to torch sparse COO
        X_coo = graph.attr_matrix.tocoo()
        X_idx = torch.tensor(np.vstack([X_coo.row, X_coo.col]), dtype=torch.long)
        X_val = torch.tensor(X_coo.data, dtype=torch.float32)
        X_sparse = torch.sparse_coo_tensor(X_idx, X_val, size=(N, d)).coalesce()

        # Convert adjacency to torch sparse COO (ensure symmetric for undirected graphs)
        A_coo = graph.adj_matrix.tocoo()
        A_idx = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long)
        A_val = torch.tensor(A_coo.data, dtype=torch.float32)
        A_sparse = torch.sparse_coo_tensor(A_idx, A_val, size=(N, N)).coalesce()

        # Optional modified edges M as sparse COO (or set to None)

        N, d = X_sparse.size()
        model = lle.GNNLeastLikelyEdges(in_dim=d, hidden=64, out_dim=64, dropout=0.1).to(X_sparse.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        # Train with BCE (positives vs sampled non-edges)
        for ep in range(1, 71):
            model.train()
            loss = model.bce_loss(X_sparse, A_sparse, M_sparse=None, neg_ratio=1, max_pos_per_batch=64000)
            opt.zero_grad();
            loss.backward();
            opt.step()
            if ep % 10 == 0 or ep == 1:
                print(f"Epoch {ep:03d} | loss={float(loss):.6f}")

            # --- 3) Inference: take least-likely existing edges ---
        model.eval()
        # default b: fill a block or use user-specified
        want = self.block_size
        edges_idx, edge_probs, _ = model(X_sparse, A_sparse, M_sparse=None, b=want)  # edges_idx: (2, b_sel) with u<v

        # --- 4) Map to linear index space and trim/unique like sample_random_block ---
        # edges_idx are already u < v (upper-triangle). Convert to linear upper-tri indices.
        lin = self.pairs_to_linear_uppertri(edges_idx.to(self.device), N)  # (b_sel,)
        # keep unique + sorted (should already be unique, but mirror random_block behavior)
        lin = torch.unique(lin, sorted=True)

        # Cap to self.block_size to honor the contract
        if lin.numel() > self.block_size:
            # because edges came sorted by ascending prob, keep front part
            lin = lin[: self.block_size]

        self.current_search_space = lin  # (m,)

        # --- 5) Set modified_edge_index exactly like sample_random_block ---
        if self.make_undirected:
            self.modified_edge_index = PRBCD.linear_to_triu_idx(N, self.current_search_space)
        else:
            self.modified_edge_index = PRBCD.linear_to_full_idx(N, self.current_search_space)
            # drop self-loops in the directed/full case
            is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
            if not torch.all(is_not_self_loop):
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

        # --- 6) Initialize perturbed_edge_weight like sample_random_block ---
        self.perturbed_edge_weight = torch.full(
            (self.current_search_space.size(0),),
            self.eps, dtype=torch.float32, device=self.device, requires_grad=True
        )

        # --- 7) Budget sanity (mirror sample_random_block) ---
        if self.current_search_space.size(0) < n_perturbations:
            raise RuntimeError(
                f"GNN block smaller than n_perturbations: "
                f"{self.current_search_space.size(0)} < {n_perturbations}. "
                f"Try increasing b or lowering the budget."
            )

    def sample_block_from_linkpred_gnn(self, graph, n_perturbations):
        import numpy as np
        import torch
        import torch.nn.functional as F

        # --- 1) Convert inputs to torch sparse ---
        N, d = graph.attr_matrix.shape

        # X (features) -> sparse COO
        X_coo = graph.attr_matrix.tocoo()
        X_idx = torch.tensor(np.vstack([X_coo.row, X_coo.col]), dtype=torch.long, device=self.device)
        X_val = torch.tensor(X_coo.data, dtype=torch.float32, device=self.device)
        X_sparse = torch.sparse_coo_tensor(X_idx, X_val, size=(N, d), device=self.device).coalesce()

        # A (adjacency) -> sparse COO (assume undirected input; if not, symmetrize outside)
        A_coo = graph.adj_matrix.tocoo()
        A_idx = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long, device=self.device)
        A_val = torch.tensor(A_coo.data, dtype=torch.float32, device=self.device)
        A_sparse = torch.sparse_coo_tensor(A_idx, A_val, size=(N, N), device=self.device).coalesce()

        # --- 2) Build & train the LP GNN (same as before) ---
        N, d = X_sparse.size()
        model = lle.GNNLeastLikelyEdges(in_dim=d, hidden=64, out_dim=64, dropout=0.1).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        for ep in range(1, 121):
            model.train()
            loss = model.bce_loss(X_sparse, A_sparse, M_sparse=None, neg_ratio=5, max_pos_per_batch=64000)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ep % 10 == 0 or ep == 1:
                print(f"Epoch {ep:03d} | loss={float(loss):.6f}")

        # --- 3) Inference: use HIGH-LP adds/removes initializer (new logic) ---
        model.eval()
        # total candidates you want in the block
        B = self.block_size

        # NOTE: init_block_highLP_add_remove must be added as a method on the model class (as discussed).
        # It returns (u<v) pairs for both adds (non-edges) and removes (edges).
        edges_idx, probs, is_add_mask = model.init_block_highLP_add_remove(
            X_sparse, A_sparse, M_sparse=None,
            B=B,
            add_ratio=0.5,  # tune split
            p_min=0.6,  # plausibility threshold
            per_node_cap=8,  # diversity
            rank_by="prob",  # or "prob"
            dense_limit=7500,
            block_size=50000
        )  # edges_idx: (2, m), probs: (m,), is_add_mask: (m,)

        # If init pool smaller than B, edges_idx may be shorter. That's fine; PRBCD just gets fewer candidates.

        # --- 4) Map (u,v) to your linear upper-triangle indices (like random) ---
        # edges_idx are already u < v (upper-triangle).
        lin = self.pairs_to_linear_uppertri(edges_idx.to(self.device), N)  # (m,)

        # Keep unique + sorted (mirrors random block behavior)
        lin = torch.unique(lin, sorted=True)

        # Cap to block_size (keep highest-priority first: since we sorted only by index, we reselect by probs if needed)
        if lin.numel() > self.block_size:
            # Re-rank the kept indices by their original priority (probs desc) before truncation:
            # Build a map from linear -> score
            with torch.no_grad():
                # rebuild linear for original ordering
                lin_all = self.pairs_to_linear_uppertri(edges_idx.to(self.device), N)
                # sort by score (desc), then stable-filter to uniques
                order = torch.argsort(probs.to(self.device), descending=True)
                lin_ranked = lin_all[order]
                # keep first occurrences up to block_size
                seen = torch.zeros(lin.max().item() + 1, dtype=torch.bool, device=self.device)
                sel = []
                for i in range(lin_ranked.numel()):
                    li = lin_ranked[i].item()
                    if not seen[li]:
                        sel.append(lin_ranked[i])
                        seen[li] = True
                        if len(sel) >= self.block_size:
                            break
                lin = torch.stack(sel) if sel else lin[: self.block_size]

        self.current_search_space = lin  # (m,)

        # --- 5) Set modified_edge_index like sample_random_block ---
        if self.make_undirected:
            self.modified_edge_index = PRBCD.linear_to_triu_idx(N, self.current_search_space)
        else:
            self.modified_edge_index = PRBCD.linear_to_full_idx(N, self.current_search_space)
            is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
            if not torch.all(is_not_self_loop):
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

        # --- 6) Initialize perturbed_edge_weight like random ---
        self.perturbed_edge_weight = torch.full(
            (self.current_search_space.size(0),),
            self.eps, dtype=torch.float32, device=self.device, requires_grad=True
        )

        # (Optional) If PRBCD needs to know add vs remove for each pair:
        # You could store a boolean mask aligned to current_search_space.
        # Build mask by recomputing linear indices for adds/removes separately and intersecting with lin.
        # Example (cheap and safe even if we truncated/reordered above):
        #   lin_all = self.pairs_to_linear_uppertri(edges_idx.to(self.device), N)
        #   is_add_full = is_add_mask.to(self.device)
        #   add_lin = lin_all[is_add_full]
        #   is_add_flag = torch.isin(self.current_search_space, add_lin)
        #   self.is_add_mask = is_add_flag  # save if your PRBCD uses it

        # --- 7) Budget sanity ---
        if self.current_search_space.size(0) < n_perturbations:
            raise RuntimeError(
                f"GNN block smaller than n_perturbations: "
                f"{self.current_search_space.size(0)} < {n_perturbations}. "
                f"Try increasing B or lowering the budget."
            )

    def sample_block_from_pgdtopk_direct(
            self,
            k_nodes: int = 350,
            prefer_edges: str = "intra",  # 'intra' | 'incident' | 'mix'
            loss_nodes=None,
            loss_type: str = "ce",
            kappa: float = 0.0,
            max_intra_ratio: float = 0.1,
    ):
        """
        Build PR-BCD block directly from a single PGD-style node saliency pass:
          1) y_node = PGD node importance (N,)
          2) topk = arg top-K y_node
          3) candidate edges = intra(topk) plus incident(topk <-> outside) to fill block_size
          4) finalize PR-BCD block tensors
        """
        import math
        import torch

        dev = self.device
        N = int(self.n)

        # --- (1) PGD node importances (teacher) ---

        y_node = self._pgd_node_labels(loss_nodes=loss_nodes, loss_type=loss_type, kappa=kappa)  # (N,)
            # z-score is already done in _pgd_node_labels; if you remove it there, uncomment:
            # y_node = (y_node - y_node.mean()) / (y_node.std() + 1e-6)

        # --- (2) pick Top-K nodes directly from y_node ---
        k = int(min(max(2, k_nodes), N))
        topk = torch.topk(y_node, k=k, largest=True).indices.to(dev)
        self.topk_nodes = topk  # (for inspection)

        # --- helper to convert (u,v) to upper-tri linear index ---
        def pair_to_lin(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            uu = torch.minimum(u, v)
            vv = torch.maximum(u, v)
            mask = uu < vv
            if mask.sum() == 0:
                return torch.empty(0, dtype=torch.long, device=dev)
            uu, vv = uu[mask], vv[mask]
            full_idx = torch.stack([uu, vv], dim=0)
            return PRBCD.triu_idx_to_linear_idx(N, full_idx)

        # --- (3) candidate edges: intra-TopK and incident to outside ---
        # Intra-TopK
        if topk.numel() >= 2:
            comb = torch.combinations(topk, r=2, with_replacement=False)  # (C(k,2), 2)
            intra_u, intra_v = comb[:, 0], comb[:, 1]
            intra_scores = y_node[intra_u] + y_node[intra_v]
            intra_lin = pair_to_lin(intra_u, intra_v)
        else:
            intra_u = intra_v = torch.empty(0, dtype=torch.long, device=dev)
            intra_scores = torch.empty(0, device=dev)
            intra_lin = torch.empty(0, dtype=torch.long, device=dev)

        # Incident: connect TopK to outside
        all_idx = torch.arange(N, device=dev)
        is_top = torch.zeros(N, dtype=torch.bool, device=dev)
        is_top[topk] = True
        outside = all_idx[~is_top]

        incident_lin = torch.empty(0, dtype=torch.long, device=dev)
        incident_scores = torch.empty(0, device=dev)
        if outside.numel() > 0 and prefer_edges in ("incident", "mix"):
            target_block = int(self.block_size)
            intra_quota = int(round(max_intra_ratio * target_block)) if prefer_edges in ("intra", "mix") else 0
            remaining = max(0, target_block - intra_quota)
            per_u = max(1, math.ceil(remaining / max(1, topk.numel()))) if remaining > 0 else 0

            inc_u_list, inc_v_list, inc_s_list = [], [], []
            if per_u > 0:
                for u in topk:
                    if outside.numel() <= per_u:
                        v = outside
                    else:
                        sel = torch.randperm(outside.numel(), device=dev)[:per_u]
                        v = outside[sel]
                    if v.numel() == 0:
                        continue
                    inc_u_list.append(u.repeat(v.numel()))
                    inc_v_list.append(v)
                    inc_s_list.append(y_node[u].repeat(v.numel()) + y_node[v])

            if len(inc_u_list) > 0:
                inc_u = torch.cat(inc_u_list)
                inc_v = torch.cat(inc_v_list)
                inc_scores = torch.cat(inc_s_list)

                # unique within incident pool (upper-tri)
                uu = torch.minimum(inc_u, inc_v)
                vv = torch.maximum(inc_u, inc_v)
                lin = (uu * N + vv).to(torch.long)

                # dedupe via sort + mask to get original indices
                lin_sorted, perm = torch.sort(lin)
                keep = torch.ones_like(lin_sorted, dtype=torch.bool, device=lin.device)
                keep[1:] = lin_sorted[1:] != lin_sorted[:-1]
                uniq_idx = perm[keep]

                inc_u, inc_v, inc_scores = inc_u[uniq_idx], inc_v[uniq_idx], inc_scores[uniq_idx]
                incident_scores = inc_scores
                incident_lin = pair_to_lin(inc_u, inc_v)

        # --- (4) assemble the block by scores ---
        chosen = []
        need_total = int(self.block_size)

        # how much intra to take first
        intra_quota = int(round(max_intra_ratio * need_total)) if prefer_edges in ("intra", "mix") else 0

        if intra_lin.numel() > 0 and intra_quota > 0:
            take = min(intra_quota, intra_lin.numel())
            idx = torch.topk(intra_scores, k=take, largest=True).indices
            chosen.append(intra_lin[idx])

        taken = sum(x.numel() for x in chosen) if len(chosen) else 0
        need = max(0, need_total - taken)

        if need > 0 and incident_lin.numel() > 0:
            take = min(need, incident_lin.numel())
            idx = torch.topk(incident_scores, k=take, largest=True).indices
            chosen.append(incident_lin[idx])
            need -= take

        # fallback: random upper-tri edges to pad
        if need > 0:
            n_possible = N * (N - 1) // 2
            chosen.append(torch.randint(n_possible, (need,), device=dev, dtype=torch.long))

        sel_lin = torch.unique(torch.cat(chosen), sorted=True) if len(chosen) else torch.empty(0, dtype=torch.long,
                                                                                               device=dev)

        # --- finalize PR-BCD tensors ---
        self.current_search_space = sel_lin
        self.modified_edge_index = PRBCD.linear_to_triu_idx(N, sel_lin)
        self.perturbed_edge_weight = torch.full(
            (sel_lin.numel(),), self.eps, dtype=torch.float32, device=dev, requires_grad=True
        )

        # (optional) warn if very small
        if self.current_search_space.size(0) < need_total:
            import logging
            logging.warning("[PGD-TopK-Direct] Assembled %d < block_size=%d edges.",
                            self.current_search_space.size(0), need_total)

    def _selector_online_step(self):
        # no-op / lazy init guards (keep yours if present)
        if getattr(self, "selector", None) is None:
            return
        if getattr(self, "selector_opt", None) is None:
            self.selector_opt = torch.optim.Adam(
                self.selector.parameters(), lr=1e-3, weight_decay=5e-4
            )

        # teacher labels don't need grad
        with torch.enable_grad():
            y = self._make_margin_labels().to(self.device)

        x = self.attr.to(self.device).float()
        ei, ew = self.get_modified_adj()
        ei = ei.to(self.device)
        ew = ew.to(self.device).float()

        # <-- turn grads back on just for the selector step
        with torch.enable_grad():
            self.selector.train()
            self.selector_opt.zero_grad(set_to_none=True)

            h = self.selector.embed(x, ei, ew)  # requires grad
            pred = self.selector.node_head(h).squeeze(-1)  # requires grad
            loss = F.smooth_l1_loss(pred, y.detach())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.selector.parameters(), 5.0)
            self.selector_opt.step()
            self.selector.eval()

    def _selector_online_step_pgd(self):
        if getattr(self, "selector", None) is None:
            return
        if getattr(self, "selector_opt", None) is None:
            self.selector_opt = torch.optim.Adam(self.selector.parameters(), lr=1e-3, weight_decay=5e-4)

        # --- teacher: node vulnerability from a single PGD-style backward
        # IMPORTANT: must be OUTSIDE no_grad, since it runs a backward() internally
        with torch.enable_grad():
            y = self._pgd_node_labels(loss_nodes=self.idx_attack, loss_type="ce").detach()  # (N,)

        # --- data on current (attacked) graph
        x = self.attr.to(self.device).float()
        ei, ew = self.get_modified_adj()
        ei = ei.to(self.device)
        ew = ew.to(self.device).float()

        # --- one training step for the selector
        with torch.enable_grad():
            self.selector.train()
            self.selector_opt.zero_grad(set_to_none=True)
            h = self.selector.embed(x, ei, ew)  # (N, d)
            pred = self.selector.node_head(h).squeeze(-1)  # (N,)
            loss = F.smooth_l1_loss(pred, y)  # teacher already z-scored
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.selector.parameters(), 5.0)
            self.selector_opt.step()
            self.selector.eval()

    def sample_block_from_selector(
            self,
            n_perturbations: int = 0,
            k_nodes: int = None,
            prioritize_intra: bool = True,
            ensure_pretrain: bool = True,
            pretrain_epochs: int = 20,
            pretrain_lr: float = 1e-3,
            pretrain_wd: float = 5e-4,
            # blend control
            alpha_saliency: float = 0.1,  # node_score = α·sal + (1-α)·pred
            saliency_edges: bool = True,  # True => edge probs from saliency only
            gumbel_nodes: bool = True,  # sample top-k nodes via Gumbel-softmax
            gumbel_edges: bool = True,  # sample edges via Gumbel-softmax
            tau_nodes: float = 1.0,  # temperature for node sampling
            tau_edges: float = 1.0,  # temperature for edge sampling
            max_intra_ratio: float = 0.1,  # quota for intra-topk edges
            log_examples: int = 10,
    ):
        """
        Build a PR-BCD block over ALL undirected pairs (i<j), using stochastic
        Gumbel-Softmax sampling:
          • Nodes: sample k via Gumbel-top-k over softmax(node_score / τ_nodes)
          • Edges: sample via Gumbel-top-k over softmax(edge_score / τ_edges)

        Edge scores can be taken from *pure saliency* (default) or the blended node score.
        """
        import math
        import torch

        dev, N = self.device, self.n

        # 0) Availability Check
        if ensure_pretrain and not getattr(self, "_margin_trained", False):
            logging.info("[Selector] Pretraining NodeBlockScorer...")
            self._pretrain_margin_scorer(epochs=pretrain_epochs, lr=pretrain_lr, wd=pretrain_wd)
        else:
            logging.info("[Selector] Using NodeBlockScorer (pretrained=%s).",
                         getattr(self, "_margin_trained", False))

        sal = self._node_saliency_from_victim(subset="attack").to(dev)
        sal = (sal - sal.mean()) / (sal.std() + 1e-6)

        self.selector.eval()
        with torch.no_grad():
            x = self.attr.to(dev).float()
            ei = self.edge_index.to(dev)
            ew = (self.edge_weight.to(dev).float()
                  if self.edge_weight is not None
                  else torch.ones(ei.size(1), device=dev))
            h = self.selector.embed(x, ei, ew)
            pred = self.selector.node_head(h).squeeze(-1)
            pred = (pred - pred.mean()) / (pred.std() + 1e-6)

        # 2) Blended node score for *ranking/probabilities*
        alpha = float(alpha_saliency)
        node_score = alpha * sal + (1.0 - alpha) * pred

        # Helper: mapped (u,v) -> linear upper-tri index
        def pair_to_lin(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            uu = torch.minimum(u, v)
            vv = torch.maximum(u, v)
            mask = (uu < vv)
            if mask.sum() == 0:
                return torch.empty(0, dtype=torch.long, device=dev)
            uu, vv = uu[mask], vv[mask]
            return PRBCD.triu_idx_to_linear_idx(N, torch.stack([uu, vv], dim=0))

        # 4) Pick k nodes — either greedy top‑k or Gumbel‑top‑k over softmax(node_score/τ)
        if k_nodes is None:
            k_nodes = int(math.ceil((1.0 + math.sqrt(1.0 + 8.0 * self.block_size)) / 2.0))
        k = int(min(k_nodes, N))

        if gumbel_nodes:
            node_logits = torch.log_softmax(node_score / max(1e-6, float(tau_nodes)), dim=0)
            topk_nodes = PRBCD._gumbel_topk(self ,node_logits, k).to(dev)
        else:
            topk_nodes = torch.topk(node_score, k=k, largest=True).indices.to(dev)

        if log_examples > 0:
            logging.info("[Selector] k=%d | example top nodes: %s", k, topk_nodes[:log_examples].tolist())

        # 5) Build intra-topk candidate edges + scores
        if topk_nodes.numel() >= 2:
            comb = torch.combinations(topk_nodes, r=2, with_replacement=False)  # (C(k,2), 2)
            intra_u, intra_v = comb[:, 0], comb[:, 1]
            intra_lin = pair_to_lin(intra_u, intra_v)
        else:
            intra_u = intra_v = torch.empty(0, dtype=torch.long, device=dev)
            intra_lin = torch.empty(0, dtype=torch.long, device=dev)

        # Edge scores: by default use *pure saliency* (requested), or blended
        node_for_edges = sal if bool(saliency_edges) else node_score
        intra_scores = (node_for_edges[intra_u] + node_for_edges[intra_v]) if intra_lin.numel() > 0 \
            else torch.empty(0, device=dev)

        # 6) Incident candidates (topk ↔ outside) — sample partners by node probabilities
        all_idx = torch.arange(N, device=dev)
        is_top = torch.zeros(N, dtype=torch.bool, device=dev)
        is_top[topk_nodes] = True
        outside = all_idx[~is_top]

        incident_lin_list, incident_scores_list = [], []
        need_total = int(self.block_size)
        intra_quota = int(round(max_intra_ratio * need_total)) if prioritize_intra else 0
        # aim to fill remaining with incident; split roughly across top nodes
        remaining_need = max(0, need_total - (intra_lin.numel() if intra_quota > 0 else 0))
        per_top_quota = max(1, math.ceil(remaining_need / max(1, topk_nodes.numel()))) if remaining_need > 0 else 0

        if outside.numel() > 0 and per_top_quota > 0:
            outside_logits = torch.log_softmax(node_for_edges[outside] / max(1e-6, float(tau_edges)), dim=0)
            for u in topk_nodes:
                if per_top_quota <= 0:
                    break
                # sample partners for u (with replacement) by Gumbel-top-k over outside logits
                # (equivalent to drawing the highest Gumbel-perturbed log-probs)
                v_sel = self._gumbel_topk(outside_logits, per_top_quota)
                v = outside[v_sel]
                u_rep = u.repeat(v.numel())
                lin = pair_to_lin(u_rep, v)
                if lin.numel() == 0:
                    continue
                incident_lin_list.append(lin)
                incident_scores_list.append(node_for_edges[u_rep] + node_for_edges[v])

        if len(incident_lin_list) > 0:
            incident_lin = torch.cat(incident_lin_list, dim=0)
            incident_scores = torch.cat(incident_scores_list, dim=0)
            # unique within incident pool
            uniq_lin, uniq_idx = torch.unique(incident_lin, sorted=False, return_inverse=False, return_counts=False,
                                              return_indices=True)
            incident_lin = uniq_lin
            incident_scores = incident_scores[uniq_idx]
        else:
            incident_lin = torch.empty(0, dtype=torch.long, device=dev)
            incident_scores = torch.empty(0, device=dev)

        # 7) Assemble block: sample edges via Gumbel-top-k (or greedy if gumbel_edges=False)
        def pick_edges(idx_lin: torch.Tensor, scores: torch.Tensor, m: int) -> torch.Tensor:
            if idx_lin.numel() == 0 or m <= 0:
                return torch.empty(0, dtype=torch.long, device=dev)
            m = min(m, idx_lin.numel())
            if gumbel_edges:
                logits = torch.log_softmax(scores / max(1e-6, float(tau_edges)), dim=0)
                sel = self._gumbel_topk(logits, m)
            else:
                sel = torch.topk(scores, k=m, largest=True).indices
            return idx_lin[sel]

        chosen, need = [], need_total

        if prioritize_intra and intra_lin.numel() > 0:
            take_intra = pick_edges(intra_lin, intra_scores, min(intra_quota, need))
            chosen.append(take_intra)
            need -= take_intra.numel()

        if need > 0 and incident_lin.numel() > 0:
            take_inc = pick_edges(incident_lin, incident_scores, need)
            chosen.append(take_inc)
            need -= take_inc.numel()

        # fallback: random from the whole upper-tri space to keep exploration
        if need > 0:
            n_possible = N * (N - 1) // 2
            chosen.append(torch.randint(n_possible, (need,), device=dev, dtype=torch.long))
            need = 0

        sel_lin = torch.unique(torch.cat(chosen), sorted=True) if len(chosen) else torch.empty(0, dtype=torch.long,
                                                                                               device=dev)
        if sel_lin.numel() < self.block_size:
            logging.warning("[Selector] Only %d candidates assembled (requested %d).", sel_lin.numel(), self.block_size)

        # 8) Finalize PR-BCD block tensors
        self.current_search_space = sel_lin
        self.modified_edge_index = PRBCD.linear_to_triu_idx(N, sel_lin)
        self.perturbed_edge_weight = torch.full((sel_lin.numel(),), self.eps, dtype=torch.float32, device=dev,
                                                requires_grad=True)

        # Budget sanity
        if self.current_search_space.size(0) < n_perturbations:
            logging.warning("[Selector] Block smaller than n_perturbations: %d < %d.",
                            self.current_search_space.size(0), n_perturbations)

    def _current_node_scores(self, blend_alpha: float = 0.0) -> torch.Tensor:
        """
        Compute node scores used for guided sampling.
        blend_alpha=0 -> pure selector prediction (fast);
        blend_alpha>0 -> blend with saliency for stability: alpha*sal + (1-alpha)*pred
        """
        x = self.attr.to(self.device).float()
        ei, ew = self.get_modified_adj()
        ei = ei.to(self.device)
        ew = ew.to(self.device).float()

        with torch.no_grad():
            h = self.selector.embed(x, ei, ew)
            pred = self.selector.node_head(h).squeeze(-1)
            pred = (pred - pred.mean()) / (pred.std() + 1e-6)

        if blend_alpha <= 1e-9:
            return pred

        with torch.enable_grad():
            sal = self._node_saliency_from_victim(subset="attack").to(self.device)
            sal = (sal - sal.mean()) / (sal.std() + 1e-6)
        return float(blend_alpha) * sal + (1.0 - float(blend_alpha)) * pred

    def _assemble_edges_from_nodes(
            self,
            node_scores: torch.Tensor,
            need: int,
            prefer: str = "mix",
            max_intra_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Build up to `need` new upper-tri *linear* edge indices guided by node_scores.
        prefer: 'intra' | 'incident' | 'mix'
        """
        import math
        dev = self.device
        N = int(self.n)

        # ---- choose k so C(k,2) isn't trivially < need ----
        k_nodes = int(min(N, max(2, math.ceil(0.5 * (1 + math.sqrt(1 + 8 * need))))))  # ~inverse of C(k,2)
        topk = torch.topk(node_scores, k=k_nodes, largest=True).indices.to(dev)

        # ---- intra-topk candidates ----
        if topk.numel() >= 2:
            comb = torch.combinations(topk, r=2, with_replacement=False)  # (C(k,2), 2)
            intra_u, intra_v = comb[:, 0], comb[:, 1]
            uu = torch.minimum(intra_u, intra_v)
            vv = torch.maximum(intra_u, intra_v)
            intra_lin = PRBCD.triu_idx_to_linear_idx(N, torch.stack([uu, vv], dim=0))
            intra_scores = node_scores[intra_u] + node_scores[intra_v]
        else:
            intra_lin = torch.empty(0, dtype=torch.long, device=dev)
            intra_scores = torch.empty(0, device=dev)

        # ---- incident candidates (topk ↔ outside) ----
        all_idx = torch.arange(N, device=dev)
        is_top = torch.zeros(N, dtype=torch.bool, device=dev)
        is_top[topk] = True
        outside = all_idx[~is_top]

        inc_lin = torch.empty(0, dtype=torch.long, device=dev)
        inc_scores = torch.empty(0, device=dev)

        if outside.numel() > 0 and prefer in ("incident", "mix"):
            per_u = max(1, math.ceil(need / max(1, topk.numel())))
            us, vs, ss = [], [], []
            for u in topk:
                if outside.numel() <= per_u:
                    v = outside
                else:
                    sel = torch.randperm(outside.numel(), device=dev)[:per_u]
                    v = outside[sel]
                if v.numel() == 0:
                    continue
                uu = torch.minimum(u.repeat(v.numel()), v)
                vv = torch.maximum(u.repeat(v.numel()), v)
                us.append(uu);
                vs.append(vv)
                ss.append(node_scores[u].repeat(v.numel()) + node_scores[v])

            if len(us) > 0:
                uu = torch.cat(us);
                vv = torch.cat(vs);
                ss = torch.cat(ss)

                # ---- dedupe (uu,vv): use sort+mask (works on all PyTorch) ----
                lin_raw = uu * N + vv  # linearize pairs (upper-tri since uu<=vv)
                lin_sorted, perm = torch.sort(lin_raw)  # sort to detect dups
                keep = torch.ones_like(lin_sorted, dtype=torch.bool, device=dev)
                keep[1:] = lin_sorted[1:] != lin_sorted[:-1]
                uniq_idx = perm[keep]  # indices into uu/vv/ss to keep

                uu = uu[uniq_idx];
                vv = vv[uniq_idx];
                ss = ss[uniq_idx]
                inc_lin = PRBCD.triu_idx_to_linear_idx(N, torch.stack([uu, vv], dim=0))
                inc_scores = ss

        # ---- assemble by quota ----
        chosen = []
        quota_intra = int(round(max_intra_ratio * need)) if prefer in ("intra", "mix") else 0

        if intra_lin.numel() > 0 and quota_intra > 0:
            take = min(quota_intra, intra_lin.numel())
            idx = torch.topk(intra_scores, k=take, largest=True).indices
            chosen.append(intra_lin[idx])

        taken = sum(x.numel() for x in chosen) if chosen else 0
        need_rest = max(0, need - taken)

        if need_rest > 0 and inc_lin.numel() > 0:
            take = min(need_rest, inc_lin.numel())
            idx = torch.topk(inc_scores, k=take, largest=True).indices
            chosen.append(inc_lin[idx])
            need_rest -= take

        if need_rest > 0:
            # pad with random upper-tri edges
            n_possible = N * (N - 1) // 2
            chosen.append(torch.randint(n_possible, (need_rest,), device=dev, dtype=torch.long))

        # final linear indices (unique to be safe against intra∩incident overlap)
        return torch.unique(torch.cat(chosen), sorted=True) if chosen else torch.empty(0, dtype=torch.long, device=dev)

    def resample_block_from_selector(
            self,
            n_perturbations: int,
            blend_alpha: float = 1.0,  # <-- default to using saliency (α=1.0)
            prefer_edges: str = "mix",
            max_intra_ratio: float = 0.5,
    ):
        if self.keep_heuristic != "WeightOnly":
            raise NotImplementedError("Only keep_heuristic=`WeightOnly` supported")

        # --- keep phase (same as before) ---
        sorted_idx = torch.argsort(self.perturbed_edge_weight)  # ascending
        idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
        if idx_keep < sorted_idx.size(0) // 2:
            idx_keep = sorted_idx.size(0) // 2
        keep_idx = sorted_idx[idx_keep:]

        kept_lin = self.current_search_space[keep_idx].to(self.device)
        kept_w = self.perturbed_edge_weight[keep_idx].to(self.device)
        kept_pairs = PRBCD.linear_to_triu_idx(self.n, kept_lin) if self.make_undirected \
            else PRBCD.linear_to_full_idx(self.n, kept_lin)

        # --- selector update (PGD teacher) ---
        self._selector_online_step_pgd()

        # --- refill guided by (α·saliency + (1−α)·selector) node scores ---
        need = int(self.block_size) - int(kept_lin.numel())
        if need <= 0:
            self.current_search_space = kept_lin
            self.modified_edge_index = kept_pairs
            self.perturbed_edge_weight = kept_w
            return


        node_scores = self._current_node_scores(blend_alpha=blend_alpha)  # α>0 uses saliency
        add_lin = self._assemble_edges_from_nodes(
            node_scores, need, prefer=prefer_edges, max_intra_ratio=max_intra_ratio
        )
        with torch.no_grad():
            if add_lin.numel() > 0:
                merged = torch.cat([kept_lin, add_lin.to(self.device)], dim=0)
                all_lin, inv = torch.unique(merged, sorted=True, return_inverse=True)
                pos_kept = inv[: kept_lin.numel()]
            else:
                all_lin = kept_lin
                pos_kept = torch.arange(kept_lin.numel(), device=self.device, dtype=torch.long)

            self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, all_lin) if self.make_undirected \
                else PRBCD.linear_to_full_idx(self.n, all_lin)

            new_w = torch.full((all_lin.numel(),), self.eps, dtype=torch.float32, device=self.device)
            new_w[pos_kept] = kept_w
            self.perturbed_edge_weight = new_w
            self.current_search_space = all_lin

            if not self.make_undirected:
                is_not_self = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self]
                self.perturbed_edge_weight = self.perturbed_edge_weight[is_not_self]

        if self.current_search_space.size(0) <= n_perturbations:
            logging.warning("[SelectorResample] Block size %d ≤ n_perturbations %d; consider increasing block_size.",
                            self.current_search_space.size(0), n_perturbations)

    def resample_random_block_from_cert_radii(self, grid_radii, n_perturbations: int = 0): #TODO: still work to be done
        #self.current_node_search_space = np.where(grid_radii[:, 2, 2] == False)[0]
        self.current_node_search_space = np.where(grid_radii[:, 1, 1] == False)[0]
        if self.keep_heuristic == 'WeightOnly':
            sorted_idx = torch.argsort(self.perturbed_edge_weight)
            idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
            # Keep at most half of the block (i.e. resample low weights)
            if idx_keep < sorted_idx.size(0) // 2:
                idx_keep = sorted_idx.size(0) // 2
        else:
            raise NotImplementedError('Only keep_heuristic=`WeightOnly` supported')

        sorted_idx = sorted_idx[idx_keep:]
        self.current_search_space = self.current_search_space[sorted_idx]
        self.modified_edge_index = self.modified_edge_index[:, sorted_idx]
        self.perturbed_edge_weight = self.perturbed_edge_weight[sorted_idx]

        # Sample until enough edges were drawn
        for i in range(self.max_final_samples):
            n_edges_resample = self.block_size - self.current_search_space.size(0)

            # resample new edges
            if not self.semi:
                self.build_full_idx_matrix(True, n_edges_resample)
            else:
                self.build_full_idx_matrix_semi(True, n_edges_resample)
            lin_index = self.edges_to_current_search_space(self.n)

            self.current_search_space, unique_idx = torch.unique(
                torch.cat((self.current_search_space, lin_index)),
                sorted=True,
                return_inverse=True
            )

            if self.make_undirected:
                self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = PRBCD.linear_to_full_idx(self.n, self.current_search_space)

            # Merge existing weights with new edge weights
            perturbed_edge_weight_old = self.perturbed_edge_weight.clone()
            self.perturbed_edge_weight = torch.full_like(self.current_search_space, self.eps, dtype=torch.float32)
            self.perturbed_edge_weight[
                unique_idx[:perturbed_edge_weight_old.size(0)]
            ] = perturbed_edge_weight_old

            if not self.make_undirected:
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]
                self.perturbed_edge_weight = self.perturbed_edge_weight[is_not_self_loop]

            if self.current_search_space.size(0) > n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    def resample_random_block_from_cert_binary_class(self, grid_binary_class, n_perturbations: int = 0): #TODO: still work to be done
        if self.keep_heuristic == 'WeightOnly':
            sorted_idx = torch.argsort(self.perturbed_edge_weight)
            idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
            # Keep at most half of the block (i.e. resample low weights)
            if idx_keep < sorted_idx.size(0) // 2:
                idx_keep = sorted_idx.size(0) // 2
        else:
            raise NotImplementedError('Only keep_heuristic=`WeightOnly` supported')

        sorted_idx = sorted_idx[idx_keep:]
        self.current_search_space = self.current_search_space[sorted_idx]
        self.modified_edge_index = self.modified_edge_index[:, sorted_idx]
        self.perturbed_edge_weight = self.perturbed_edge_weight[sorted_idx]

        # sums = np.sum(grid_binary_class, axis=(1, 2))  # shape: (2810,)
        # smallest_indices = np.argsort(sums)[:len(sums) // 2]
        # self.current_node_search_space = torch.from_numpy(smallest_indices)

        # Sample until enough edges were drawn
        for i in range(self.max_final_samples):
            # testing a random sample pattern
            self.sample_current_node_search_space_random_cert(grid_binary_class, 1000)
            n_edges_resample = self.block_size - self.current_search_space.size(0)

            # resample new edges
            if not self.semi:
                self.build_full_idx_matrix(True, n_edges_resample)
            else:
                self.build_full_idx_matrix_semi(True, n_edges_resample)
            lin_index = self.edges_to_current_search_space(self.n)

            self.current_search_space, unique_idx = torch.unique(
                torch.cat((self.current_search_space, lin_index)),
                sorted=True,
                return_inverse=True
            )

            if self.make_undirected:
                self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = PRBCD.linear_to_full_idx(self.n, self.current_search_space)

            # Merge existing weights with new edge weights
            perturbed_edge_weight_old = self.perturbed_edge_weight.clone()
            self.perturbed_edge_weight = torch.full_like(self.current_search_space, self.eps, dtype=torch.float32)
            self.perturbed_edge_weight[
                unique_idx[:perturbed_edge_weight_old.size(0)]
            ] = perturbed_edge_weight_old

            if not self.make_undirected:
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]
                self.perturbed_edge_weight = self.perturbed_edge_weight[is_not_self_loop]

            if self.current_search_space.size(0) > n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    def resample_block_from_prior_gumbel(self, n_perturbations: int, tau: float = 1.0):
        """
        Keep at most half of the current block (largest weights),
        then refill from the selector prior via Gumbel-Softmax + masking.
        """
        # --- keep heuristic (same semantics as your resample_random_block) ---
        if self.keep_heuristic != 'WeightOnly':
            raise NotImplementedError('Only keep_heuristic=`WeightOnly` supported')

        sorted_idx = torch.argsort(self.perturbed_edge_weight)  # ascending
        idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
        if idx_keep < sorted_idx.size(0) // 2:
            idx_keep = sorted_idx.size(0) // 2
        keep_idx = sorted_idx[idx_keep:]  # keep largest half

        # slice current block
        self.current_search_space = self.current_search_space[keep_idx]
        self.modified_edge_index = self.modified_edge_index[:, keep_idx]
        self.perturbed_edge_weight = self.perturbed_edge_weight[keep_idx]

        # --- refill from prior ---
        need = self.block_size - self.current_search_space.numel()
        if need <= 0:
            # nothing to refill
            self._selector_log_probs = None
            return

        # sample_block_from_prior_gumbel / resample_block_from_prior_gumbel

        logits = self.pre_gnn_prior(self.attr, self.edge_index, self.edge_weight)  # (E,)
        logits = logits.to(self.device).float()  # <- ensure float dtype

        # mask already-selected global edges
        mask = torch.zeros_like(logits)
        mask[self.current_search_space] = float('-inf')

        picked, logps = [], []
        for _ in range(need):
            y = F.gumbel_softmax(logits + mask, tau=tau, hard=True, dim=0)  # (E,)
            j = int(y.argmax())
            picked.append(j)

            logp = F.log_softmax(logits + mask, dim=0)[j]
            logps.append(logp)

            mask[j] = float('-inf')

        add = torch.tensor(picked, device=self.device, dtype=torch.long)

        # extend the block; no duplicates because of mask
        self.current_search_space = torch.cat([self.current_search_space, add], dim=0)
        self.modified_edge_index = self.edge_index[:, self.current_search_space]
        self.perturbed_edge_weight = torch.cat([
            self.perturbed_edge_weight,
            torch.full((add.numel(),), self.eps, dtype=torch.float32, device=self.device)
        ], dim=0)

        # store log-probs for PG update
        self._selector_log_probs = torch.stack(logps)

    def _make_margin_labels(self):
        with torch.no_grad():
            x = self.attr.to(self.device)
            ei = self.edge_index.to(self.device)

            ew = self.edge_weight
            if ew is None:
                ew = torch.ones(ei.size(1), device=self.device)
            else:
                ew = ew.to(self.device)

            logits = self.attacked_model(data=x, adj=(ei, ew))

            top2_vals = torch.topk(logits, k=2, dim=1).values
            margins = (top2_vals[:, 0] - top2_vals[:, 1]).clamp_min(0.0)

            mean, std = margins.mean(), margins.std().clamp_min(1e-6)
            return (margins - mean) / std

    def _pretrain_margin_scorer(self, epochs: int = 20, lr: float = 1e-3, wd: float = 5e-4):
        logging.info(f"[Selector] Pretraining NodeBlockScorer for {epochs} epochs "
                     f"(lr={lr}, wd={wd}, device={self.device}).")
        self.selector.train()
        y = self._make_margin_labels().to(self.device)

        x = self.attr.to(self.device)
        ei = self.edge_index.to(self.device)
        ew = self.edge_weight.to(self.device)

        opt = torch.optim.Adam(self.selector.parameters(), lr=lr, weight_decay=wd)
        with torch.enable_grad():
            for e in range(epochs):
                opt.zero_grad()
                h = self.selector.embed(x, ei, ew)
                pred = self.selector.node_head(h).squeeze(-1)
                loss = F.smooth_l1_loss(pred, y)
                loss.backward()
                opt.step()

                # log a few times during training
                if (e + 1) % max(1, epochs // 5) == 0 or e == 0 or (e + 1) == epochs:
                    logging.info(f"[Selector] pretrain epoch {e + 1}/{epochs} - loss={loss.item():.4f}")

        self.selector.eval()
        self._margin_trained = True
        logging.info("[Selector] Pretraining finished.")

    def _pretrain_pregnn_linkpred(
            self,
            epochs: int = 20,
            lr: float = 1e-3,
            wd: float = 5e-4,
            neg_per_pos: int = 1,
            batch_size: int = 65536,
            use_bilinear: bool = True,
            edge_dropout: float = 0.1,
            temp: float = 1.0,
            log_every: int = 5,
            # ---- NEW knobs for Top-K sampling ----
            topk_pos_ratio: float | None = None,  # e.g., 0.25 -> keep top 25% positives
            topk_pos_mode: str = "loss",  # {"loss","score"} ranking criterion
            hard_negatives: bool = True,  # do hard negative mining
            neg_oversample: int = 4,  # oversample pool for hard negs
    ):
        """
        Pretrain the GNN encoder via link prediction, with optional Top-K edge sampling.

        Top-K positives:
            - If topk_pos_ratio is not None, compute scores (or per-edge loss)
              for *all* positives this epoch and keep only the Top-K fraction.

        Hard negatives:
            - If hard_negatives is True, for each batch oversample K*neg_oversample
              random non-edges, score them, and keep only the highest-scoring K.

        Loss: BCEWithLogits(pos=1, neg=0)
        """
        import math
        import torch
        import logging
        from torch import nn
        from torch.nn import functional as F

        device = self.device
        self.selector.train()

        # --- Data on device ---
        x = self.attr.to(device).float()
        ei = self.edge_index.to(device)  # (2, E)
        ew = (self.edge_weight.to(device).float()
              if self.edge_weight is not None else None)

        N = int(self.n)
        E = int(ei.size(1))

        # --- Undirected unique positives (u<v) ---
        u = torch.minimum(ei[0], ei[1])
        v = torch.maximum(ei[0], ei[1])
        mask = (u != v)
        u, v = u[mask], v[mask]
        pos_uv = torch.unique(torch.stack([u, v], dim=0), dim=1)  # (2, Epos)
        Epos = pos_uv.size(1)

        # --- Membership set for negatives ---
        pos_keys = (pos_uv[0].long() * N + pos_uv[1].long()).tolist()
        pos_set = set(pos_keys)

        # --- Optional bilinear scorer ---
        bilinear = None
        if use_bilinear:
            bilinear = None  # lazy-init after we see embedding dim

        def score_pairs(h, a, b):
            nonlocal bilinear
            if use_bilinear:
                if bilinear is None:
                    d = h.size(-1)
                    bilinear = nn.Parameter(torch.empty(d, d, device=device))
                    nn.init.xavier_uniform_(bilinear)
                    self.selector.register_parameter("pregnn_bilinear_W", bilinear)
                return (h[a] @ bilinear @ h[b].T).diag()
            else:
                return (h[a] * h[b]).sum(dim=-1)

        def sample_negatives(num_neg: int):
            out_u, out_v = [], []
            needed, tries, max_tries = num_neg, 0, 10
            while needed > 0 and tries < max_tries:
                a = torch.randint(0, N, (needed * 2,), device=device)
                b = torch.randint(0, N, (needed * 2,), device=device)
                valid = (a != b)
                a, b = a[valid], b[valid]
                if a.numel() == 0:
                    tries += 1
                    continue
                uu = torch.minimum(a, b)
                vv = torch.maximum(a, b)
                keys = (uu.long() * N + vv.long()).tolist()
                keep = [k not in pos_set for k in keys]
                keep = torch.tensor(keep, device=device, dtype=torch.bool)
                uu, vv = uu[keep], vv[keep]
                if uu.numel() == 0:
                    tries += 1
                    continue
                take = min(needed, uu.numel())
                out_u.append(uu[:take])
                out_v.append(vv[:take])
                needed -= take
            if len(out_u) == 0:
                uu = torch.arange(0, num_neg, device=device) % (N - 1)
                vv = (uu + 1) % N
                return uu, vv
            return torch.cat(out_u), torch.cat(out_v)

        # --- Optimizer ---
        params = list(self.selector.parameters())
        opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)

        bce = torch.nn.BCEWithLogitsLoss()
        logging.info(
            f"[PreGNN] LinkPred pretrain: epochs={epochs}, lr={lr}, wd={wd}, "
            f"neg_per_pos={neg_per_pos}, batch_size={batch_size}, edge_dropout={edge_dropout}, "
            f"topk_pos_ratio={topk_pos_ratio}, topk_pos_mode={topk_pos_mode}, "
            f"hard_negatives={hard_negatives}, neg_oversample={neg_oversample}"
        )

        for ep in range(1, epochs + 1):
            self.selector.train()

            # DropEdge for robustness
            if edge_dropout > 0.0 and E > 0:
                keep_mask = torch.rand(E, device=device) > edge_dropout
                ei_train = ei[:, keep_mask]
                ew_train = (ew[keep_mask] if ew is not None else None)
            else:
                ei_train, ew_train = ei, ew

            # Full-batch embeddings (one forward per epoch)
            h = self.selector.embed(x, ei_train, ew_train)  # (N, d)

            # ---- Top-K POSITIVES selection (optional) ----
            pos_idx_pool = torch.arange(Epos, device=device)
            if topk_pos_ratio is not None and 0.0 < topk_pos_ratio < 1.0:
                pu_all = pos_uv[0, :]
                pv_all = pos_uv[1, :]
                s_pos_all = score_pairs(h, pu_all, pv_all)
                if temp != 1.0:
                    s_pos_all = s_pos_all / temp

                if topk_pos_mode == "score":
                    # keep the *highest-scoring* positive edges
                    k_pos = max(1, int(math.ceil(Epos * topk_pos_ratio)))
                    topk = torch.topk(s_pos_all, k=k_pos, largest=True).indices
                    pos_idx_pool = topk
                elif topk_pos_mode == "loss":
                    # keep positives with highest BCE loss vs label=1 (hardest positives)
                    # loss_pos = -log(sigmoid(s_pos))
                    loss_pos = F.softplus(-s_pos_all)  # numerically stable
                    k_pos = max(1, int(math.ceil(Epos * topk_pos_ratio)))
                    topk = torch.topk(loss_pos, k=k_pos, largest=True).indices
                    pos_idx_pool = topk
                else:
                    logging.warning(f"[PreGNN] Unknown topk_pos_mode={topk_pos_mode}; using full positives.")
            else:
                # shuffle all positives if not doing top-k
                perm = torch.randperm(Epos, device=device)
                pos_idx_pool = pos_idx_pool[perm]

            # batching over selected positives
            total_loss = 0.0
            num_batches = int(math.ceil(pos_idx_pool.numel() / batch_size))

            for b in range(num_batches):
                start = b * batch_size
                end = min(pos_idx_pool.numel(), (b + 1) * batch_size)
                idx = pos_idx_pool[start:end]

                pu = pos_uv[0, idx]
                pv = pos_uv[1, idx]

                # negatives needed for this mini-batch
                need_neg = (end - start) * max(1, int(neg_per_pos))

                if hard_negatives:
                    # oversample a pool, then keep the Top-K hardest (highest score)
                    nu_pool, nv_pool = sample_negatives(need_neg * max(1, int(neg_oversample)))
                    s_neg_pool = score_pairs(h, nu_pool, nv_pool)
                    if temp != 1.0:
                        s_neg_pool = s_neg_pool / temp
                    # Top hardest negatives
                    keep_k = min(need_neg, s_neg_pool.numel())
                    hard_idx = torch.topk(s_neg_pool, k=keep_k, largest=True).indices
                    nu, nv = nu_pool[hard_idx], nv_pool[hard_idx]
                else:
                    nu, nv = sample_negatives(need_neg)

                # compute scores
                s_pos = score_pairs(h, pu, pv)
                s_neg = score_pairs(h, nu, nv)
                if temp != 1.0:
                    s_pos = s_pos / temp
                    s_neg = s_neg / temp

                y_pos = torch.ones_like(s_pos)
                y_neg = torch.zeros_like(s_neg)

                loss = bce(s_pos, y_pos) + bce(s_neg, y_neg)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.selector.parameters(), max_norm=5.0)
                opt.step()

                total_loss += loss.item()

            if (ep % max(1, log_every) == 0) or ep == 1 or ep == epochs:
                avg_loss = total_loss / max(1, num_batches)
                logging.info(f"[PreGNN] epoch {ep}/{epochs} - linkpred loss={avg_loss:.4f}")

        self.selector.eval()
        self._pregnn_trained = True
        logging.info("[PreGNN] Link prediction pretraining finished with Top-K sampling.")

    @staticmethod
    def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
        row_idx = (
            n
            - 2
            - torch.floor(torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
        ).long()
        col_idx = (
            lin_idx
            + row_idx
            + 1 - n * (n - 1) // 2
            + (n - row_idx) * ((n - row_idx) - 1) // 2
        )
        return torch.stack((row_idx, col_idx))

    @staticmethod
    def linear_to_full_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
        row_idx = lin_idx // n
        col_idx = lin_idx % n
        return torch.stack((row_idx, col_idx))

    @staticmethod
    def full_to_linear_idx(n: int, full_idx: torch.Tensor) -> torch.Tensor:
        # this function made errors in the undirected case, maybe this will be useful for the directed case
        row_idx, col_idx = full_idx[0], full_idx[1]
        lin_idx = row_idx * n + col_idx
        return lin_idx

    @staticmethod
    def triu_idx_to_linear_idx(n: int, full_idx: torch.Tensor) -> torch.Tensor:
        # if im correct in the undirected case, the indexing of the block matrix is different to the directed case (see my note: (1) )
        row_idx, col_idx = full_idx[0], full_idx[1]
        lin_idx = (n * row_idx - row_idx * (row_idx + 1) // 2) + (col_idx - row_idx - 1)
        return lin_idx

    def build_full_idx_matrix(self, reset: bool, sample_size: int):
        if reset:
            # empty edges_to_attack_index for resample or other cases
            self.edges_to_attack_index = torch.empty((2, 0), dtype=torch.long)
        nodes_1 = torch.from_numpy(np.random.choice(self.current_node_search_space, size=sample_size, replace=True))
        # We could change this to only draw nodes_1 from unrobust nodes and nodes_2 from all possible nodes
        # Done: see next method
        nodes_2 = torch.from_numpy(np.random.choice(self.current_node_search_space, size=sample_size, replace=True))
        edges_idx = torch.cat([nodes_1.unsqueeze(0), nodes_2.unsqueeze(0)], dim=0)
        self.edges_to_attack_index = torch.cat([self.edges_to_attack_index, edges_idx], dim=1)
        return edges_idx

    def build_full_idx_matrix_semi(self, reset: bool, sample_size: int):
        if reset:
            # empty edges_to_attack_index for resample or other cases
            self.edges_to_attack_index = torch.empty((2, 0), dtype=torch.long)
        nodes_1 = torch.from_numpy(np.random.choice(self.current_node_search_space, size=sample_size, replace=True))
        nodes_2 = torch.randint(self.n, (sample_size,), device=self.device)
        edges_idx = torch.cat([nodes_1.unsqueeze(0), nodes_2.unsqueeze(0)], dim=0)
        self.edges_to_attack_index = torch.cat([self.edges_to_attack_index, edges_idx], dim=1)
        return edges_idx

    def setup_search_space_undirected(self, n: int):
        # matrix: random drawn block matrix this matrix is drawn from two node sets which are
        # not robust according to certificates
        # from here we want to follow same steps as sample_random_block
        self.current_search_space = self.edges_to_current_search_space(n)

        #while self.current_search_space.size(0) < self.block_size:
        #    # here we want to fill up current_search_space so that there are block_size many entries
        #    if not self.semi:
        #        self.build_full_idx_matrix(False, self.block_size)
        #    else:
        #        self.build_full_idx_matrix_semi(False, self.block_size)
        #    self.current_search_space = self.edges_to_current_search_space(n)

        # now we follow same steps as sample_random_block
        #self.current_search_space = self.current_search_space[
        #                                torch.randperm(self.current_search_space.size(0))[:self.block_size]
        #                                ]
        self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
        # if the new logic is usable this method is redundant
        return

    def edges_to_current_search_space(self, n: int):
        # first we cut edges so that index build a triu matrix
        # (function triu_idx_to_linear only support triu matrix idx)
        self.edges_to_attack_index = PRBCD.flip_matrix_idx_to_triu_idx(self.edges_to_attack_index)
        # we then build linear idx which is the current_search_space
        lin_idx = PRBCD.triu_idx_to_linear_idx(n, self.edges_to_attack_index)
        lin_idx = torch.unique(lin_idx, sorted=True)
        return lin_idx

    def sample_current_node_search_space_det(self, grid_binary_class):
        if self.use_cert in ("sampling_grid_binary_class_alt_11",):
            print("running alternative sampling with grid binary class (1,1)")
            unrobust_nodes_lin_index = np.where(grid_binary_class[:, 1, 1] < 0.3)  # shape: (2810,)
        else:
            print("running alternative sampling with grid binary class (2,2)")
            unrobust_nodes_lin_index = np.where(grid_binary_class[:, 2, 2] < 0.2)
        unrobust_nodes_lin_index = unrobust_nodes_lin_index[0]
        self.current_node_search_space = torch.unique(torch.from_numpy(unrobust_nodes_lin_index), sorted=False)
        return

    def sample_current_node_search_space_random_cert(self, grid_binary_class, sample_size):
        search_space = np.array([], dtype=np.int64)
        print("running random certificate samples")
        # Todo: Try different setups like different sample_size, while loop only to ensure no empty search_space, etc.
        while search_space.shape[0] <= 0:
            unrobust_nodes_lin_idx = np.where(grid_binary_class[:,
                                              np.random.randint(0, 3),
                                              np.random.randint(0, 5)] < np.random.uniform(0.1, 0.4))
            unrobust_nodes_lin_idx = unrobust_nodes_lin_idx[0]
            search_space = np.concatenate([search_space, unrobust_nodes_lin_idx], axis=0)

        self.current_node_search_space = torch.unique(torch.from_numpy(search_space), sorted=False)
        return

    def _append_attack_statistics(self, loss: float, accuracy: float,
                                  probability_mass_update: float, probability_mass_projected: float):
        self.attack_statistics['loss'].append(loss)
        self.attack_statistics['accuracy'].append(accuracy)
        self.attack_statistics['nonzero_weights'].append((self.perturbed_edge_weight > self.eps).sum().item())
        self.attack_statistics['probability_mass_update'].append(probability_mass_update)
        self.attack_statistics['probability_mass_projected'].append(probability_mass_projected)

    @staticmethod
    def cut_matrix_idx_to_triu_idx(matrix: torch.tensor) -> torch.Tensor:
        # cut all entries of matrix where the entry (x,y) holds x >= y
        row_idx = matrix[0]
        col_idx = matrix[1]
        mask = row_idx < col_idx
        #returns undirected triu matrix
        return matrix[:, mask]

    @staticmethod
    def pairs_to_linear_uppertri(pairs: torch.Tensor, N: int, *, drop_self_loops: bool = True) -> torch.Tensor:
        """
        Convert 2×b node index pairs into a (b,) vector of linear indices over the
        upper-triangular part (u < v) of an N×N matrix, row-major over (u,v).

        Mapping (u < v) -> k:
            k = u*(2*N - u - 1)//2 + (v - u - 1)
        Range: k ∈ [0, N*(N-1)//2)

        Args:
            pairs: LongTensor of shape (2, b) or (b, 2). Each pair is (u, v).
            N:     number of nodes.
            drop_self_loops: if True, removes u==v; if False, they’re filtered anyway by u<v.

        Returns:
            lin: LongTensor of shape (b_valid,), linear indices for valid u<v pairs.
        """
        # Accept either (2,b) or (b,2)
        if pairs.dim() != 2 or not (pairs.size(0) == 2 or pairs.size(1) == 2):
            raise ValueError("pairs must be shape (2, b) or (b, 2) of longs")
        if pairs.size(0) == 2:
            u, v = pairs[0], pairs[1]
        else:
            u, v = pairs[:, 0], pairs[:, 1]

        # Make undirected by ordering (u,v) -> (min,max)
        uu = torch.minimum(u, v)
        vv = torch.maximum(u, v)

        # Keep only strict upper-tri (uu < vv)
        mask = uu < vv
        if drop_self_loops:
            # already excluded by uu < vv, but keeps intent explicit
            pass

        if mask.sum() == 0:
            return torch.empty(0, dtype=torch.long, device=pairs.device)

        uu = uu[mask]
        vv = vv[mask]

        # Apply the closed-form linearization for the upper triangle
        # k = uu*(2N - uu - 1)//2 + (vv - uu - 1)
        N = int(N)
        lin = uu * (2 * N - uu - 1) // 2 + (vv - uu - 1)
        return lin

    @staticmethod
    def flip_matrix_idx_to_triu_idx(matrix: torch.tensor) -> torch.tensor:
        matrix = PRBCD.cut_diagonal_entries(matrix)
        row_idx = matrix[0]
        col_idx = matrix[1]
        # flip all entries of matrix where the entry (x,y) holds x>y
        mask = row_idx > col_idx
        temp = col_idx[mask]
        col_idx[mask] = row_idx[mask]
        row_idx[mask] = temp
        return matrix

    @staticmethod
    def cut_diagonal_entries(matrix: torch.tensor) -> torch.Tensor:
        row_idx = matrix[0]
        col_idx = matrix[1]
        mask = row_idx != col_idx
        # returns directed matrix (diagonal is cut)
        return matrix[:, mask]
