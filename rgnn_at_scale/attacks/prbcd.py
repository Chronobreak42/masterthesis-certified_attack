import logging

from collections import defaultdict
import math
from typing import Tuple, Optional

from tqdm import tqdm
import numpy as np
import torch
import torch_sparse
from torch_sparse import SparseTensor

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
                 **kwargs):
        super().__init__(**kwargs)

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

        self.current_search_space: torch.Tensor = None
        self.current_node_search_space: torch.Tensor = None
        self.sample_space: torch.Tensor = None
        self.edges_to_attack_index: torch.Tensor = torch.empty((2, 0), dtype=torch.long)
        self.modified_edge_index: torch.Tensor = None
        self.perturbed_edge_weight: torch.Tensor = None
        self.semi = None

        if self.make_undirected:
            self.n_possible_edges = self.n * (self.n - 1) // 2
        else:
            self.n_possible_edges = self.n ** 2  # We filter self-loops later

        self.lr_factor = lr_factor * max(math.log2(self.n_possible_edges / self.block_size), 1.)

    def _attack(self, n_perturbations, semi=False, use_cert="none", grid_radii: Optional[np.ndarray] = None, grid_binary_class: Optional[np.ndarray] = None, **kwargs):
        """Perform attack (`n_perturbations` is increasing as it was a greedy attack).

        Parameters
        ----------
        n_perturbations : int
            Number of edges to be perturbed (assuming an undirected graph)
        """
        self.semi = semi
        self.current_node_search_space = np.where(grid_radii[:, 2, 2] == False)[0]
        assert self.block_size > n_perturbations, \
            f'The search space size ({self.block_size}) must be ' \
            + f'greater than the number of permutations ({n_perturbations})'

        # For early stopping (not explicitly covered by pesudo code)
        best_accuracy = float('Inf')
        best_epoch = float('-Inf')

        # For collecting attack statistics
        self.attack_statistics = defaultdict(list)

        # Sample initial search space (Algorithm 1, line 3-4)
        if use_cert in ("sampling_grid_radii", "both_1"):
            self.sample_block_from_certificates_radii(grid_radii=grid_radii, n_perturbations=n_perturbations)
        elif use_cert in ("sampling_grid_binary_class", "both_2"):
            self.sample_block_from_certificates_binary_class(grid_binary_class=grid_binary_class, n_perturbations=n_perturbations)
        else:
            self.sample_random_block(n_perturbations)

        # Accuracy and attack statistics before the attack even started
        with torch.no_grad():
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
                    if use_cert in ("resampling", "both_1", "both_2"):
                        self.resample_random_block_from_cert_radii(grid_radii=grid_radii,
                                                                   n_perturbations=n_perturbations)  # TODO: Hier kommen die grid_radii rein.
                    else:
                        self.resample_random_block(n_perturbations) #TODO: Hier kommen die grid_radii rein.
                elif self.with_early_stopping and epoch == self.epochs_resampling - 1:
                    # Retreive best epoch if early stopping is active (not explicitly covered by pesudo code)
                    logging.info(
                        f'Loading search space of epoch {best_epoch} (accuarcy={best_accuracy}) for fine tuning\n')
                    self.current_search_space = best_search_space.to(self.device)
                    self.modified_edge_index = best_edge_index.to(self.device)
                    self.perturbed_edge_weight = best_edge_weight_diff.to(self.device)
                    self.perturbed_edge_weight.requires_grad = True

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

        # TODO: Don't we want to switch to returning things?

    def _get_logits(self, attr: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        return self.attacked_model(
            data=attr.to(self.device),
            adj=(edge_index.to(self.device), edge_weight.to(self.device))
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
                modified_edge_index, modified_edge_weight = self.modified_edge_index, self.perturbed_edge_weight
            edge_index = torch.cat((self.edge_index.to(self.device), modified_edge_index), dim=-1)
            edge_weight = torch.cat((self.edge_weight.to(self.device), modified_edge_weight))

            edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=self.n, n=self.n, op='sum')
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
                    modified_edge_index, modified_edge_weight = self.modified_edge_index, self.perturbed_edge_weight
                edge_index = torch.cat((self.edge_index.to(self.device), modified_edge_index), dim=-1)
                edge_weight = torch.cat((self.edge_weight.to(self.device), modified_edge_weight))

                edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=self.n, n=self.n, op='sum')
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
                self.perturbed_edge_weight
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
            #self.current_node_search_space = np.where(grid_radii[:, 2, 2] == False)[0] #moved this to the beginning

            #### TESTING ####
            #false_counts = np.sum(grid_radii == False, axis=(1, 2))
            #top_n_indices = np.argsort(-false_counts)[:n_perturbations]  # negative sign for descending order
            #self.current_node_search_space = top_n_indices
            ####^^^^ TESTING ^^^^
            # self.current_node_search_space = np.where(grid_radii[:, 2, 2] == False)[0]

            # draw edges: draw nodes from current_node_search_space and concatenate
            # First attempt
            if not self.semi:
                edges_idx = self.build_full_idx_matrix(False, self.block_size)
            else:
                self.build_full_idx_matrix_semi(False, self.block_size)
            # bis hierhin wird für die gesamte Blockmatrix gezogen

            # Second attempt with draw_undirected_matrix
            #edges_idx = PRBCD.draw_undirected_matrix(nodes_1, self.current_node_search_space)

            if self.make_undirected:
                # make undirected: cut all (x,y) where x >= y
                self.setup_search_space_undirected(self.n)
                # TODO: This will cut arbitrary number of entries, I made a function draw_undirected_matrix() to bypass this, but may be slow
                # maybe return later to this idea
            else:
                # TODO: i have not checked if it works for the directed case, I think it will NOT work
                self.modified_edge_index = PRBCD.cut_diagonal_entries(edges_idx)

            #self.current_search_space = PRBCD.full_to_linear_idx(self.n, self.modified_edge_index)
            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            if self.current_search_space.size(0) >= n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    def sample_block_from_certificates_binary_class(self, grid_binary_class, n_perturbations: int = 0):
        for _ in range(self.max_final_samples):

            sums = np.sum(grid_binary_class, axis=(1, 2))  # shape: (2810,)
            smallest_indices = np.argsort(sums)[:self.block_size]
            self.current_node_search_space = torch.from_numpy(smallest_indices)

            # draw edges: draw nodes from current_node_search_space and concatenate
            # First attempt
            if not self.semi:
                edges_idx = self.build_full_idx_matrix(False, self.block_size)
            else:
                self.build_full_idx_matrix_semi(False, self.block_size)
            # bis hierhin wird für die gesamte Blockmatrix gezogen

            if self.make_undirected:
                # make undirected: cut all (x,y) where x >= y
                self.setup_search_space_undirected(self.n)
            else:
                # TODO: i have not checked if it works for the directed case, I think it will NOT work
                self.modified_edge_index = PRBCD.cut_diagonal_entries(edges_idx)

            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            if self.current_search_space.size(0) >= n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    def resample_random_block(self, n_perturbations: int): #TODO: still work to be done
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
            lin_index = torch.randint(self.n_possible_edges, (n_edges_resample,), device=self.device)

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

    def resample_random_block_from_cert_radii(self, grid_radii, n_perturbations: int = 0): #TODO: still work to be done
        #self.current_node_search_space = np.where(grid_radii[:, 2, 2] == False)[0] #moved this to beginning
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

        # first we cut edges so that index build a triu matrix
        # (function triu_idx_to_linear only support triu matrix idx)
        # self.edges_to_attack_index = PRBCD.cut_matrix_idx_to_triu_idx(self.edges_to_attack_index)
        # we then build linear idx which is the current_search_space
        # self.current_search_space = PRBCD.triu_idx_to_linear_idx(n, self.edges_to_attack_index)
        # self.current_search_space = torch.unique(self.current_search_space, sorted=True)
        # The three code lines in this comment block are necessairy steps to keep this experimental approach conform with the rest of code
        # These could be outsourced as they are repeated in the while loop, may be repeated for resample aswell
        # there may be better approaches, at this time this works, so we keep this until it stops running
        # UPDATE: edges_to_current_search_space() is the outsourced method, lets see if this works

        while self.current_search_space.size(0) < self.block_size:
            # here we want to fill up current_search_space so that there are block_size many entries
            if not self.semi:
                self.build_full_idx_matrix(False, self.block_size)
            else:
                self.build_full_idx_matrix_semi(False, self.block_size)
            self.current_search_space = self.edges_to_current_search_space(n)
            #self.edges_to_attack_index = PRBCD.cut_matrix_idx_to_triu_idx(self.edges_to_attack_index)
            #self.current_search_space = PRBCD.triu_idx_to_linear_idx(n, self.edges_to_attack_index)
            #self.current_search_space = torch.unique(self.current_search_space, sorted=True)


        # now we follow same steps as sample_random_block
        #self.current_search_space = torch.unique(self.current_search_space, sorted=True)
        self.current_search_space = self.current_search_space[
                                        torch.randperm(self.current_search_space.size(0))[:self.block_size]
                                        ]
        self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
        return

    def edges_to_current_search_space(self, n: int):
        self.edges_to_attack_index = PRBCD.cut_matrix_idx_to_triu_idx(self.edges_to_attack_index)
        lin_idx = PRBCD.triu_idx_to_linear_idx(n, self.edges_to_attack_index)
        lin_idx = torch.unique(lin_idx, sorted=True)
        return lin_idx

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
    def cut_diagonal_entries(matrix: torch.tensor) -> torch.Tensor:
        row_idx = matrix[0]
        col_idx = matrix[1]
        mask = row_idx != col_idx
        # returns directed matrix (diagonal is cut)
        return matrix[:, mask]

    @staticmethod
    def draw_undirected_matrix(row_idx: torch.tensor, sample_node_space) -> torch.tensor:
        # this method takes a row_idx tensor and draws a col_idx from sample_node_space
        # because we want to draw an undirected matrix every entry (x,y) needs to hold x < y
        # this is a fast solution and can be improved:
        numpy_row_idx = row_idx.cpu().numpy()
        # in the following line we draw a node for each entry of row_idx, but we only draw nodes that holds the condition above
        # this line also has a problem when there is at least one node for which no other node holds sample_node_space > node
        # this means we would try to sample from an empty set
        # WE WILL SKIP THIS METHOD entirley unless we get a problem with the alternative way
        col_idx = torch.tensor([np.random.choice(sample_node_space[sample_node_space > node],
                                                 size=1, replace=True) for node in numpy_row_idx])
        # this ensures that we generate a triu undirected matrix
        row_idx = torch.from_numpy(row_idx)
        edges_idx = torch.cat([row_idx.unsqueeze(0), col_idx.T], dim=0)

        # CONCERNS: this method could be computational inefficient for large sample_node_space
        # I will not include this method yet
        return edges_idx
