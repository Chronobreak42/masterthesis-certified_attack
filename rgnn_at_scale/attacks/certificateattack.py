import logging
import math
import random
from collections import defaultdict
from typing import Tuple, Optional

import numpy as np
import torch
import torch_sparse
from torch_sparse import SparseTensor
from tqdm import tqdm
import experiments.certificatehelper as certificatehelper

from rgnn_at_scale.attacks.base_attack import Attack, SparseAttack
from rgnn_at_scale.attacks.modification_methods import Method, TargetNodesDrawMethod
from rgnn_at_scale.attacks.prbcd_sampling_modification import PRBCDSamplingModification
# from rgnn_at_scale.models import MODEL_TYPE
from rgnn_at_scale.helper import utils


class CertificateAttack(SparseAttack):
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
        self.current_node_search_space: torch.Tensor = None
        self.sample_space: torch.Tensor = None
        self.edges_to_attack_index: torch.Tensor = torch.empty((2, 0), dtype=torch.long)
        self.modified_edge_index: torch.Tensor = None
        self.perturbed_edge_weight: torch.Tensor = None
        self.draw_nodes_partly_from_method = None
        self.score = None
        self.degrees = None
        self.node_probability_1 = None
        self.node_probability_2 = None
        self.highest_ra = None
        self.highest_rd = None
        self.PRBCDSamplingModification = None

        if self.make_undirected:
            self.n_possible_edges = self.n * (self.n - 1) // 2
        else:
            self.n_possible_edges = self.n ** 2  # We filter self-loops later

        self.lr_factor = lr_factor * max(math.log2(self.n_possible_edges / self.block_size), 1.)

    def _attack(self, n_perturbations,
                method_to_use: Method = Method.STANDARD,
                method_for_second_nodeset: Method = Method.STANDARD,
                replace_sampling_method=False,
                replace_resampling_method=False,
                draw_nodes_partly_from_method=False,
                grid_radii: Optional[np.ndarray] = None,
                grid_binary_class: Optional[np.ndarray] = None,
                seed: Optional[int] = 1,
                **kwargs):
        """Perform attack (`n_perturbations` is increasing as it was a greedy attack).

        Parameters
        ----------
        n_perturbations : int
            Number of edges to be perturbed (assuming an undirected graph)
        """
        certificatehelper.setParams(seed)
        self.grid_radii = grid_radii
        self.grid_binary_class = grid_binary_class
        self.draw_nodes_partly_from_method = draw_nodes_partly_from_method
        self.PRBCDSamplingModification = PRBCDSamplingModification(self.n, self.device)

        # deprecated method storage
        self.method = method_to_use.label
        self.method_for_second_nodeset = method_for_second_nodeset.label

        # new
        self.method_for_target_nodes_1 = method_to_use
        self.method_for_target_nodes_2 = method_for_second_nodeset

        if self.method_for_target_nodes_1 != Method.STANDARD:
            self.score, self.highest_ra, self.highest_rd = self.PRBCDSamplingModification.compute_score(self.grid_radii,
                                                                                                        self.method_for_target_nodes_1)
        # TODO: degrees is only calculated once, can this be further improved by regular updates?
        self.degrees = torch.bincount(self.edge_index.flatten(), minlength=self.n)

        assert self.block_size > n_perturbations, \
            f'The search space size ({self.block_size}) must be ' \
            + f'greater than the number of permutations ({n_perturbations})'

        '''
        # Loop over the epochs (Algorithm 1, line 5)
        for epoch in tqdm(range(self.epochs)):
            self.perturbed_edge_weight.requires_grad = True

            # Retreive sparse perturbed adjacency matrix `A \oplus p_{t-1}` (Algorithm 1, line 6)
            # TODO: change get_modified_adj
            edge_index, edge_weight = self.get_modified_adj()

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Calculate logits for each node (Algorithm 1, line 6)
            logits = self._get_logits(self.attr, edge_index, edge_weight)
            # Calculate loss combining all each node (Algorithm 1, line 7)
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
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
                # TODO: Projection nicht unbedingt notwendig
                self.perturbed_edge_weight = Attack.project(
                    n_perturbations, self.perturbed_edge_weight, self.eps)
                # For monitoring
                probability_mass_projected = self.perturbed_edge_weight.sum().item()

                # Calculate accuracy after the current epoch (overhead for monitoring and early stopping)
                # TODO: change get_modified_adj
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
                    renew_certificate = False
                    if ((epoch + 1) % 10 == 0) and self.method not in ("standard-CertificateAttack",) and renew_certificate:
                        # TODO: use get_modified_adj()
                        certificatehelper.overwriteModel(self.perturbed_edge_weight)
                        print("Compute New Certificate")
                        self.grid_binary_class, self.grid_radii = certificatehelper.computeNewCertificate()
                        print("Computation finished successfully")
                        self.score = self.PRBCDSamplingModification.compute_score(self.grid_radii,
                                                                                  self.method_for_target_nodes_1)

                    if replace_resampling_method:
                        # old code:
                        if method_to_use in ("resampling_grid_radii",
                                             "both_1"):
                            print(method_to_use, "run RESAMPLING_grid_radii")
                            self.resample_random_block_from_cert_radii(grid_radii=self.grid_radii,
                                                                       n_perturbations=n_perturbations)
                        elif method_to_use in ("resampling_grid_binary_class",
                                               "both_2",
                                               "both_2_random", "low_certificate_values", "high_certificate_values"):
                            print(method_to_use, "run RESAMPLING_grid_binary_class")
                            self.resample_random_block_from_cert_binary_class(grid_binary_class=self.grid_binary_class,
                                                                              n_perturbations=n_perturbations)
                        # end olde code

                        else:
                            print("---------RESAMPLING---------", method_to_use, )
                            unique_grids, counts = np.unique(self.grid_binary_class, axis=0, return_counts=True)
                            print("Number of different certificate grids in grid_binary_class:", unique_grids.shape[0])

                            self.resample_random_block_with_new_method(n_perturbations=n_perturbations)
                    else:
                        print(method_to_use, "run RESAMPLING with standard random approach")
                        self.resample_random_block(n_perturbations)
                elif self.with_early_stopping and epoch == self.epochs_resampling - 1:
                    # Retreive best epoch if early stopping is active (not explicitly covered by pesudo code)
                    logging.info(
                        f'Loading search space of epoch {best_epoch} (accuarcy={best_accuracy}) for fine tuning\n')
                    self.current_search_space = best_search_space.to(self.device)
                    self.modified_edge_index = best_edge_index.to(self.device)
                    self.perturbed_edge_weight = best_edge_weight_diff.to(self.device)
                    self.perturbed_edge_weight.requires_grad = True
        '''

        # Retreive best epoch if early stopping is active (not explicitly covered by pesudo code)
        # if self.with_early_stopping:
        #    self.current_search_space = best_search_space.to(self.device)
        #    self.modified_edge_index = best_edge_index.to(self.device)
        #    self.perturbed_edge_weight = best_edge_weight_diff.to(self.device)

        # Sample final discrete graph (Algorithm 1, line 16)

        # For early stopping (not explicitly covered by pesudo code)
        best_accuracy = float('Inf')
        best_epoch = float('-Inf')

        # For collecting attack statistics
        self.attack_statistics = defaultdict(list)
        if self.method_for_target_nodes_1 == Method.STANDARD:
            self.sample_random_block(n_perturbations)
        else:
            edges_to_attack = torch.empty((2, 0), dtype=torch.long)
            self.current_search_space, self.modified_edge_index = self.get_edges_to_attack(n_perturbations,
                                                                                           edges_to_attack)
            while self.current_search_space.size(dim=0) < n_perturbations:
                # TODO: Teste hier das Ziehen von n_perturbations vielen samples und das droppen von zu vielen einträgen
                # TODO: Außerdem überlege schlaue zieh patterns
                self.current_search_space, self.modified_edge_index = self.get_edges_to_attack(
                                                                                n_perturbations,
                                                                                self.modified_edge_index)
                if self.current_search_space.size(dim=0) > n_perturbations:
                    self.current_search_space = self.current_search_space[:n_perturbations]
        self.perturbed_edge_weight = torch.ones_like(self.current_search_space, dtype=torch.float32)
        # Accuracy and attack statistics before the attack even started
        with torch.no_grad():

            logits = self._get_logits(self.attr, self.edge_index, self.edge_weight)
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
            accuracy = utils.accuracy(logits, self.labels, self.idx_attack)

            logging.info(f'\nBefore the attack - Loss: {loss.item()} Accuracy: {100 * accuracy:.3f} %\n')

            self._append_attack_statistics(loss.item(), accuracy, 0., 0.)

            del logits, loss

        #self.edge_index = self.addXOR(self.edge_index, self.modified_edge_index)
        # self.apply_edge_toggles(self.modified_edge_index)
        #self.edge_weight = torch.ones_like(self.edge_index[0], dtype=torch.float32)
        self.edge_index, self.edge_weight = self.get_modified_adj()
        self.attr_adversary = self.attr

        with torch.no_grad():

            logits = self._get_logits(self.attr, self.edge_index, self.edge_weight)
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
            accuracy = utils.accuracy(logits, self.labels, self.idx_attack)
            del logits
            self._append_attack_statistics(loss, accuracy, 0, 0)
        ###

        self.adj_adversary = SparseTensor.from_edge_index(
            self.edge_index,
            torch.ones_like(self.edge_index[0], dtype=torch.float32),
            (self.n, self.n)
        ).coalesce().detach()
        self.attr_adversary = self.attr

        # TODO: Don't we want to switch to returning things? Haha yeah me too

    def addXOR(self, edge_index, modified_edge_index):
        '''
        edge_index_sparse = SparseTensor.from_edge_index(
            edge_index,
            torch.ones_like(edge_index[0], dtype=torch.float32),
            (self.n, self.n)
        )

        modified_edge_index_sparse = SparseTensor.from_edge_index(
            modified_edge_index,
            torch.ones_like(modified_edge_index[0], dtype=torch.float32),
            (self.n, self.n)
        )

        summed_sparse = edge_index_sparse + modified_edge_index_sparse


        mask_index_duplicate_entry_clean = torch.ones(edge_index.size(1), dtype=torch.long)
        mask_index_duplicate_entry_modified = torch.ones(modified_edge_index.size(1), dtype=torch.long)
        index_duplicate_entry_clean = torch.empty(1, dtype=torch.long)
        index_duplicate_entry_modified = torch.empty(1, dtype=torch.long)

        for clean_edge_i in range(edge_index.shape[1]):
            for modified_edge_j in range(modified_edge_index.shape[1]):
                if (edge_index[:, clean_edge_i][0] == modified_edge_index[:, modified_edge_j][0] and
                        edge_index[:, clean_edge_i][1] == modified_edge_index[:, modified_edge_j][1]):
                    index_duplicate_entry_clean = torch.cat([index_duplicate_entry_clean, clean_edge_i], dim=0)
                    index_duplicate_entry_modified = torch.cat([index_duplicate_entry_modified, modified_edge_j], dim=0)

        mask_index_duplicate_entry_clean[index_duplicate_entry_clean] = 0
        mask_index_duplicate_entry_modified[index_duplicate_entry_modified] = 0

        new_edges = torch.cat(
            [edge_index[mask_index_duplicate_entry_clean],
             modified_edge_index[mask_index_duplicate_entry_modified]],
            dim=1)

        return new_edges
        '''
        e1 = edge_index.t()
        e2 = modified_edge_index.t()

        # Convert edges to tuples via hashing
        e1_set = {tuple(e.tolist()) for e in e1}
        e2_set = {tuple(e.tolist()) for e in e2}

        xor_set = e1_set ^ e2_set  # symmetric difference

        xor_edges = torch.tensor(list(xor_set), dtype=edge_index.dtype)
        return xor_edges.t()

    def apply_edge_toggles(self, perturbed_edges: torch.Tensor):
        """
        Toggles edges in self.edge_index using perturbed_edges.

        If an edge exists → remove it
        If it does not exist → add it

        perturbed_edges: (2, P)
        """

        # Canonicalize both
        edge_idx = self._canonicalize_edges(self.edge_index)
        perturbed_edges = self._canonicalize_edges(perturbed_edges)

        # Convert to set of tuples (CPU for hashing)
        edge_set = {
            (int(u), int(v)) for u, v in edge_idx.t().cpu().tolist()
        }

        for u, v in perturbed_edges.t().cpu().tolist():
            key = (int(u), int(v))
            if key in edge_set:
                edge_set.remove(key)  # remove existing edge
            else:
                edge_set.add(key)  # add new edge

        if len(edge_set) == 0:
            raise RuntimeError("All edges removed – graph became empty.")

        # Rebuild edge_idx
        new_edge_idx = torch.tensor(
            list(edge_set),
            device=self.edge_index.device,
            dtype=torch.long
        ).t().contiguous()

        self.edge_index = new_edge_idx

    def _canonicalize_edges(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Ensures a canonical representation for edge comparison.
        Undirected: (u, v) with u < v
        Directed: unchanged
        """
        if self.make_undirected:
            return torch.stack([
                torch.minimum(edge_index[0], edge_index[1]),
                torch.maximum(edge_index[0], edge_index[1]),
            ], dim=0)
        return edge_index

    def get_edges_to_attack(self, n_perturbations, edges_to_attack_as_matrix_idx):
        print()
        edges_idx = edges_to_attack_as_matrix_idx
        for _ in range(self.max_final_samples):

            self.current_node_search_space, self.PRBCDSamplingModification = self.setup_current_node_search_space()

            edges_idx = self.PRBCDSamplingModification.build_full_idx_matrix_method_dependent(
                sample_size=n_perturbations,
                method_for_target_nodes_1=self.method_for_target_nodes_1,
                method_for_target_nodes_2=self.method_for_target_nodes_2,
                target_edges_as_matrix_idx=edges_idx,
                draw_only_nodes_1_from_method=self.draw_nodes_partly_from_method,
                score=self.score,
                degrees=self.degrees,
                grid_binary_class=self.grid_binary_class,
                current_node_search_space=self.current_node_search_space,
                highest_ra=self.highest_ra
            )

            if self.make_undirected:
                # make undirected: cut all (x,y) where x >= y
                current_search_space = self.edges_as_matrix_idx_to_current_search_space(self.n, edges_idx)
                modified_edge_index = CertificateAttack.linear_to_triu_idx(self.n, current_search_space)

            else:
                # TODO: i have not checked if it works for the directed case, I think it will NOT work
                current_search_space = self.edges_as_matrix_idx_to_current_search_space(self.n, edges_idx)
                modified_edge_index = CertificateAttack.cut_diagonal_entries(edges_idx)

            self.perturbed_edge_weight = torch.full_like(
                current_search_space, 1, dtype=torch.float32, requires_grad=True
            )
            if current_search_space.size(0) >= n_perturbations:
                return current_search_space, modified_edge_index
        raise RuntimeError('Sampling random block was not successfully. Please decrease `n_perturbations`.')

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
                # Ensure all values are within [0, 1]
                perturbed_edge_weight[perturbed_edge_weight < 0] = 0.0
                perturbed_edge_weight[perturbed_edge_weight > 1] = 1.0

                # Replace any NaN or inf values
                perturbed_edge_weight = torch.nan_to_num(perturbed_edge_weight, nan=0.5, posinf=1.0, neginf=0.0)

                # Sample safely
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
                # self.n_possible_edges, (self.block_size,), device=self.device)
                self.n_possible_edges, (n_perturbations,), device=self.device)
            self.current_search_space = torch.unique(self.current_search_space, sorted=True)
            if self.make_undirected:
                self.modified_edge_index = CertificateAttack.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = CertificateAttack.linear_to_full_idx(self.n, self.current_search_space)
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            if self.current_search_space.size(0) >= n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    def sample_block_use_cert(self, grid_radii, grid_binary_class, n_perturbations: int = 0):
        for _ in range(self.max_final_samples):
            # Tried different grid_cells to use for. [1,1] and [2,2] showed best results (determined with 5 examples each)

            if self.method in ("sampling_grid_radii_alt_11",):
                print(self.method, "run sampling_grid_radii_alt_11")
                self.current_node_search_space = np.where(grid_radii[:, 1, 1] == False)[0]

            elif self.method in ("sampling_grid_radii", "both_1", "both_2_random"):
                print(self.method, "run sampling_grid_radii")
                self.current_node_search_space = np.where(grid_radii[:, 2, 2] == False)[0]

            elif self.method in ("sampling_grid_binary_class", "sampling_grid_binary_class_alt_11", "both_2"):
                print(self.method, "run sampling_grid_binary_class")
                print("using alternative current_node_search_space sampling")
                self.sample_current_node_search_space_det(grid_binary_class)

            elif self.method in ("low_certificate_values",):
                self.current_node_search_space = self.get_low_certificate_value_idx(grid_binary_class)

            elif self.method in ("high_certificate_values",):
                self.current_node_search_space = self.get_high_certificate_value_idx(grid_binary_class)

            # TODO: The following lines till setup_search_space_undirected could be improved in terms of readability, change method_to_use outputs
            if not self.draw_nodes_partly_from_method:
                edges_idx = self.build_full_idx_matrix(False, self.block_size)
            else:
                self.build_full_idx_matrix_semi(False, self.block_size)
            # bis hierhin wird für die gesamte Blockmatrix gezogen

            if self.make_undirected:
                # make undirected: cut all (x,y) where x >= y
                self.current_search_space = self.edges_to_current_search_space(self.n)
                self.modified_edge_index = CertificateAttack.linear_to_triu_idx(self.n, self.current_search_space)

            else:
                # TODO: i have not checked if it works for the directed case, I think it will NOT work
                self.modified_edge_index = CertificateAttack.cut_diagonal_entries(edges_idx)

            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            if self.current_search_space.size(0) >= n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    def sample_block_with_new_method(self, grid_radii, n_perturbations: int = 0):
        print()
        for _ in range(self.max_final_samples):

            self.current_node_search_space, self.PRBCDSamplingModification = self.setup_current_node_search_space()

            edges_idx = self.PRBCDSamplingModification.build_full_idx_matrix_method_dependent(
                sample_size=self.block_size,
                method_for_target_nodes_1=self.method_for_target_nodes_1,
                method_for_target_nodes_2=self.method_for_target_nodes_2,
                target_edges_as_matrix_idx=None,
                draw_only_nodes_1_from_method=self.draw_nodes_partly_from_method,
                score=self.score,
                degrees=self.degrees,
                grid_binary_class=self.grid_binary_class,
                current_node_search_space=self.current_node_search_space,
                highest_ra=self.highest_ra
            )

            if self.make_undirected:
                # make undirected: cut all (x,y) where x >= y
                self.current_search_space = self.edges_as_matrix_idx_to_current_search_space(self.n, edges_idx)
                self.modified_edge_index = CertificateAttack.linear_to_triu_idx(self.n, self.current_search_space)

            else:
                # TODO: i have not checked if it works for the directed case, I think it will NOT work
                self.modified_edge_index = CertificateAttack.cut_diagonal_entries(edges_idx)

            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            if self.current_search_space.size(0) >= n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfully. Please decrease `n_perturbations`.')

    def resample_random_block(self, n_perturbations: int):  # TODO: still work to be done
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
                self.modified_edge_index = CertificateAttack.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = CertificateAttack.linear_to_full_idx(self.n, self.current_search_space)

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

    def resample_random_block_from_cert_radii(self, grid_radii,
                                              n_perturbations: int = 0):  # TODO: still work to be done
        self.current_node_search_space = np.where(grid_radii[:, 2, 2] == False)[0]
        # self.current_node_search_space = np.where(grid_radii[:, 1, 1] == False)[0]
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
            if not self.draw_nodes_partly_from_method:
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
                self.modified_edge_index = CertificateAttack.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = CertificateAttack.linear_to_full_idx(self.n, self.current_search_space)

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

    def resample_random_block_from_cert_binary_class(self, grid_binary_class,
                                                     n_perturbations: int = 0):  # TODO: still work to be done
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

        if self.method in ("low_certificate_values",):
            self.current_node_search_space = self.get_low_certificate_value_idx(grid_binary_class)

        elif self.method in ("high_certificate_values",):
            self.current_node_search_space = self.get_high_certificate_value_idx(grid_binary_class)

        # Sample until enough edges were drawn
        for i in range(self.max_final_samples):
            n_edges_resample = self.block_size - self.current_search_space.size(0)

            # resample new edges
            if not self.draw_nodes_partly_from_method:
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
                self.modified_edge_index = CertificateAttack.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = CertificateAttack.linear_to_full_idx(self.n, self.current_search_space)

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

    def get_low_certificate_value_idx(self, grid_binary_class):
        print(self.method, "run sampling with low_certificate_values")
        grid_mean = grid_binary_class.mean(axis=(1, 2))
        grid_mean = np.asarray(grid_mean)  # ensure numpy
        index_at_half = len(grid_mean) // 2  # 50%

        lowest_half_idx = np.argsort(grid_mean)[:index_at_half]
        return lowest_half_idx

    def get_high_certificate_value_idx(self, grid_binary_class):
        print(self.method, "run sampling with high_certificate_values")
        grid_mean = grid_binary_class.mean(axis=(1, 2))
        grid_mean = np.asarray(grid_mean)  # ensure numpy
        index_at_half = len(grid_mean) // 2  # 50%

        highest_half_idx = np.argsort(grid_mean)[-index_at_half:]
        return highest_half_idx

    def resample_random_block_with_new_method(self, n_perturbations: int = 0):  # TODO: still work to be done
        print()
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
            self.current_node_search_space, self.PRBCDSamplingModification = self.setup_current_node_search_space()
            edges_idx = self.PRBCDSamplingModification.build_full_idx_matrix_method_dependent(
                sample_size=n_edges_resample,
                method_for_target_nodes_1=self.method_for_target_nodes_1,
                method_for_target_nodes_2=self.method_for_target_nodes_2,
                target_edges_as_matrix_idx=None,
                draw_only_nodes_1_from_method=self.draw_nodes_partly_from_method,
                score=self.score,
                degrees=self.degrees,
                grid_binary_class=self.grid_binary_class,
                current_node_search_space=self.current_node_search_space,
                highest_ra=self.highest_ra
            )

            lin_index = self.edges_as_matrix_idx_to_current_search_space(self.n, edges_idx)
            self.current_search_space, unique_idx = torch.unique(
                torch.cat((self.current_search_space, lin_index)),
                sorted=True,
                return_inverse=True
            )

            if self.make_undirected:
                self.modified_edge_index = CertificateAttack.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = CertificateAttack.linear_to_full_idx(self.n, self.current_search_space)

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

    def setup_current_node_search_space(self):
        print()
        print("Drawing Nodeset 1 with method: ", self.method_for_target_nodes_1.label)
        if self.draw_nodes_partly_from_method:
            print("Drawing Nodeset 2 with Standard Random Approach")
        else:
            print("Drawing Nodeset 2 with method: ", self.method_for_target_nodes_2.label)

        if self.method_for_target_nodes_1.targetNodesDrawMethod == TargetNodesDrawMethod.WEIGHTED_PROBABILITY:
            print(TargetNodesDrawMethod.WEIGHTED_PROBABILITY.label)
            print("current node search space: ALL NODES")
            current_node_search_space = range(self.n)
        else:
            print(TargetNodesDrawMethod.RESTRICTED_NODE_SEARCH_SPACE.label)
            if self.method_for_target_nodes_1 == Method.GRID_RADII_11:
                print("current node search space: all nodes where grid_radii[1,1] < 0.5")
                current_node_search_space = np.where(self.grid_radii[:, 1, 1] == False)[0]
            elif self.method_for_target_nodes_1 == Method.GRID_RADII_12:
                print("current node search space: all nodes where grid_radii[1,2] < 0.5")
                current_node_search_space = np.where(self.grid_radii[:, 1, 2] == False)[0]
            elif self.method_for_target_nodes_1 == Method.GRID_RADII_21:
                print("current node search space: all nodes where grid_radii[2,1] < 0.5")
                current_node_search_space = np.where(self.grid_radii[:, 2, 1] == False)[0]
            else:
                print("ERROR: something went wrong in setting up current node search space ")
                current_node_search_space = None
        return current_node_search_space, PRBCDSamplingModification(self.n, self.device)

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

    def edges_to_current_search_space(self, n: int):
        # first we cut edges so that index build a triu matrix
        # (function triu_idx_to_linear only support triu matrix idx)
        self.edges_to_attack_index = CertificateAttack.flip_matrix_idx_to_triu_idx(self.edges_to_attack_index)
        # we then build linear idx which is the current_search_space
        lin_idx = CertificateAttack.triu_idx_to_linear_idx(n, self.edges_to_attack_index)
        lin_idx = torch.unique(lin_idx, sorted=True)
        return lin_idx

    def edges_as_matrix_idx_to_current_search_space(self, n: int, edges_as_matrix_idx):
        # first we cut edges so that index build a triu matrix
        # (function triu_idx_to_linear only support triu matrix idx)
        edges_as_triu_matrix_idx = CertificateAttack.flip_matrix_idx_to_triu_idx(edges_as_matrix_idx)
        # we then build linear idx which is the current_search_space
        lin_idx = CertificateAttack.triu_idx_to_linear_idx(n, edges_as_triu_matrix_idx)
        lin_idx = torch.unique(lin_idx, sorted=True)
        return lin_idx

    def sample_current_node_search_space_det(self, grid_binary_class):
        if self.method in ("sampling_grid_binary_class_alt_11",):
            print("running alternative sampling with grid binary class (1,1)")
            unrobust_nodes_lin_index = np.where(grid_binary_class[:, 1, 1] < 0.3)  # shape: (2810,)
        else:
            print("running alternative sampling with grid binary class (2,2)")
            unrobust_nodes_lin_index = np.where(grid_binary_class[:, 2, 2] < 0.2)
        unrobust_nodes_lin_index = unrobust_nodes_lin_index[0]
        self.current_node_search_space = torch.unique(torch.from_numpy(unrobust_nodes_lin_index), sorted=False)
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
        # returns undirected triu matrix
        return matrix[:, mask]

    @staticmethod
    def flip_matrix_idx_to_triu_idx(matrix: torch.tensor) -> torch.tensor:
        matrix = CertificateAttack.cut_diagonal_entries(matrix)
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
