import random

import numpy as np
import torch

from rgnn_at_scale.attacks.modification_methods import Method, TargetNodesDrawMethod, MethodGroups


class PRBCDSamplingModification:
    def __init__(self, n, device):
        self.n = n
        self.device = device
        self.draw_probability_for_target_nodes_1 = None
        self.draw_probability_for_target_nodes_2 = None
        self.highest_ra = None
        self.highest_rd = None

    def compute_score(self, certificate, method: Method):
        print()
        if method not in MethodGroups.SCORE_BASED:
            print("This method is not score-based. Skip score computation.")
            return None, None, None
            
        _, num_rows, num_cols = certificate.shape
        row_mask = certificate.any(axis=2)
        col_mask = certificate.any(axis=1)

        last_row_index = row_mask[:, ::-1].argmax(axis=1)
        last_row_index = num_rows - last_row_index  # Because we reversed

        last_col_index = col_mask[:, ::-1].argmax(axis=1)
        last_col_index = num_cols - last_col_index  # Because we reversed
        self.highest_ra = last_row_index
        self.highest_rd = last_col_index
        # handle cases with no 1s at all
        # so lowest score is 1/ trying 0.5
        last_row_index[~row_mask.any(axis=1)] = 1  # so this case should never occur
        last_col_index[~col_mask.any(axis=1)] = 1
        score = last_row_index * last_col_index
        # max_radii = np.stack([last_row_index, last_col_index], axis=1)
        print("EXPECTED: Using ", method.label, "as SCORE")
        if method in MethodGroups.RA_BASED:
            print("ACTUAL: using ra as SCORE")
            score = last_row_index
        elif method in MethodGroups.RD_BASED:
            print("ACTUAL: using rd as SCORE")
            score = last_col_index
        elif method in MethodGroups.SUM_RA_AND_RD_BASED:
            print("ACTUAL: using rd+ra as SCORE")
            score = last_col_index + last_col_index
        else:
            print("ACTUAL: using rd*ra as SCORE")
        return score, self.highest_ra, self.highest_rd

    def draw_target_edges_from_restricted_node_search_space(self, draw_only_nodes_1_with_method: bool,
                                                            target_edges_as_matrix_idx, current_node_search_space,
                                                            sample_size: int):
        print()
        print("---------Running: draw_target_edges_from_restricted_node_search_space")
        print("Drawing nodeset 1 from restricted node search space")
        nodes_1 = torch.from_numpy(np.random.choice(current_node_search_space, size=sample_size, replace=True))
        if draw_only_nodes_1_with_method:
            print("Drawing nodeset 1 from RANDOMLY from ALL NODES")
            nodes_2 = torch.randint(self.n, (sample_size,), device=self.device)
        else:
            print("Drawing nodeset 2 from restricted node search space")
            nodes_2 = torch.from_numpy(np.random.choice(current_node_search_space, size=sample_size, replace=True))
        new_target_edges_as_matrix_idx = torch.cat([nodes_1.unsqueeze(0), nodes_2.unsqueeze(0)], dim=0)
        new_target_edges_as_matrix_idx = torch.cat([new_target_edges_as_matrix_idx, target_edges_as_matrix_idx], dim=1)
        return new_target_edges_as_matrix_idx

    def build_full_idx_matrix_method_dependent(self, sample_size: int,
                                               method_for_target_nodes_1: Method,
                                               method_for_target_nodes_2: Method,
                                               target_edges_as_matrix_idx,
                                               draw_only_nodes_1_from_method,
                                               score,
                                               degrees,
                                               grid_binary_class,
                                               current_node_search_space,
                                               highest_ra):
        print()
        if target_edges_as_matrix_idx is None:
            # empty target_edges_as_matrix_idx for resample or other cases
            target_edges_as_matrix_idx = torch.empty((2, 0), dtype=torch.long)

        new_target_edges_as_matrix_idx = torch.empty((2, 0), dtype=torch.long)
        if method_for_target_nodes_1.targetNodesDrawMethod == TargetNodesDrawMethod.RESTRICTED_NODE_SEARCH_SPACE:
            new_target_edges_as_matrix_idx = self.draw_target_edges_from_restricted_node_search_space(
                draw_only_nodes_1_with_method=draw_only_nodes_1_from_method,
                target_edges_as_matrix_idx=target_edges_as_matrix_idx,
                current_node_search_space=current_node_search_space,
                sample_size=sample_size)

        elif method_for_target_nodes_1.targetNodesDrawMethod == TargetNodesDrawMethod.WEIGHTED_PROBABILITY:
            new_target_edges_as_matrix_idx = self.draw_target_edges_with_node_probability(
                sample_size=sample_size,
                method_for_target_nodes_1=method_for_target_nodes_1,
                method_for_target_nodes_2=method_for_target_nodes_2,
                target_edges_as_matrix_idx=target_edges_as_matrix_idx,
                draw_only_nodes_1_from_method=draw_only_nodes_1_from_method,
                score=score,
                degrees=degrees,
                grid_binary_class=grid_binary_class,
                current_node_search_space=current_node_search_space,
                highest_ra=highest_ra)

        # TODO: see if following line is needed anymore right now it is redundant, all build_methods_ have this
        new_target_edges_as_matrix_idx = torch.cat([new_target_edges_as_matrix_idx, target_edges_as_matrix_idx], dim=1)
        return new_target_edges_as_matrix_idx

    def draw_target_edges_with_node_probability(self, sample_size: int,
                                                method_for_target_nodes_1: Method,
                                                method_for_target_nodes_2: Method,
                                                target_edges_as_matrix_idx,
                                                draw_only_nodes_1_from_method,
                                                score,
                                                degrees,
                                                grid_binary_class,
                                                current_node_search_space,
                                                highest_ra):
        print()
        print("---------Running: draw_target_edges_with_node_probability")
        if self.draw_probability_for_target_nodes_1 is None:
            print("Calculating node_probability_1")
            self.draw_probability_for_target_nodes_1 = method_for_target_nodes_1.compute_node_probability(
                score=score,
                degrees=degrees,
                grid_binary_class=grid_binary_class,
                highest_ra=highest_ra)

        print("Drawing nodeset 1:")
        nodes_1 = self.draw_nodes(
            method_for_target_nodes=method_for_target_nodes_1,
            current_node_search_space=current_node_search_space,
            weights=self.draw_probability_for_target_nodes_1,
            sample_size=sample_size)

        if draw_only_nodes_1_from_method:
            print("Drawing nodeset 2 RANDOMLY from ALL NODES")
            nodes_2 = torch.randint(self.n, (sample_size,), device=self.device)

        else:
            if self.draw_probability_for_target_nodes_2 is None:
                print("Calculating node_probability_2")
                if method_for_target_nodes_2 == Method.METHOD_1:
                    print("Using first method for nodeset 2 draw probability")
                    self.draw_probability_for_target_nodes_2 = self.draw_probability_for_target_nodes_1
                else:
                    self.draw_probability_for_target_nodes_2 = method_for_target_nodes_2.compute_node_probability(
                        score=score,
                        degrees=degrees,
                        grid_binary_class=grid_binary_class,
                        highest_ra=highest_ra)
            print("Drawing nodeset 2:")
            nodes_2 = self.draw_nodes(method_for_target_nodes=method_for_target_nodes_2,
                                      current_node_search_space=current_node_search_space,
                                      weights=self.draw_probability_for_target_nodes_2,
                                      sample_size=sample_size)

        # drop half the nodes based on score
        nodes_1 = self.drop_nodes(nodeset=nodes_1, method_for_target_nodes=method_for_target_nodes_1, score=score)
        nodes_2 = self.drop_nodes(nodeset=nodes_2, method_for_target_nodes=method_for_target_nodes_1, score=score)

        new_target_edges_as_matrix_idx = torch.cat([nodes_1.unsqueeze(0), nodes_2.unsqueeze(0)], dim=0)
        new_target_edges_as_matrix_idx = torch.cat([target_edges_as_matrix_idx, new_target_edges_as_matrix_idx], dim=1)
        return new_target_edges_as_matrix_idx

    def draw_nodes(self, method_for_target_nodes: Method, current_node_search_space, weights, sample_size):
        print()
        if method_for_target_nodes.label == Method.DEGREE_POW2_WITH_SCORE_FILTER:
            # I want to draw more samples but discard one half using a score-based decision function
            used_sample_size = sample_size * 2
        else:
            used_sample_size = sample_size

        print("ACTUAL method for draw probability: ", method_for_target_nodes.label)
        nodes_drawn = torch.tensor(
            random.choices(current_node_search_space, weights=weights, k=used_sample_size))
        return nodes_drawn

    def drop_nodes(self, method_for_target_nodes: Method, nodeset, score):
        print()
        if method_for_target_nodes.label == Method.DEGREE_POW2_WITH_SCORE_FILTER:
            print("Filter nodes using score")
            scores_nodeset = score[nodeset]
            ### Dropping half the nodes while preserving original order
            k = len(nodeset) // 2
            # Indices of the k smallest scores (unordered)
            keep_idx = np.argpartition(scores_nodeset, k)[:k]
            # Sort those indices to restore original order
            keep_idx = np.sort(keep_idx)
            # Select nodes, preserving original order
            nodeset_reduced = nodeset[keep_idx]
            return nodeset_reduced
        else:
            return nodeset
