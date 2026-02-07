from enum import Enum

import numpy as np
import torch


class TargetNodesDrawMethod(Enum):
    STANDARD = "draw random edges from all possible edges"
    WEIGHTED_PROBABILITY = "draw nodes using weighted probabilities"
    RESTRICTED_NODE_SEARCH_SPACE = "draw nodes of a pre filtered node_search_space"

    def __init__(self, label):
        self.label = label


class Method(Enum):
    STANDARD = ("standard-PR-BCD", TargetNodesDrawMethod.STANDARD)
    SCORE = ("score", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)
    SCORE_RA = ("score_ra", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)
    SCORE_RD = ("score_rd", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)
    SCORE_SUM_RD_RA = ("score_rd+ra", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)
    SCORE_DEGREE = ("score_degree", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)
    SCORE_RA_DEGREE = ("score_ra_degree", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)
    SCORE_RD_DEGREE = ("score_rd_degree", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)
    SCORE_SUM_RD_RA_DEGREE = ("score_rd+ra_degree", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)
    REVERSED_SCORE_DEGREE_POW2 = ("high_degree^2_nodeset", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)
    DEGREE = ("degree", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)
    DEGREE_POW2 = ("degree^2", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)
    DEGREE_POW2_WITH_SCORE_FILTER = ("degree^2_with_score_finetune", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)
    GRID_BINARY_CLASS_11 = ("grid_binary_class_11", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)
    GRID_BINARY_CLASS_12 = ("grid_binary_class_12", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)
    GRID_BINARY_CLASS_21 = ("grid_binary_class_21", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)
    GRID_BINARY_CLASS_MEAN = ("grid_binary_class_mean of cells 11, 12, 21", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)
    GRID_BINARY_CLASS_P_HIGHEST_RA = ("grid_binary_class_prob_highest_ra", TargetNodesDrawMethod.WEIGHTED_PROBABILITY)

    GRID_RADII_11 = ("grid_radii_11", TargetNodesDrawMethod.RESTRICTED_NODE_SEARCH_SPACE)
    GRID_RADII_12 = ("grid_radii_12", TargetNodesDrawMethod.RESTRICTED_NODE_SEARCH_SPACE)
    GRID_RADII_21 = ("grid_radii_21", TargetNodesDrawMethod.RESTRICTED_NODE_SEARCH_SPACE)
    METHOD_1 = ("same as method 1", None)

    def __init__(self, label, target_nodes_draw_method):
        self.label = label
        self.targetNodesDrawMethod = target_nodes_draw_method

    def compute_node_probability(self, score, degrees, grid_binary_class, highest_ra):
        print()
        if self.targetNodesDrawMethod == TargetNodesDrawMethod.RESTRICTED_NODE_SEARCH_SPACE:
            print("WARNING: This method does not use node probability")
        else:
            ones = np.ones_like(score, dtype=float)
            node_probability = ones
            if self in MethodGroups.SCORE_BASED:
                node_probability = torch.from_numpy(ones / score)
            print("EXPECTED draw method: ", self.label)

            if self in MethodGroups.SCORE_DEGREE_BASED:
                print("ACTUAL draw Probability: 1/(score*degree)")
                node_probability = node_probability / degrees

            elif self in MethodGroups.DEGREE_POW2_BASED:
                print("ACTUAL draw Probability: 1/(degree^2)")
                node_probability = ones / degrees
                node_probability = torch.pow(node_probability, 2)

            elif self in MethodGroups.DEGREE_BASED:
                print("ACTUAL draw Probability: 1/(degree)")
                node_probability = ones / degrees

            elif self == Method.REVERSED_SCORE_DEGREE_POW2:
                print("ACTUAL draw method: reversed degree^2")
                node_probability = degrees
                node_probability = torch.pow(node_probability, 2)

            elif self == Method.GRID_BINARY_CLASS_P_HIGHEST_RA:
                print("ACTUAL draw Probability: to 1 - grid_binary_class(MAX(ra),0)")
                size = grid_binary_class.shape[0]
                grid_probabilities = grid_binary_class[
                    np.arange(size),
                    highest_ra - 1,
                    0
                ]
                node_probability = np.where(highest_ra != 0,
                                            ones - grid_probabilities,
                                            1)
            elif self == Method.GRID_BINARY_CLASS_11:
                print("ACTUAL draw probability: 1 - grid_binary_class cell [1,1]")
                node_probability = ones - grid_binary_class[:, 1, 1]
            elif self == Method.GRID_BINARY_CLASS_12:
                print("ACTUAL draw probability 1: - grid_binary_class cell [1,2]")
                node_probability = ones - grid_binary_class[:, 1, 2]
            elif self == Method.GRID_BINARY_CLASS_21:
                print("ACTUAL draw probability 1: - grid_binary_class cell [2,1]")
                node_probability = ones - grid_binary_class[:, 2, 1]
            elif self == Method.GRID_BINARY_CLASS_MEAN:
                mean_grid_binary_class = (grid_binary_class[:, 1, 1]
                            + grid_binary_class[:, 1, 2]
                            + grid_binary_class[:, 2, 1]) / 3
                print("ACTUAL draw probability 1: - mean_grid_binary_class of cells [1,1] [1,2] [2,1]")
                node_probability = ones - mean_grid_binary_class
            else:
                print("ACTUAL draw probability: 1/score")
            return node_probability


class MethodGroups:
    RA_BASED = {
        Method.SCORE_RA,
        Method.SCORE_RA_DEGREE,
    }

    RD_BASED = {
        Method.SCORE_RD,
        Method.SCORE_RD_DEGREE,
    }

    SUM_RA_AND_RD_BASED = {
        Method.SCORE_SUM_RD_RA,
        Method.SCORE_SUM_RD_RA_DEGREE,
    }

    SCORE_DEGREE_BASED = {
        Method.SCORE_DEGREE,
        Method.SCORE_RA_DEGREE,
        Method.SCORE_RD_DEGREE,
        Method.SCORE_SUM_RD_RA_DEGREE,
    }

    SCORE_BASED = {
        Method.SCORE,
        Method.SCORE_RA,
        Method.SCORE_RD,
        Method.SCORE_SUM_RD_RA,
        Method.SCORE_DEGREE,
        Method.SCORE_RA_DEGREE,
        Method.SCORE_RD_DEGREE,
        Method.SCORE_SUM_RD_RA_DEGREE,
        Method.REVERSED_SCORE_DEGREE_POW2,
        Method.DEGREE_POW2_WITH_SCORE_FILTER,
        Method.GRID_BINARY_CLASS_P_HIGHEST_RA,
    }

    DEGREE_POW2_BASED = {
        Method.DEGREE_POW2,
        Method.DEGREE_POW2_WITH_SCORE_FILTER,
    }

    DEGREE_BASED = {
        Method.DEGREE,
    }
