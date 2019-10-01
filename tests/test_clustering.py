from cops.clustering import *
import numpy as np


def test_to_frontier():
    G = Graph()
    # Add edges to graph
    connectivity_edges = [0, 1, 2, 3, 4, 5]
    transition_edges = [0, 1, 2, 3, 4, 5]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [4, 6, 7, 8]
    transition_edges = [4, 6, 7, 8]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [1, 9, 10, 11, 12, 13]
    transition_edges = [1, 9, 10, 11, 12, 13]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [9, 14, 15, 12]
    transition_edges = [9, 14, 15, 12]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [1, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    transition_edges = [1, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [8, 10]
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [15, 24]
    G.add_connectivity_path(connectivity_edges)
    # Set node positions
    node_positions = {
        0: (0, 0),
        1: (0, 1),
        2: (-2, 1),
        3: (-2, 2),
        4: (-3, 2),
        5: (-3, 1),
        6: (-3, 3),
        7: (-3.5, 3.5),
        8: (-2.5, 3.5),
        9: (0, 3),
        10: (-1.8, 3),
        11: (-1.6, 4),
        12: (0, 4),
        13: (0, 5),
        14: (1, 3),
        15: (1, 4),
        16: (1.5, 1),
        17: (2.5, 1.3),
        18: (4, 1.3),
        19: (5, 1.3),
        20: (5, 2),
        21: (4, 3),
        22: (5, 3),
        23: (5, 4),
        24: (3.5, 4),
    }

    G.set_node_positions(node_positions)
    frontiers = {6: 1, 13: 1, 22: 1}
    G.set_frontiers(frontiers)
    G.init_agents({0: 0, 1: 1, 2: 2, 3: 14})

    # CLUSTERING
    cp = ClusterProblem()
    cp.graph = G
    cp.k = 2  # number of desired clusters
    cp.master = 0
    cp.static_agents = [0]
    cp.solve_to_frontier_problem(soft=True)

    end_pos = set([cp.traj[r, cp.T_sol] for r in range(4)])
    num_front = len(end_pos & set(frontiers.keys()))

    np.testing.assert_array_less(3, num_front + 0.5)  # all three frontiers
