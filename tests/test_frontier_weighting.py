import numpy as np

from cops.graph import Graph
from cops.problem import ConnectivityProblem

def import_gurobi():
    try:
        import gurobipy
        return True
    except ModuleNotFoundError as e:
        return False

def test_frontier_weighting():

    n = 4

    # Define a connectivity graph
    G = Graph()
    G.add_transition_path(list(range(n)))
    G.add_connectivity_path(list(range(n)))
    G.set_node_positions({i: (0, i) for i in range(n)})

    frontiers = {0: 1, 1: 1, 2: 4, 3: 1}
    G.set_frontiers(frontiers)

    agent_positions = {0: 0, 1: 2, 2: 3}
    G.init_agents(agent_positions)

    # Set up the connectivity problem
    cp = ConnectivityProblem()
    cp.graph = G
    cp.T = 6
    cp.static_agents = []
    cp.master = 0
    cp.src = [2]
    cp.snk = [1]

    # Solve
    if import_gurobi():
        cp.solve_flow(master=True, frontier_reward=True, connectivity=True, cut=True)

        # Check solution

        for t in range(3):
            np.testing.assert_equal(cp.traj[0, t], [0, 1, 2 ,2][t])
            np.testing.assert_equal(cp.traj[1, t], 2)
            np.testing.assert_equal(cp.traj[2, t], 3)
            np.testing.assert_equal(cp.conn[0], set())
            np.testing.assert_equal(cp.conn[1], set())
            np.testing.assert_equal(cp.conn[2], set([(2, 3, "master"), (3, 2, 2)]))
