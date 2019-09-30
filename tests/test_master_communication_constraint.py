from cops.problem import *
import numpy as np

def test_master_comm():

    n = 4  # size of graph

    # Define a connectivity graph
    G = Graph()
    G.add_transition_path(list(range(n)))
    G.add_connectivity_path(list(range(n)))
    G.set_node_positions({i: (0,i) for i in range(n)})

    frontiers = {1: 1}
    G.set_frontiers(frontiers)

    agent_positions = {0: 0, 1: 2, 2: 3}
    G.init_agents(agent_positions)

    # Set up the connectivity problem
    cp = ConnectivityProblem()
    cp.graph = G
    cp.T = 5
    cp.static_agents = []
    cp.master = 0
    cp.src = [2]
    cp.snk = [1]

    #Solve
    cp.solve_flow(optimal = True, frontier_reward = True, master = True, connectivity = True, verbose = True, cut = False)

    # positions of robots
    np.testing.assert_equal(cp.traj[0,5], 1)
    np.testing.assert_equal(cp.traj[1,5], 1)
    np.testing.assert_equal(cp.traj[2,5], 1)
    np.testing.assert_equal(cp.conn[1] , set([(1,2,'master'),(2,3,'master'),(3,2,2)]))
