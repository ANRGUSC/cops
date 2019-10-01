from cops.problem import *
import numpy as np

def test_master_comm():

    n = 4  # size of graph

    # Define a connectivity graph
    G = Graph()
    G.add_transition_path(list(range(n)))
    G.add_connectivity_path(list(range(n)))
    G.set_node_positions({i: (0,i) for i in range(n)})
    G.set_frontiers({1:1})
    G.init_agents({0: 0, 1: 2, 2: 3})

    # Set up the connectivity problem
    cp = ConnectivityProblem()
    cp.graph = G
    cp.T = 5
    cp.static_agents = []
    cp.master = 0
    cp.src = [2]
    cp.snk = [1]


    # Solve with master, without frontier rewards
    cp.solve_flow(optimal=True, frontier_reward=False, master=True, connectivity=True, cut=False)

    # Check solution
    for t in range(6):
        np.testing.assert_equal(cp.traj[0,t], 0 if t==0 else 1)
        np.testing.assert_equal(cp.traj[1,t], 2)
        np.testing.assert_equal(cp.traj[2,t], 3)

        np.testing.assert_equal(cp.conn[t], set([(1,2,'master'),(2,3,'master'),(3,2,2)]) if t==1 else set())

    # Solve with master, with frontier rewards
    cp.solve_flow(optimal=True, frontier_reward=True, master=True, connectivity=True, cut=False)

    # Check solution
    for t in range(6):
        np.testing.assert_equal(cp.traj[0,t], 0 if t==0 else 1)
        np.testing.assert_equal(cp.traj[1,t], 2 if t<2 else 1)
        np.testing.assert_equal(cp.traj[2,t], [3, 3, 2, 1, 1, 1][t])

        np.testing.assert_equal(cp.conn[t], set([(1,2,'master'),(2,3,'master')]) if t==1 else set())

    # Solve without master, without frontier rewards
    cp.solve_flow(optimal=True, frontier_reward=False, master=False, connectivity=True, cut=False)

    # Check solution
    for t in range(6):
        np.testing.assert_equal(cp.traj[0,t], 0)
        np.testing.assert_equal(cp.traj[1,t], 2)
        np.testing.assert_equal(cp.traj[2,t], 3)

        np.testing.assert_equal(cp.conn[t], set([(3,2,2)]) if t==0 else set())

    # Solve without master, with frontier rewards
    cp.solve_flow(optimal=True, frontier_reward=True, master=False, connectivity=True, cut=False)

    # Check solution
    for t in range(6):
        np.testing.assert_equal(cp.traj[0,t], 0 if t==0 else 1)
        np.testing.assert_equal(cp.traj[1,t], 2 if t<1 else 1)
        np.testing.assert_equal(cp.traj[2,t], [3, 2, 1, 1, 1, 1][t])
        np.testing.assert_equal(cp.conn[t], set())


