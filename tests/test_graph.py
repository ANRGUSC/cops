from graph_connectivity.def_ilp import *
import numpy as np

def test_pre_S():
    G = Graph()
    connectivity_edges = [0,1,2,3] #directed connectivity path (one way)
    transition_edges = [0,1,2,3]   #directed transition path (one way)
    #Add edges to graph
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)

    S_v_t = set([(1,0), (0,1)])

    np.testing.assert_equal(G.get_pre_S_transition(S_v_t),
                            set([(0,0,0)]))

    np.testing.assert_equal(G.get_pre_S_connectivity(S_v_t),
                            set([(0,1,0), (2,1,0), (1,0,1)]))

def test_pre_S2():
    G = Graph()
    connectivity_edges = [0,1,2,3] #directed connectivity path (one way)
    transition_edges = [0,1,2,3]   #directed transition path (one way)
    #Add edges to graph
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)

    S_v_t = set([(1, 0), (2, 1), (0, 2), (1, 2), (3, 2)])

    # np.testing.assert_equal(G.get_pre_S_transition(S_v_t),
    #                         set([(0,0,0)]))

    np.testing.assert_equal(G.get_pre_S_connectivity(S_v_t),
                            set([(0, 1, 0), (2, 1, 0),
                                 (1, 2, 1), (3, 2, 1),
                                 (2, 1, 2), (2, 3, 2)]))
    np.testing.assert_equal(G.get_pre_S_transition(S_v_t),
                            set([(3, 2, 0), (0, 1, 1),
                                 (0, 0, 1), (1, 0, 1),
                                 (2, 2, 0), (3, 3, 1), (1, 1, 1)]))
