from graph_connectivity.graph import *
import numpy as np

def test_pre_S():
    G = Graph()
    connectivity_edges = [0,1,2,3] #directed connectivity path (one way)
    transition_edges = [0,1,2,3]   #directed transition path (one way)
    #Add edges to graph
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)

    S_v_t = set([(1,0), (0,1)])

    np.testing.assert_equal(G.pre_tran_vt(S_v_t),
                            set([(0,0,0)]))

    np.testing.assert_equal(G.pre_conn_vt(S_v_t),
                            set([(0,1,0), (2,1,0), (1,0,1)]))

def test_pre_S2():
    G = Graph()
    connectivity_edges = [0,1,2,3] #directed connectivity path (one way)
    transition_edges = [0,1,2,3]   #directed transition path (one way)
    #Add edges to graph
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)

    S_v_t = set([(1, 0), (2, 1), (0, 2), (1, 2), (3, 2)])

    # np.testing.assert_equal(G.pre_tran_vt(S_v_t),
    #                         set([(0,0,0)]))

    np.testing.assert_equal(G.pre_conn_vt(S_v_t),
                            set([(0, 1, 0), (2, 1, 0),
                                 (1, 2, 1), (3, 2, 1),
                                 (2, 1, 2), (2, 3, 2)]))
    np.testing.assert_equal(G.pre_tran_vt(S_v_t),
                            set([(3, 2, 0), (0, 1, 1),
                                 (0, 0, 1), (1, 0, 1),
                                 (2, 2, 0), (3, 3, 1), (1, 1, 1)]))


def test_pre():

    G = Graph()

    G.add_path([0,1,2,3], type='transition')
    G.add_path([3, 2], type='connectivity')
    G.add_path([1, 0], type='connectivity')

    print(G.edges(data=True))

    np.testing.assert_equal(G.pre_tran([2,3]), set([1,2]))
    np.testing.assert_equal(G.pre_tran([2]), set([1]))    

    np.testing.assert_equal(G.post_tran([2,3]), set([3]))
    np.testing.assert_equal(G.post_tran([2]), set([3]))        


    np.testing.assert_equal(G.pre_conn([2]), set([3]))
    np.testing.assert_equal(G.pre_conn([0,2]), set([1,3]))    

    np.testing.assert_equal(G.post_conn([0,1,2,3]), set([2,0]))
