import numpy as np
from cops.graph import Graph
from cops.clustering import ClusterProblem, ClusterStructure, inflate_agent_clusters


def test_activationinflate1():
    G = Graph()

    G.add_connectivity_path([0, 1])
    G.add_connectivity_path([0, 2])

    agent_positions = {0: 0, 1: 1, 2: 2}
    G.init_agents(agent_positions)

    cs = ClusterStructure(agent_clusters={"c0": [0], "c1": [1], "c2": [2]})
    master = 0

    cp = ClusterProblem()
    cp.graph = G
    cp.master = master
    cp.prepare_problem(remove_dead=False)

    cs = inflate_agent_clusters(cp, cs)

    np.testing.assert_equal(cs.subgraphs["c0"], set([0]))
    np.testing.assert_equal(cs.subgraphs["c1"], set([1]))
    np.testing.assert_equal(cs.subgraphs["c2"], set([2]))

    np.testing.assert_equal(cs.child_clusters["c0"], {("c1", 1), ("c2", 2)})
    np.testing.assert_equal(cs.child_clusters["c1"], set())
    np.testing.assert_equal(cs.child_clusters["c2"], set())

    np.testing.assert_equal(cs.parent_clusters["c1"], ("c0", 0))
    np.testing.assert_equal(cs.parent_clusters["c2"], ("c0", 0))


def test_inflate2():
    G = Graph()
    G.add_transition_path(list(range(0, 12)))

    G.add_connectivity_path(list(range(0, 12)))
    G.add_connectivity_path([6, 8])

    agent_positions = {0: 0, 1: 1, 2: 4, 3: 6, 4: 8, 5: 10}
    G.init_agents(agent_positions)

    cs = ClusterStructure(agent_clusters={"c0": [0, 1], "c1": [2, 3], "c2": [4, 5]})
    master = 0

    cp = ClusterProblem()
    cp.graph = G
    cp.master = master
    cp.prepare_problem(remove_dead=False)

    cs = inflate_agent_clusters(cp, cs)

    np.testing.assert_equal(cs.subgraphs["c0"], set([0, 1, 2, 3]))
    np.testing.assert_equal(cs.subgraphs["c1"], set([4, 5, 6, 7]))
    np.testing.assert_equal(cs.subgraphs["c2"], set([8, 9, 10, 11]))

    np.testing.assert_equal(cs.child_clusters["c0"], {("c1", 4)})
    np.testing.assert_equal(cs.child_clusters["c1"], {("c2", 8)})
    np.testing.assert_equal(cs.child_clusters["c2"], set())

    np.testing.assert_equal(cs.parent_clusters["c1"], ("c0", 3))
    np.testing.assert_equal(cs.parent_clusters["c2"], ("c1", 6))
