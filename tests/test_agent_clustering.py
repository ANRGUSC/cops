from cops.clustering import *
import numpy as np


def test_agent_clustering():
    G = Graph()
    G.add_transition_path(list(range(0, 50)))
    G.add_connectivity_path(list(range(0, 50)))

    agent_positions = {0: 0, 1: 1, 2: 35, 3: 30, 4: 25}
    G.init_agents(agent_positions)

    cp = ClusterProblem()
    cp.graph = G
    cp.static_agents = [0]

    cp.prepare_problem(remove_dead=False)

    agent_clusters = agent_clustering(cp, 2)

    np.testing.assert_equal({0, 1} in map(set, agent_clusters.values()), True)
    np.testing.assert_equal({2, 3, 4} in map(set, agent_clusters.values()), True)


def test_agent_clustering():
    G = Graph()
    G.add_transition_path(list(range(0, 25)))
    G.add_transition_path(list(range(26, 50)))
    G.add_transition_path([0, 26])

    agent_positions = {0: 0, 1: 3, 2: 27, 3: 20, 4: 24, 5: 42, 6: 47}
    G.init_agents(agent_positions)

    cp = ClusterProblem()
    cp.graph = G
    cp.static_agents = [0]

    cp.prepare_problem(remove_dead=False)

    agent_clusters = agent_clustering(cp, 3)

    np.testing.assert_equal({0, 1, 2} in map(set, agent_clusters.values()), True)
    np.testing.assert_equal({3, 4} in map(set, agent_clusters.values()), True)
    np.testing.assert_equal({5, 6} in map(set, agent_clusters.values()), True)
