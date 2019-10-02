import numpy as np

from cops.graph import Graph
from cops.clustering import (
    ClusterProblem,
    ClusterStructure,
    agent_clustering,
    inflate_agent_clusters,
    kill_largest_frontiers,
)


def test_strategy2():

    G = Graph()
    G.add_transition_path(list(range(0, 25)))
    G.add_transition_path(list(range(26, 50)))
    G.add_connectivity_path(list(range(0, 25)))
    G.add_connectivity_path(list(range(26, 50)))
    G.add_transition_path([0, 26])
    G.set_frontiers({24: 1, 49: 1})

    agent_positions = {0: 0, 1: 3, 2: 27, 3: 6, 4: 8, 5: 42, 6: 47}
    G.init_agents(agent_positions)

    cp = ClusterProblem()
    cp.graph = G
    cp.static_agents = [0]
    cp.master = 0
    cp.max_problem_size = 1

    cp.prepare_problem(remove_dead=False)

    cs = ClusterStructure(agent_clusters=agent_clustering(cp, 3))

    cs = inflate_agent_clusters(cp, cs)
    kill_list = kill_largest_frontiers(cp, cs)

    np.testing.assert_equal(kill_list, set(range(48, 50)) | set(range(9, 25)))
