from cops.clustering import *
from cops.animate import *

from graph_examples import get_medium_graph

G = get_medium_graph()

frontiers = {6: 1, 13: 1, 22: 1}
G.set_frontiers(frontiers)

# Set initial position of agents
agent_positions = {0: 0, 1: 1, 2: 2, 3: 9}
G.init_agents(agent_positions)

master = 0
static_agents = [0]

# CLUSTERING
cp = ClusterProblem()
cp.graph = G
cp.num_clusters = 2  # number of clusters to create
cp.master = master
cp.static_agents = static_agents
tofront = cp.solve_to_frontier_problem(soft=True)

animate_cluster(G, cp.traj, cp.conn, tofront.cs.subgraphs)
