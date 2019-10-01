from cops.problem import *
from cops.animate import *

from graph_examples import get_medium_graph

G = get_medium_graph()

# Set small nodes
small_nodes = [1]
G.set_small_node(small_nodes)

frontiers = {2: 1, 14: 3}
G.set_frontiers(frontiers)

# Set initial position of agents
agent_positions = {0: 0, 1: 1, 2: 1}  # agent:position
G.init_agents(agent_positions)

# Set up the connectivity problem
cp = ConnectivityProblem()
cp.graph = G  # graph
cp.T = 7  # time
cp.master = [0]  # master_agent
cp.static_agents = [0]  # static agents
cp.big_agents = [0, 1, 2]

# Define sources and sinks as subsets of agents
cp.src = []
cp.snk = []

# Solve
cp.solve_flow(master=True, connectivity=True)

# Animate solution
animate(G, cp.traj, cp.conn)
