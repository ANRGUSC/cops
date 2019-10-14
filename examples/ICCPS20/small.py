from cops.problem import ConnectivityProblem
from cops.animate import animate

from examples.graph_examples import get_small_graph

G = get_small_graph()

# Set small nodes
small_nodes = [G.nodes]
G.set_small_node(small_nodes)

frontiers = {}
G.set_frontiers(frontiers)

# Set initial position of agents
agent_positions = {0: 0, 1: 3, 2: 11, 3: 6}  # agent:position
G.init_agents(agent_positions)

# Set up the connectivity problem
cp = ConnectivityProblem()
cp.graph = G  # graph
cp.T = 2  # time
cp.master = [0]  # master_agent
cp.static_agents = []  # static agents
cp.big_agents = [0, 1, 2, 3]

# Solve
cp.solve_flow(master=False, connectivity=True, frontier_reward = False)

# Animate solution
animate(G, cp.traj, cp.conn)
