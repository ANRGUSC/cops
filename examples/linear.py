from cops.problem import *
from cops.animate import *

from graph_examples import get_linear_graph

G = get_linear_graph(8)

frontiers = {5: 1}
G.set_frontiers(frontiers)

agent_positions = {0: 0, 1: 1, 2: 3}
G.init_agents(agent_positions)

# Set up the connectivity problem
cp = ConnectivityProblem()
cp.graph = G
cp.T = 6
cp.static_agents = [0]
cp.master = 0

# Solve
cp.solve_flow(frontier_reward=True)

# Plot Graph (saves image as graph.png)ß
# plot_solution(cp)

# Animate solution
animate(G, cp.traj, cp.conn)
