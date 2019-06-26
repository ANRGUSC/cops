from graph_connectivity.problem import *
from graph_connectivity.animate import *

n = 8  # size of graph

# Define a connectivity graph
G = Graph()
G.add_transition_path(list(range(n)))
G.add_connectivity_path(list(range(n)))
G.set_node_positions({i: (0,i) for i in range(n)})

frontiers = {5: 1}
G.set_frontiers(frontiers)

agent_positions = {0: 0, 1: 1, 2: 3}
#agent_positions = {0: 0, 1: n//2, 2: n-1}
G.init_agents(agent_positions)

G.plot_graph()

# Set up the connectivity problem
cp = ConnectivityProblem()
cp.graph = G
cp.T = 6
cp.static_agents = [0]
cp.master = 0

#Solve
cp.solve_flow(optimal = True, frontier_reward = True)

#Plot Graph (saves image as graph.png)ß
# plot_solution(cp)

#Animate solution
animate(G, cp.traj, cp.conn)
