from graph_connectivity.problem import *
from graph_connectivity.animate import *

n = 3  # size of graph

# Define a connectivity graph
G = Graph()
G.add_transition_path(list(range(n)))
G.add_connectivity_path(list(range(n)))
G.set_node_positions({i: (0,i) for i in range(n)})

agent_positions = {0: 0, 1: 1, 2: n-2, 3: n-1}
agent_positions = {0: 0, 1: 1}
#agent_positions = {0: 0, 1: n//2, 2: n-1}
G.init_agents(agent_positions)

G.plot_graph()

# Set up the connectivity problem
cp = ConnectivityProblem()
cp.graph = G                   # graph
cp.T = 8          # time
cp.static_agents = [0, 3]

#Solve
#cp.solve_flow(optimal = True, frontier_reward = False)

#Plot Graph (saves image as graph.png)ß
# plot_solution(cp)

#Animate solution
#animate(G, cp.traj, cp.conn)
