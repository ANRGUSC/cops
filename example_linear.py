from graph_connectivity.problem import *

n = 11  # size of graph

# Define a connectivity graph
G = Graph()
G.add_transition_path(list(range(n)))
G.add_connectivity_path(list(range(n)))
G.set_node_positions({i: (0,i) for i in range(n)})

agent_positions = {0: 0, 1: 1, 2: n-1}
G.init_agents(agent_positions)

# Set up the connectivity problem
cp = ConnectivityProblem()
cp.graph = G                   # graph
cp.T = 5                       # time
cp.static_agents = []          # static agents

#Solve
sol = cp.solve_flow()

#Plot Graph (saves image as graph.png)
cp.plot_solution()

#Animate solution
cp.animate_solution()
