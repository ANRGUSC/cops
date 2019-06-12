from graph_connectivity.problem import *
from graph_connectivity.problem_animate import *

n = 7  # size of graph

# Define a connectivity graph
G = Graph()
G.add_transition_path(list(range(n)))
G.add_connectivity_path(list(range(n)))
G.set_node_positions({i: (0,i) for i in range(n)})

agent_positions = {0: 0, 1: n//2, 2: n-1}
G.init_agents(agent_positions)

# Set up the connectivity problem
cp = ConnectivityProblem()
cp.graph = G                   # graph
cp.T = 2                       # time
cp.static_agents = []          # static agents
cp.src = [1]

#Solve
sol = cp.solve_adaptive()

#Plot Graph (saves image as graph.png)
plot_solution(cp)

#Animate solution
animate_solution(cp)
