import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product
from graph_connectivity.def_ilp import *

# Define a connectivity graph
G = Graph()

#Add edges to graph
connectivity_edges = list(range(4)) #directed connectivity path (one way)
transition_edges = list(range(4))   #directed transition path (one way)
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)

#Set node positions
node_positions = {i: (0,i) for i in range(4)}   #example {0: (0,0), 1: (1,1)}
G.set_node_positions(node_positions)

#Set initial position of agents
agent_positions = {0: 0, 1: 1, 2: 3}    #agent:position
G.init_agents(agent_positions)

#Plot Graph (saves image as graph.png)
G.plot_graph()

# Set up the connectivity problem
cp = ConnectivityProblem()
cp.graph = G                            #graph
cp.T = 2                                #time
cp.b = {0: 0, 1: 1, 2:3}                #base:node
cp.static_agents = [0, 2]               #static agents

#Compile (compile() or compile_adaptive())
cp.compile_adaptive()

#Solve (solve() or solve_adaptive())
print('Solving')
st0 = time.time()
z, e, y= cp.solve_adaptive()
print("solver time:", time.time() - st0)

#Plot Graph (saves image as graph.png)
cp.plot_solution()

print("Finished")
print("z:", len(z), '\n', np.reshape(z, (cp.T+1, cp.num_v * cp.num_r)))
print("e:", len(e), '\n', np.reshape(e, (cp.T, cp.num_v**2)))
print("y:", len(y), '\n', np.reshape(y, (cp.T+1, cp.num_v * len(cp.b))))
