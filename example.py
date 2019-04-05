import time

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from itertools import product

from graph_connectivity.def_ilp import *

# Define a connectivity graph
G = Graph()
connectivity_edges = [0,1,2,3] #directed connectivity path (one way)
transition_edges = [0,1,2,3]   #directed transition path (one way)
#Add edges to graph
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)

#Define position of each node as a dict (use ! after coordinate to get accurate plots)
node_positions = {0: (0,0), 1: (0, 1), 2: (0,2), 3: (0,3)}
#node_positions = {0: '0,0!', 1: '1, 1!', 2: '0,2!', 3: '1,3!'}
#Add nodes with positions to graph
G.set_node_positions(node_positions)

#Set initial position of agents
agent_positions = {0: 0, 1: 1, 2: 3}    #agent:position
#Init agents in graphs
G.init_agents(agent_positions)

#Plot Graph (saves image as graph.png)
G.plot_graph()

# Set up the connectivity problem
cp = ConnectivityProblem()

# Specify graph
cp.graph = G
# Specify time horizon
<<<<<<< HEAD
cp.T = 2
=======
cp.T = 3
>>>>>>> 32d76db74da9deb1f451dddf873ed482cecc3d9e
# Specify bases
cp.b = {0: 0, 1: 1, 2: 3}

cp.static_agents = [0, 2]


#Set up Dynamic Constraints
dc = DynamicConstraints(cp)

#Set up Connectivity contraint
t0 = time.time()
cc = ConnectivityConstraint(cp)
print("cc took", time.time() - t0)

cp.add_constraints([dc, cc])


#Compile Problem
cp.compile()

print('Solving')
z, e, y= cp.solve()

#Plot Graph (saves image as graph.png)
cp.plot_solution()

#cp.test_solution()

print("finished")

#ani = animate_count(cp, 8)
#plt.show()

print("z:", len(z), '\n', np.reshape(z, (cp.T+1, cp.num_v * cp.num_r)))
print("e:", len(e), '\n', np.reshape(e, (cp.T, cp.num_v**2)))
print("y:", len(y), '\n', np.reshape(y, (cp.T+1, cp.num_v)))
