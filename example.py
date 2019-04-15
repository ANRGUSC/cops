import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product

#for powerset ilp
#from graph_connectivity.def_ilp import *
#for flow ilp
from graph_connectivity.def_flow_ilp import *

# Define a connectivity graph
G = Graph()

#Add edges to graph
connectivity_edges = list(range(20)) #directed connectivity path (one way)
transition_edges = list(range(20))   #directed transition path (one way)
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)

#Set node positions
node_positions = {i: (0,i) for i in range(20)}   #example {0: (0,0), 1: (1,1)}
G.set_node_positions(node_positions)

#Set initial position of agents
agent_positions = {0: 0, 1: 1, 2: 19}    #agent:position
G.init_agents(agent_positions)

#Plot Graph (saves image as graph.png)
G.plot_graph()

# Set up the connectivity problem
cp = ConnectivityProblem()
cp.graph = G                            #graph
cp.T = 9                                #time
cp.b = {0: 0, 1: 1, 2: 19}              #base:node
cp.static_agents = []                   #static agents

#Solve (solve() or solve_adaptive())
st0 = time.time()
sol = cp.solve()
print("solver time: {:.2f}s".format(time.time() - st0))

#Plot Graph (saves image as graph.png)
cp.plot_solution()



'''
#Powerset
z = cp.solution['x'][0 : cp.num_z]
e = cp.solution['x'][cp.num_z : cp.num_z + cp.num_e]
y = cp.solution['x'][cp.num_z + cp.num_e : cp.num_z + cp.num_e + cp.num_y]

print("Finished")
print("z:", len(z), '\n', np.reshape(z, (cp.T+1, cp.num_v * cp.num_r)))
print("e:", len(e), '\n', np.reshape(e, (cp.T, cp.num_v**2)))
print("y:", len(y), '\n', np.reshape(y, (cp.T+1, cp.num_v * len(cp.b))))
'''


'''
#Flow
z = cp.solution['x'][0 : cp.num_z]
e = cp.solution['x'][cp.num_z : cp.num_z + cp.num_e]
c = cp.solution['x'][cp.num_z + cp.num_e : cp.num_z + cp.num_e + cp.num_c]
cbar = cp.solution['x'][cp.num_z + cp.num_e + cp.num_c: cp.num_z + cp.num_e + cp.num_c + cp.num_cbar]
f = cp.solution['x'][cp.num_z + cp.num_e + cp.num_c +cp.num_cbar : cp.num_z + cp.num_e + cp.num_c +cp.num_cbar + cp.num_f]
fbar = cp.solution['x'][cp.num_z + cp.num_e + cp.num_c +cp.num_cbar + cp.num_f : cp.num_z + cp.num_e + cp.num_c +cp.num_cbar + cp.num_f + cp.num_fbar]

print("Finished")
print("z:", len(z), '\n', np.reshape(z, (cp.T+1, cp.num_v * cp.num_r)))
print("f:", len(f), '\n', np.reshape(f, (cp.T, cp.num_b*cp.num_v**2)))
print("fbar:", len(fbar), '\n', np.reshape(fbar, (cp.T+1, cp.num_b*cp.num_v**2)))
'''
