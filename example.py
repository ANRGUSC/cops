import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product

#animation
from graph_connectivity.animate import *

#powerset ilp
#from graph_connectivity.def_ilp import *

#flow ilp
from graph_connectivity.def_flow_ilp import *

# Define a connectivity graph
G = Graph()

#LINEAR EXAMPLE
#Add edges to graph
#connectivity_edges = list(range(11)) #directed connectivity path (one way)
#transition_edges = list(range(11))   #directed transition path (one way)
#G.add_transition_path(transition_edges)
#G.add_connectivity_path(connectivity_edges)

#Add edges to graph
connectivity_edges = [0, 1, 2, 3, 4, 5]
transition_edges = [0, 1, 2, 3, 4, 5]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [4, 6, 7, 8]
transition_edges = [4, 6, 7, 8]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [1, 9, 10, 11, 12, 13]
transition_edges = [1, 9, 10, 11, 12, 13]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [9, 14, 15, 12]
transition_edges = [9, 14, 15, 12]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [1, 16, 17, 18, 19, 20, 21, 22, 23, 24]
transition_edges = [1, 16, 17, 18, 19, 20, 21, 22, 23, 24]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [8, 10]
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [15, 24]
G.add_connectivity_path(connectivity_edges)

#Set node positions
#LINEAR EXAMPLE
#node_positions = {i: (0,i) for i in range(11)}   #example {0: (0,0), 1: (1,1)}
node_positions = {0: (0,0), 1: (0,1), 2: (-2,1), 3: (-2,2), 4: (-3,2), 5: (-3,1),
                6: (-3,3), 7: (-3.5,3.5), 8: (-2.5,3.5), 9: (0,3), 10: (-1.8,3),
                11: (-1.6,4), 12: (0,4), 13: (0,5), 14: (1,3), 15: (1,4),
                16: (1.5,1), 17: (2.5,1.3), 18: (4,1.3), 19: (5,1.3), 20: (5,2),
                21: (4,3), 22: (5,3), 23: (5,4), 24: (3.5,4)}
G.set_node_positions(node_positions)

#Set initial position of agents
agent_positions = {0: 0, 1: 8, 2: 12, 3: 23}    #agent:position
G.init_agents(agent_positions)

#Plot Graph (saves image as graph.png)
G.plot_graph()

# Set up the connectivity problem
cp = ConnectivityProblem()
cp.graph = G                            #graph
cp.T = 5                                #time
cp.b = agent_positions                  #base:node
cp.static_agents = []                  #static agents

#Solve (solve() or solve_adaptive())
st0 = time.time()
sol = cp.solve()
print("solver time: {:.2f}s".format(time.time() - st0))

#Plot Graph (saves image as graph.png)
cp.plot_solution()

#Animate solution
animate_solution(cp)



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
