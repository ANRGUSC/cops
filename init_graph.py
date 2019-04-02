import sys
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from itertools import product

sys.path.append('../')
from def_ilp import *

# Define a connectivity graph
G = Graph()
connectivity_edges = [0,1,2,3] #directed connectivity path (one way)
transition_edges = [0,1,2,3]   #directed transition path (one way)
#Add edges to graph
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)

#Define position of each node as a dict (use ! after coordinate to get accurate plots)
node_positions = {0: '0,0!', 1: '1, 1!', 2: '0,2!', 3: '1,3!'}
#Add nodes with positions to graph
G.set_node_positions(node_positions)

#Set initial position of agents
agent_positions = {0: 0, 1: 0, 2: 3}    #agent:position
#Init agents in graphs
G.init_agents(agent_positions)

#Plot Graph (saves image as graph.png)
G.plot_graph()



# Set up the connectivity problem
cp = ConnectivityProblem()

# Specify graph
cp.graph = G
# Specify time horizon
cp.T = 3
# Specify bases
cp.b = agent_positions


#Set up Dynamic Constraints
dc = DynamicConstraints(cp)


cp.add_constraints([dc])

solution = cp.solve()

#cp.test_solution()

print("finished")

#ani = animate_count(cp, 8)
#plt.show()


print(solution)
