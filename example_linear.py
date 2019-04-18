import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product

from graph_connectivity.animate import *
from graph_connectivity.def_flow_ilp import *

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
cp.T = 4                       # time
cp.b = agent_positions         # base:node
cp.static_agents = []          # static agents

#Solve (solve() or solve_adaptive())
st0 = time.time()
sol = cp.solve()
print("solver time: {:.2f}s".format(time.time() - st0))

#Plot Graph (saves image as graph.png)
cp.plot_solution()

#Animate solution
animate_solution(cp)
