import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product

from cops.animate import *
from cops.problem import *
from cops.explore_problem import *
from copy import deepcopy

# MASTERGRAPH--------------------------------------------------------------------
G = Graph()

# Add edges to graph
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

# Set node positions
node_positions = {
    0: (0, 0),
    1: (0, 1),
    2: (-2, 1),
    3: (-2, 2),
    4: (-3, 2),
    5: (-3, 1),
    6: (-3, 3),
    7: (-3.5, 3.5),
    8: (-2.5, 3.5),
    9: (0, 3),
    10: (-1.8, 3),
    11: (-1.6, 4),
    12: (0, 4),
    13: (0, 5),
    14: (1, 3),
    15: (1, 4),
    16: (1.5, 1),
    17: (2.5, 1.3),
    18: (4, 1.3),
    19: (5, 1.3),
    20: (5, 2),
    21: (4, 3),
    22: (5, 3),
    23: (5, 4),
    24: (3.5, 4),
}
G.set_node_positions(node_positions)

# Set initial position of agents
agent_positions = {0: 0, 1: 0, 2: 0, 3: 0}  # agent:position
G.init_agents(agent_positions)

# Set known attribute
for v in G.nodes():
    G.nodes[v]["known"] = False
for r, v in agent_positions.items():
    G.nodes[v]["known"] = True


problem_list = []

master = 1
static_agents = [0]

# MAIN-LOOP----------------------------------------------------------------------
while not G.is_known():

    # find frontiers
    frontiers = {v: 1 for v in G.nodes if G.is_frontier(v)}
    G.set_frontiers(frontiers)

    # create sub-graph
    g = deepcopy(G)
    unknown = [v for v in G.nodes if not G.nodes[v]["known"]]
    g.remove_nodes_from(unknown)

    # Process1-TRAVERSE TO FRONTIERS-------------------------------------------------
    cp1 = ConnectivityProblem()
    cp1.graph = g  # graph
    cp1.master = master  # master_agent
    cp1.static_agents = static_agents  # static agents
    cp1.graph.agents = agent_positions
    cp1.src = []
    cp1.snk = []
    cp1.diameter_solve_flow(master=True, connectivity=False, optimal=True)
    agent_positions = {r: cp1.traj[(r, cp1.T)] for r in cp1.graph.agents}
    problem_list.append(cp1)

    # Process2-EXPLORE FRONTIERS-----------------------------------------------------
    ep = ExplorationProblem()
    ep.graph = G  # full graph
    ep.T = 8  # exploration time
    ep.static_agents = static_agents  # static agents
    ep.graph.agents = agent_positions
    ep.solve()
    problem_list.append(ep)

    # Process3-SEND DATA TO DASE-----------------------------------------------------
    cp2 = ConnectivityProblem()
    cp2.graph = g  # graph
    cp2.master = master  # master_agent
    cp2.static_agents = static_agents  # static agents
    cp2.graph.agents = agent_positions  # end positions from process 1
    cp2.src = [r for r in cp1.graph.agents if g.is_local_frontier(cp1.traj[(r, cp1.T)])]
    cp2.snk = [0]
    sol = cp2.linear_search_solve_flow(master=False, connectivity=True, optimal=True)
    agent_positions = {r: cp2.traj[(r, cp2.T)] for r in cp2.graph.agents}

    problem_list.append(cp2)

# ANIMATION----------------------------------------------------------------------
animate_sequence(G, problem_list, FPS=15, STEP_T=0.5)
