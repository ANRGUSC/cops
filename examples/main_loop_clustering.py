import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product

from colorama import Fore, Style

from cops.animate import *
from cops.problem import *
from cops.explore_problem import *
from cops.clustering import *
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
connectivity_edges = [11, 25, 26, 27, 28, 29, 30, 31]
transition_edges = [11, 25, 26, 27, 28, 29, 30, 31]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [29, 32, 33, 34, 35]
transition_edges = [29, 32, 33, 34, 35]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [22, 36, 37, 38, 39, 40, 41]
transition_edges = [22, 36, 37, 38, 39, 40, 41]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [39, 42, 43, 44, 45]
transition_edges = [39, 42, 43, 44, 45]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [5, 46, 47, 48, 49]
transition_edges = [5, 46, 47, 48, 49]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [15, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
transition_edges = [15, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [55, 61, 62, 63, 64, 65, 66, 67]
transition_edges = [55, 61, 62, 63, 64, 65, 66, 67]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [46, 68, 69, 70, 71, 72, 73, 74, 75, 76, 33]
transition_edges = [46, 68, 69, 70, 71, 72, 73, 74, 75, 76, 33]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [75, 77, 78, 27]
transition_edges = [75, 77, 78, 27]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [76, 79, 80, 81, 82, 83, 84, 85, 86, 87]
transition_edges = [76, 79, 80, 81, 82, 83, 84, 85, 86, 87]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [36, 88, 89, 90, 91, 92, 93, 94, 95]
transition_edges = [36, 88, 89, 90, 91, 92, 93, 94, 95]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [24, 96, 97, 98, 99]
transition_edges = [24, 96, 97, 98, 99]
G.add_transition_path(transition_edges)
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
    25: (-1.6, 5),
    26: (-1.7, 6),
    27: (-2, 6.5),
    28: (-2, 7.5),
    29: (-1.5, 8.5),
    30: (-1.5, 10),
    31: (-1.5, 11),
    32: (-2.5, 8.8),
    33: (-3.5, 9),
    34: (-4, 10),
    35: (-4.5, 11),
    36: (6, 3),
    37: (6.5, 3.8),
    38: (7.2, 4.5),
    39: (7.5, 5.5),
    40: (7, 6),
    41: (7, 7),
    42: (8.5, 6),
    43: (9.5, 6.5),
    44: (10, 7),
    45: (11, 7),
    46: (-4, 1),
    47: (-4.5, 2),
    48: (-5, 3),
    49: (-5, 4),
    50: (1, 5),
    51: (1.2, 6),
    52: (1.5, 7),
    53: (2, 8),
    54: (2, 9),
    55: (3, 10),
    56: (4, 10.5),
    57: (5, 11),
    58: (6, 11),
    59: (6.5, 12),
    60: (7, 13),
    61: (2.5, 11),
    62: (2, 12),
    63: (2, 13),
    64: (1, 14),
    65: (0, 15),
    66: (-1, 15.5),
    67: (-2.5, 15),
    68: (-5, 1),
    69: (-6, 1),
    70: (-7, 2),
    71: (-7, 3),
    72: (-7.5, 4),
    73: (-7.5, 5),
    74: (-6, 6),
    75: (-5, 7),
    76: (-4, 8),
    77: (-4, 6),
    78: (-3, 6),
    79: (-5, 8),
    80: (-5, 9.2),
    81: (-5.5, 10),
    82: (-6, 11),
    83: (-7, 11),
    84: (-7, 12),
    85: (-6.5, 13),
    86: (-6, 14),
    87: (-7, 15),
    88: (7, 3),
    89: (8, 3),
    90: (9, 4),
    91: (9.5, 5),
    92: (10, 5),
    93: (11, 6),
    94: (12, 7),
    95: (13, 8),
    96: (3.5, 5),
    97: (4, 6),
    98: (4, 7),
    99: (4.5, 8),
}
G.set_node_positions(node_positions)

# Set small nodes
small_nodes = [v for v in node_positions]
small_nodes.remove(0)
G.set_small_node(small_nodes)

# Set initial position of agents
agent_positions = {r: 0 for r in range(10)}  # agent:position
G.init_agents(agent_positions)
# exploration agents
eagents = [r for r in range(10)]

# Set known attribute
for v in G.nodes():
    G.nodes[v]["known"] = False
for r, v in agent_positions.items():
    G.nodes[v]["known"] = True

G.plot_graph()

problem_list = []

master = 0
master_node = agent_positions[master]
static_agents = [0]
MAXITER = 10000
i_iter = 0

agents_home = True
# MAIN-LOOP----------------------------------------------------------------------
while not G.is_known() or not agents_home:

    # try:
    frontiers = {v: 2 for v in G.nodes if G.is_frontier(v)}
    G.set_frontiers(frontiers)

    # create sub-graph
    g1 = deepcopy(G)
    g2 = deepcopy(G)
    unknown = [v for v in G.nodes if not G.nodes[v]["known"]]
    g1.remove_nodes_from(unknown)
    g2.remove_nodes_from(unknown)

    # Process1-TRAVERSE TO FRONTIERS-----------------------------------------
    # CLUSTERING
    print()
    print(
        Style.BRIGHT
        + Fore.BLUE
        + "Solving to frontier problem on {} known nodes".format(len(g1))
        + Style.RESET_ALL
    )
    cp1 = ClusterProblem()
    cp1.graph = g1
    cp1.master = master
    cp1.static_agents = [r for r in static_agents]
    cp1.big_agents = eagents
    cp1.eagents = eagents
    cp1.graph.init_agents(agent_positions)
    cp1.solve_to_frontier_problem(verbose=True, soft=True, dead=True)
    agent_positions = {r: cp1.traj[(r, cp1.T)] for r in cp1.graph.agents}
    problem_list.append(cp1)

    # Process2-EXPLORE FRONTIERS---------------------------------------------
    ep = ExplorationProblem()
    ep.graph = G  # full graph
    ep.T = 8  # exploration time

    ep.static_agents = [r for r in static_agents]  # static agents
    nonactivated_agents = set(agent_positions.keys()) - set(
        r for r_list in cp1.active_agents.values() for r in r_list
    )
    for r in nonactivated_agents:
        ep.static_agents.append(r)

    ep.graph.agents = agent_positions
    ep.eagents = eagents
    ep.solve()
    problem_list.append(ep)

    # Process3-SEND DATA TO BASE---------------------------------------------
    # CLUSTERING
    print()
    print(Style.BRIGHT + Fore.BLUE + "Solving to base problem" + Style.RESET_ALL)
    cp2 = ClusterProblem()
    cp2.graph = g2
    cp2.master = master
    cp2.static_agents = [r for r in static_agents]
    cp2.big_agents = eagents
    cp2.eagents = eagents
    cp2.graph.init_agents(agent_positions)
    cp2.to_frontier_problem = cp1
    cp2.solve_to_base_problem(verbose=True, dead=True)
    agent_positions = {r: cp2.traj[(r, cp2.T)] for r in cp2.graph.agents}
    problem_list.append(cp2)

    # check if all agents are home-------------------------------------------
    agents_home = True
    for r, v in agent_positions.items():
        if v != master_node:
            agents_home = False

    i_iter += 1
    if i_iter > MAXITER:
        break
# except:
#    print('Break')
#    break

# ANIMATION----------------------------------------------------------------------

print("Whole loop is completed!")

animate_cluster_sequence(G, problem_list, FPS=15, STEP_T=0.5)
