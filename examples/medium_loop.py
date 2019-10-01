import time

from cops.animate import *
from cops.problem import *
from cops.explore_problem import *
from copy import deepcopy

from graph_examples import get_medium_graph

G = get_medium_graph()

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
    cp1.diameter_solve_flow(master=True, connectivity=False)
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
    sol = cp2.linear_search_solve_flow(master=False, connectivity=True)
    agent_positions = {r: cp2.traj[(r, cp2.T)] for r in cp2.graph.agents}

    problem_list.append(cp2)

# ANIMATION----------------------------------------------------------------------
animate_sequence(G, problem_list, FPS=15, STEP_T=0.5)
