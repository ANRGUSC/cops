from colorama import Fore, Style

from cops.clustering import ClusterProblem
from cops.explore_problem import ExplorationProblem
from copy import deepcopy
from cops.animate import animate_cluster_sequence

from graph_examples import get_huge_graph

G = get_huge_graph()

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
    # cp1.big_agents = eagents
    cp1.eagents = eagents
    cp1.graph.init_agents(agent_positions)
    tofront_data = cp1.solve_to_frontier_problem(verbose=True, soft=True, dead=True)
    agent_positions = {r: cp1.traj[(r, cp1.T_sol)] for r in cp1.graph.agents}

    cp1.subgraphs = tofront_data.cs.subgraphs  # hack for animate..
    problem_list.append(cp1)

    # Process2-EXPLORE FRONTIERS---------------------------------------------
    ep = ExplorationProblem()
    ep.graph = G  # full graph
    ep.T = 8  # exploration time

    ep.static_agents = [r for r in static_agents]  # static agents
    nonactivated_agents = set(agent_positions.keys()) - set(
        r for r_list in tofront_data.active_agents.values() for r in r_list
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
    # cp2.big_agents = eagents
    cp2.eagents = eagents
    cp2.graph.init_agents(agent_positions)
    cp2.to_frontier_problem = cp1
    cp2.solve_to_base_problem(tofront_data, verbose=True, dead=True)
    agent_positions = {r: cp2.traj[(r, cp2.T_sol)] for r in cp2.graph.agents}

    cp2.subgraphs = tofront_data.cs.subgraphs  # hack for animate..
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
