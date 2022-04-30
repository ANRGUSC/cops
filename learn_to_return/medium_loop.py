from cops.animate import animate_sequence
from cops.problem import ConnectivityProblem
from cops.explore_problem import ExplorationProblem
from cops.agent_problem import AgentProblem
from copy import deepcopy

from graph_examples import get_medium_graph, get_small_graph

G = get_small_graph()

# Set initial position of agents
agent_positions = {0: 0, 1: 0}  # agent:position
G.init_agents(agent_positions)

# Set known attribute
for v in G.nodes():
    G.nodes[v]["known"] = False
for r, v in agent_positions.items():
    G.nodes[v]["known"] = True


problem_list = []

master = 0
static_agents = [0]

# MAIN-LOOP----------------------------------------------------------------------
MAXITER = 50
for i in range(MAXITER):

    # find frontiers
    frontiers = {v: 1 for v in G.nodes if G.is_frontier(v)}
    G.set_frontiers(frontiers)

    ap = AgentProblem()
    ap.graph = G
    ap.static_agents = static_agents
    ap.graph.agents = agent_positions
    ap.solve_explore()
    problem_list.append(ap)

    if G.is_known():
        print("DONE EXPLORING")
        break

# ANIMATION----------------------------------------------------------------------
animate_sequence(G, problem_list, FPS=15, STEP_T=0.5)
