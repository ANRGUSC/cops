import numpy as np
import time
import random

from colorama import Fore, Style

from cops.clustering import ClusterProblem
from cops.explore_problem import ExplorationProblem
from copy import deepcopy
from cops.animate import animate_cluster_sequence

from graph_examples import get_small_graph

# (100, 10) -> 118
# (100, 20) -> 201

MAXITER = 100
NUM_EPOCHS = 20
EPSILON = 0.5
LEARNING_RATE = 0.9
GAMMA = 0.99 # discount

START_POSITIONS = {r:0 for r in range(2)}
EAGENTS = [r for r in range(2)]
MASTER = 0

Q = {} # Q[state][action] = Q-value
all_actions = ["to_frontier", "explore 4", "explore 8", "explore 16", "return"]
# all_actions = ["to_frontier", "explore 4", "return"]


def quick_print(G):
    nodes_known = [1 if G.nodes[v]["known"] else 0 for v in G.nodes]
    print(sum(nodes_known), "out of", len(nodes_known), "graph nodes explored")
    print("agent(s) at:",G.agents)

def reward_func1(G_i, G_f):
    return G_f.count_known() - G_i.count_known()

def reward_func2(G_f,action_state_machine,last_count_known):
    if action_state_machine == "at_base":
        return G_f.count_known()-last_count_known
    else:
        return 0

def take_to_frontier_action(G,agent_positions):
    g1 = deepcopy(G)
    unknown = [v for v in G.nodes if not G.nodes[v]["known"]]
    g1.remove_nodes_from(unknown)
    cp1 = ClusterProblem()
    cp1.graph = g1
    cp1.master = MASTER
    cp1.static_agents = [MASTER]
    cp1.eagents = EAGENTS
    cp1.graph.init_agents(agent_positions)
    # soft: Whether to use hard or soft contraints for subclusters; dead: Whether to plan in dead clusters.
    tofront_data = cp1.solve_to_frontier_problem(verbose=False, soft=True, dead=True)
    agent_positions = {r: cp1.traj[(r, cp1.T_sol)] for r in cp1.graph.agents}
    return cp1, tofront_data, agent_positions

def take_explore_action(G,T,agent_positions,tofront_data):
    ep = ExplorationProblem()
    ep.graph = G  # full graph
    ep.T = T  # exploration time
    ep.static_agents = [MASTER]  # static agents
    nonactivated_agents = set(agent_positions.keys()) - set(
        r for r_list in tofront_data.active_agents.values() for r in r_list
    )
    for r in nonactivated_agents:
        ep.static_agents.append(r)
    ep.graph.agents = agent_positions
    ep.eagents = EAGENTS
    ep.solve()
    return ep

def take_return_action(G,agent_positions,cp1,tofront_data):
    g2 = deepcopy(G)
    unknown = [v for v in G.nodes if not G.nodes[v]["known"]]
    g2.remove_nodes_from(unknown)
    cp2 = ClusterProblem()
    cp2.graph = g2
    cp2.master = MASTER
    cp2.static_agents = [MASTER]
    # cp2.big_agents = eagents
    cp2.eagents = EAGENTS
    cp2.graph.init_agents(agent_positions)
    cp2.to_frontier_problem = cp1
    cp2.solve_to_base_problem(tofront_data, verbose=False, dead=True)
    agent_positions = {r: cp2.traj[(r, cp2.T_sol)] for r in cp2.graph.agents}
    return cp2, agent_positions

#### TRAIN
for epoch in range(NUM_EPOCHS):

    #### STATE
    G = get_small_graph() # 16 vertices
    agent_positions = START_POSITIONS
    G.init_agents(agent_positions)

    # Set known attribute
    for v in G.nodes():
        G.nodes[v]["known"] = False
    for r, v in agent_positions.items():
        G.nodes[v]["known"] = True

    action_state_machine = "at_base"
    tofront_data = None
    last_count_known = 0

    for i in range(MAXITER):

        frontiers = {v: 2 for v in G.nodes if G.is_frontier(v)}
        G.set_frontiers(frontiers)


        #### LIMIT ACTION SPACE
        if action_state_machine == "at_base":
            actions = ["to_frontier"]
        elif action_state_machine == "at_frontier":
            actions = [action for action in all_actions if "explore" in action]+["return"]
        elif action_state_machine == "explored":
            actions = [action for action in all_actions if "explore" in action]+["return"]

        if G not in Q:
            Q[G] = {action:0 for action in all_actions}

        if random.uniform(0, 1) < EPSILON:
            #### EXPLORE
            action = random.choice(actions)
        else:
            #### EXPLOIT
            q = 0
            action = random.choice(actions)
            for a in actions:
                if Q[G][a] > q:
                    action = a

        #### TAKE ACTION
        G_new = deepcopy(G)
        if action == "to_frontier":
            cp1, tofront_data, agent_positions = take_to_frontier_action(G_new,agent_positions)
            G_new.agents = agent_positions
            action_state_machine = "at_frontier"
        elif "explore" in action:
            T = int(action.split(" ")[-1])
            ep = take_explore_action(G_new,T,agent_positions,tofront_data)
            G_new.agents = agent_positions
            action_state_machine = "explored"
        elif action == "return":
            cp2, agent_positions = take_return_action(G_new, agent_positions, cp1, tofront_data)
            G_new.agents = agent_positions
            action_state_machine = "at_base"

        if G_new not in Q:
            Q[G_new] = {action:0 for action in all_actions}

        R = reward_func2(G_new,action_state_machine,last_count_known)
        Q[G][action] = Q[G][action] + LEARNING_RATE * (R + (GAMMA * max(Q[G_new].values())) - Q[G][action])
        last_count_known += R

        # quick_print(G)

        if last_count_known == len(G_new.nodes):
            print(i)
            print("done exploring")
            break

        G = G_new

    print(".", end =" ")
print("\nend of training")

print("\n",len(Q),"possible graph states considered")
for G_state in Q:
    if max(Q[G_state].values()) > 0:
        # quick_print(G_state)
        print(Q[G_state])
