import numpy as np
import time
import random
import pickle
import sys

from colorama import Fore, Style

from cops.clustering import ClusterProblem
from cops.explore_problem import ExplorationProblem
from copy import deepcopy
from cops.animate import animate_cluster_sequence

from graph_examples import get_small_graph

MAXITER = 100
NUM_EPOCHS = 25
EPSILON = 0.5
LEARNING_RATE = 0.9
GAMMA = 0.99 # discount

START_POSITIONS = {r:0 for r in range(2)}
EAGENTS = [r for r in range(2)]
MASTER = 0

Q = {} # Q[state][action] = Q-value
G_to_idx = {}
all_actions = ["to_frontier", "explore 4", "explore 8", "explore 16", "return"]
# all_actions = ["to_frontier", "explore 4", "return"]


def quick_print(G):
    nodes_known = [1 if G.nodes[v]["known"] else 0 for v in G.nodes]
    print(sum(nodes_known), "out of", len(nodes_known), "graph nodes explored")
    print("agent(s) at:",G.agents)

def reward_func1(G_i, G_f):
    return G_f.count_known() - G_i.count_known()

def reward_func2(G_f,action_state_machine,last_count_known):
    # TODO this rewards if any agent can transfer to the base (1 agent setting this makes sense)
    for a in G_f.agents:
        if a == MASTER:
            continue
        if G_f.agents[a] == MASTER or G_f.has_conn_edge(G_f.agents[a],MASTER):
    # if action_state_machine == "at_base":
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

def main_function_training():
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

        #### START AT THE FRONTIER
        frontiers = {v: 2 for v in G.nodes if G.is_frontier(v)}
        G.set_frontiers(frontiers)
        cp1, tofront_data, G.agents = take_to_frontier_action(G,agent_positions)
        action_state_machine = "at_frontier"
        actions_taken = [cp1]
        last_count_known = 0

        for i in range(MAXITER):

            frontiers = {v: 2 for v in G.nodes if G.is_frontier(v)}
            G.set_frontiers(frontiers)
            # print("frontiers",frontiers)

            #### LIMIT ACTION SPACE
            if action_state_machine == "at_base":
                actions = ["to_frontier"]
            elif action_state_machine == "at_frontier":
                actions = [action for action in all_actions if "explore" in action]+["return"]
            elif action_state_machine == "explored":
                actions = ["return"]+[action for action in all_actions if "explore" in action]

            if G not in Q:
                Q[G] = {action:0 for action in all_actions}
                G_to_idx[G] = len(G_to_idx)

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
                        q = Q[G][a]

            #### TAKE ACTION
            G_new = deepcopy(G)
            # print(action_state_machine,":",action)
            if action == "to_frontier":
                cp1, tofront_data, agent_positions = take_to_frontier_action(G_new,agent_positions)
                G_new.agents = agent_positions
                action_state_machine = "at_frontier"
                actions_taken.append(cp1)
            elif "explore" in action:
                T = int(action.split(" ")[-1])
                ep = take_explore_action(G_new,T,agent_positions,tofront_data)
                G_new.agents = agent_positions
                action_state_machine = "explored"
                actions_taken.append(ep)
            elif action == "return":
                cp2, agent_positions = take_return_action(G_new, agent_positions, cp1, tofront_data)
                G_new.agents = agent_positions
                action_state_machine = "at_base"
                actions_taken.append(cp2)

            if G_new not in Q:
                Q[G_new] = {action:0 for action in all_actions}
                G_to_idx[G_new] = len(G_to_idx)

            R = reward_func2(G_new,action_state_machine,last_count_known)
            # if R:
                # print("Reward received for taking action",action,"at state",G_to_idx[G])
            Q[G][action] = Q[G][action] + LEARNING_RATE * (R + (GAMMA * max(Q[G_new].values())) - Q[G][action])
            last_count_known += R

            # quick_print(G)

            if last_count_known == len(G_new.nodes):
                print(i)
                print("done exploring")
                break

            # print("G before:")
            # quick_print(G)
            # print(G_to_idx[G])
            # print("G after:")
            # quick_print(G_new)
            # print(G_to_idx[G_new])

            G = G_new
            # print(len(Q),"states considered")
            # print("")
            # print("")

        print(".", end =" ")

    print("\n",len(Q),"possible graph states considered")

    # for G_state in Q:
    #     if max(Q[G_state].values()) > 0:
    #         # quick_print(G_state)
    #         print(Q[G_state])

    print("saving Q-learned tabular policy")

    # for G_state in Q:
    #     for action in Q[G_state]:
    #         if action not in ["return"] and Q[G_state][action] > 0:
    #             print(Q[G_state])

    # for G_state in G_to_idx:
    #     print(G_to_idx[G_state])
    #     print(Q[G_state])

    try:
        policy_name = input("Policy name? (no spaces):")
    except KeyboardInterrupt:
        exit()
    pickle.dump(Q, open("learn_to_return/policies/"+policy_name+".p", "wb"))
    pickle.dump(G_to_idx, open("learn_to_return/policies/"+policy_name+"_states.p", "wb"))

    # animate_cluster_sequence(G, actions_taken, FPS=15, STEP_T=0.5, save_static_figures = False, filename="learn_to_return/policies/training_"+policy_name+".mp4")

if __name__ == "__main__":
    main_function_training()
