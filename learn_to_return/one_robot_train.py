import numpy as np
import time
import random
import pickle
import sys

from colorama import Fore, Style

from cops.clustering import ClusterProblem
from cops.explore_problem import ExplorationProblem
from cops.agent_problem import AgentProblem
from copy import deepcopy
from cops.animate import animate_cluster_sequence, animate_sequence

from graph_examples import get_small_graph

MAXITER = 50
NUM_EPOCHS = 1000
EPSILON = 0.5
LEARNING_RATE = 0.9
GAMMA = 0.99 # discount

START_POSITIONS = {r:0 for r in range(2)}
EAGENTS = [r for r in range(2)]
MASTER = 0

Q = {} # Q[state][action] = Q-value
G_to_idx = {}
actions = ["explore", "return"]

def quick_print(G):
    nodes_known = [1 if G.nodes[v]["known"] else 0 for v in G.nodes]
    print(sum(nodes_known), "out of", len(nodes_known), "graph nodes explored")
    print("agent(s) at:",G.agents)

def reward_func2(G_f,last_count_known):
    # TODO this rewards if any agent can transfer to the base (1 agent setting this makes sense)
    for a in G_f.agents:
        if a == MASTER:
            continue
        if G_f.agents[a] == MASTER or G_f.has_conn_edge(G_f.agents[a],MASTER):
            return G_f.count_known()-last_count_known
    else:
        return 0

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
        last_count_known = 1

        for i in range(MAXITER):

            frontiers = {v: 1 for v in G.nodes if G.is_frontier(v)}
            G.set_frontiers(frontiers)
            # print("frontiers",frontiers)

            if G not in Q:
                Q[G] = {action:0 for action in actions}
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

            ap = AgentProblem()
            ap.graph = G_new
            ap.static_agents = [MASTER]

            if action == "explore":
                ap.solve_explore()
            elif action == "return":
                ap.solve_return()

            if G_new not in Q:
                Q[G_new] = {action:0 for action in actions}
                G_to_idx[G_new] = len(G_to_idx)

            R = reward_func2(G_new,last_count_known)
            # if G_to_idx[G] == 0:
            #     print(f"Reward {R} received for taking action {action} at state {G_to_idx[G]}")
            Q[G][action] = Q[G][action] + LEARNING_RATE * (R + (GAMMA * max(Q[G_new].values())) - Q[G][action])
            last_count_known += R

            if last_count_known == len(G_new.nodes):
                print(i,"done exploring")
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

        # print(".", end =" ")

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

    for G_state in G_to_idx:
        if G_to_idx[G_state] == 0:
            print(Q[G_state])

    try:
        policy_name = input("Policy name? (no spaces):")
    except KeyboardInterrupt:
        exit()
    pickle.dump(Q, open("learn_to_return/policies/"+policy_name+".p", "wb"))
    pickle.dump(G_to_idx, open("learn_to_return/policies/"+policy_name+"_states.p", "wb"))

    # animate_cluster_sequence(G, actions_taken, FPS=15, STEP_T=0.5, save_static_figures = False, filename="learn_to_return/policies/training_"+policy_name+".mp4")

if __name__ == "__main__":
    main_function_training()
