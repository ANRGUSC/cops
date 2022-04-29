from centralized_train import *

START_POSITIONS = {r:0 for r in range(2)}
EAGENTS = [r for r in range(2)]
MASTER = 0

TMAX = 20 # max number of turns

def main_function_testing(policy_name=None):

    if policy_name is None:
        try:
            policy_name = input("Policy name? (no spaces):")
        except KeyboardInterrupt:
            sys.exit()

    Q = pickle.load(open("learn_to_return/policies/"+policy_name+".p", "rb")) # Q[state][action] = Q-value
    G_to_idx = pickle.load(open("learn_to_return/policies/"+policy_name+"_states.p", "rb"))
    all_actions = ["to_frontier", "explore 4", "explore 8", "explore 16", "return"]

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

    for i in range(TMAX):
        print(i)

        frontiers = {v: 2 for v in G.nodes if G.is_frontier(v)}
        print("Frontiers:",frontiers)
        G.set_frontiers(frontiers)

        #### LIMIT ACTION SPACE
        if action_state_machine == "at_base":
            actions = ["to_frontier"]
        elif action_state_machine == "at_frontier":
            actions = [action for action in all_actions if "explore" in action]#+["return"]
        elif action_state_machine == "explored":
            actions = ["return"]#+[action for action in all_actions if "explore" in action]

        if G not in Q:
            #### EXPLORE
            print("new state, acting randomly")
            action = random.choice(actions)
        else:
            #### EXPLOIT
            print(G_to_idx[G], Q[G])
            q = 0
            action = random.choice(actions)
            for a in actions:
                if Q[G][a] > q:
                    action = a

        #### TAKE ACTION
        G_new = deepcopy(G)
        print(action_state_machine,":",action)
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

        last_count_known += reward_func2(G_new,action_state_machine,last_count_known)

        if last_count_known == len(G_new.nodes):
            print(i)
            print("done exploring")
            break

        print(G == G_new)
        G = G_new

    if last_count_known < len(G.nodes):
        print(f"{G.count_known()}/{len(G.nodes)} nodes explored after {TMAX} turns")

    animate_cluster_sequence(G, actions_taken, FPS=15, STEP_T=0.5, save_static_figures = False, filename="learn_to_return/policies/"+policy_name+".mp4")

if __name__=="__main__":
    if len(sys.argv) > 1:
        main_function_testing(sys.argv[1])
    else:
        main_function_testing()
