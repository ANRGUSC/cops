from one_robot_train import *

START_POSITIONS = {r:0 for r in range(2)}
EAGENTS = [r for r in range(2)]
MASTER = 0

TMAX = 50 # max number of turns

def main_function_testing(policy_name=None):

    if policy_name is None:
        try:
            policy_name = input("Policy name? (no spaces):")
        except KeyboardInterrupt:
            sys.exit()

    Q = pickle.load(open("learn_to_return/policies/"+policy_name+".p", "rb")) # Q[state][action] = Q-value
    G_to_idx = pickle.load(open("learn_to_return/policies/"+policy_name+"_states.p", "rb"))
    actions = ["explore", "return"]

    #### STATE
    G = get_small_graph() # 16 vertices
    agent_positions = START_POSITIONS
    G.init_agents(agent_positions)

    # Set known attribute
    for v in G.nodes():
        G.nodes[v]["known"] = False
    for r, v in agent_positions.items():
        G.nodes[v]["known"] = True

    actions_taken = []
    last_count_known = 1

    for i in range(TMAX):

        frontiers = {v: 1 for v in G.nodes if G.is_frontier(v)}
        G.set_frontiers(frontiers)
        # print("frontiers",frontiers)

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
                    q = Q[G][a]

        #### TAKE ACTION
        G_new = deepcopy(G)

        ap = AgentProblem()
        ap.graph = G_new
        ap.static_agents = [MASTER]

        if action == "explore":
            print("exploring")
            ap.solve_explore()
        elif action == "return":
            print("returning")
            ap.solve_return()

        actions_taken.append(ap)
        last_count_known += reward_func2(G_new,last_count_known)

        if last_count_known == len(G_new.nodes):
            print(i)
            print("done exploring")
            break

        G = G_new

    if last_count_known < len(G.nodes):
        print(f"{G.count_known()}/{len(G.nodes)} nodes explored after {TMAX} turns")

    animate_sequence(G, actions_taken, FPS=15, STEP_T=0.5, save_static_figures = False, filename="learn_to_return/policies/"+policy_name+".mp4", extra_title_info="\nepochs=1000, gamma=0.99")

if __name__=="__main__":
    if len(sys.argv) > 1:
        main_function_testing(sys.argv[1])
    else:
        main_function_testing()
