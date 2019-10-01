from cops.problem import *
from cops.animate import *
import csv
import math


def create_linear_graph(n):
    # Define a connectivity graph
    G = Graph()
    G.add_transition_path(list(range(n)))
    G.add_connectivity_path(list(range(n)))
    G.set_node_positions({i: (0, i) for i in range(n)})
    agent_positions = {0: 0, 1: n // 2, 2: n - 1}
    G.init_agents(agent_positions)
    return G


def run_powerset(n):
    # Set up the connectivity problem
    G = create_linear_graph(n)
    cp = ConnectivityProblem()
    cp.graph = G  # graph
    cp.T = int(n / 2) - 1  # time
    # Solve
    t0 = time.time()
    cp.solve_powerset()
    return time.time() - t0


def run_adaptive(n):
    # Set up the connectivity problem
    G = create_linear_graph(n)
    cp = ConnectivityProblem()
    cp.graph = G
    cp.T = int(n / 2) - 1
    t0 = time.time()
    cp.solve_adaptive()
    return time.time() - t0


def run_flow(n):
    # Set up the connectivity problem
    G = create_linear_graph(n)
    cp = ConnectivityProblem()
    cp.graph = G  # graph
    cp.T = int(n / 2) - 1  # time
    # Solve
    t0 = time.time()
    cp.solve_flow()
    return time.time() - t0


time_powerset = [["x", "y", "logy"]]
time_adaptive = [["x", "y", "logy"]]
time_flow = [["x", "y", "logy"]]
N = 50
step = 1
for n in range(3, N, step):

    if n == 10:
        # Plot graph
        G = create_linear_graph(n)
        G.plot_graph()

    if n < 7:
        print("Solving powerset for n = ", n)
        runtime = run_powerset(n)
        time_powerset.append([n, runtime, math.log10(runtime)])

    if n < 13:
        print("Solving adaptive for n = ", n)
        runtime = run_adaptive(n)
        time_adaptive.append([n, runtime, math.log10(runtime)])

    print("Solving flow for n = ", n)
    runtime = run_flow(n)
    time_flow.append([n, runtime, math.log10(runtime)])

with open("powerset.csv", "w") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(time_powerset)

with open("adaptive.csv", "w") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(time_adaptive)

with open("flow.csv", "w") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(time_flow)


# Animate solution
# animate(G, cp.traj, cp.conn)
