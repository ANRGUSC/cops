from cops.problem import *
from cops.animate import *
import csv
import math


def create_linear_graph(n):
    G = Graph()
    G.add_transition_path(list(range(n)))
    G.add_connectivity_path(list(range(n)))
    G.set_node_positions({i: (0, i) for i in range(n)})
    agent_positions = {0: 0, 1: n // 2, 2: n - 1}
    G.init_agents(agent_positions)
    return G

def create_quadratic_graph(n):
    G = Graph()
    positions = {}
    for i in range(n):
        row = [r+n*i for r in range(n)]
        for r in range(n):
            positions[r+n*i] = (r, i)
    for i in range(n):
        G.add_connectivity_path([i+n*j for j in range(n)])
        G.add_transition_path([i+n*j for j in range(n)])
    for j in range(n):
        G.add_connectivity_path([i+n*j for i in range(n)])
        G.add_transition_path([i+n*j for i in range(n)])
    G.set_node_positions(positions)
    agent_positions = {0: 0, 1: n*n-1}
    G.init_agents(agent_positions)
    return G

def run_linear_powerset(n):
    # Set up the connectivity problem
    G = create_linear_graph(n)
    cp = ConnectivityProblem()
    cp.graph = G  # graph
    cp.T = int(n / 2) - 1  # time
    # Solve
    t0 = time.time()
    cp.solve_powerset()
    return time.time() - t0


def run_linear_adaptive(n):
    # Set up the connectivity problem
    G = create_linear_graph(n)
    cp = ConnectivityProblem()
    cp.graph = G
    cp.T = int(n / 2) - 1
    t0 = time.time()
    cp.solve_adaptive()
    return time.time() - t0


def run_linear_flow(n):
    # Set up the connectivity problem
    G = create_linear_graph(n)
    cp = ConnectivityProblem()
    cp.graph = G  # graph
    cp.T = int(n / 2) - 1  # time
    # Solve
    t0 = time.time()
    cp.solve_flow()
    return time.time() - t0


def run_quadratic_powerset(n):
    G = create_quadratic_graph(n)
    cp = ConnectivityProblem()
    cp.graph = G
    cp.T = 2*(n-1) - 1
    t0 = time.time()
    cp.solve_powerset()
    return time.time() - t0


def run_quadratic_adaptive(n):
    G = create_quadratic_graph(n)
    cp = ConnectivityProblem()
    cp.graph = G
    cp.T = 2*(n-1) - 1
    t0 = time.time()
    cp.solve_adaptive()
    return time.time() - t0


def run_quadratic_flow(n):
    G = create_quadratic_graph(n)
    cp = ConnectivityProblem()
    cp.graph = G
    cp.T = 2*(n-1) - 1
    t0 = time.time()
    cp.solve_flow()
    return time.time() - t0


time_powerset = [["x", "y", "logy"]]
time_adaptive = [["x", "y", "logy"]]
time_flow = [["x", "y", "logy"]]
N = 6
step = 1
for n in range(2, N, step):

    if n == 3:
        # Plot graph
        G = create_quadratic_graph(n)
        G.plot_graph()

    if n < 4:
        print("Solving powerset for n = ", n)
        runtime = run_quadratic_powerset(n)
        time_powerset.append([n*n, runtime, math.log10(runtime)])

    if n < 5:
        print("Solving adaptive for n = ", n)
        runtime = run_quadratic_adaptive(n)
        time_adaptive.append([n*n, runtime, math.log10(runtime)])

    print("Solving flow for n = ", n)
    runtime = run_quadratic_flow(n)
    time_flow.append([n*n, runtime, math.log10(runtime)])

with open("quad_powerset.csv", "w") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(time_powerset)

with open("quad_adaptive.csv", "w") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(time_adaptive)

with open("quad_flow.csv", "w") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(time_flow)


# Animate solution
# animate(G, cp.traj, cp.conn)
