from cops.problem import ConnectivityProblem
from cops.animate import animate

from graph_examples import get_linear_graph

n = 4

G = get_linear_graph(n)

frontiers = {i: 1 for i in range(3)}
G.set_frontiers(frontiers)

agent_positions = {0: 0, 1: 2, 2: 3}
G.init_agents(agent_positions)

# Set up the connectivity problem
cp = ConnectivityProblem()
cp.graph = G
cp.T = 6
cp.static_agents = []
cp.master = 0
cp.src = [2]
cp.snk = [1]

# Solve
cp.solve_flow(master=True, frontier_reward=True, connectivity=True, cut=True, solver='gurobi')

# Animate solution
animate(G, cp.traj, cp.conn)
