from itertools import product

import numpy as np
import scipy.sparse as sp

from cops.optimization_wrappers import Constraint


def constraint_static_master(problem, master):

    # Constructing A_iq and b_iq for equality (59) as sp.coo matrix
    A_iq_row = []
    A_iq_col = []
    A_iq_data = []
    b_iq = []
    constraint_idx = 0

    if master in problem.static_agents:

        master_node = problem.graph.agents[master]

        # find nodes connected to master through static agents
        connected_nodes = []
        active = [master_node]
        while len(active) > 0:
            v = active.pop(0)
            connected_nodes.append(v)
            for _, v1 in problem.graph.conn_out_edges(v):
                if v1 not in connected_nodes:
                    connected_nodes.append(v1)

                    # check if any static agent in nbr
                    for r in problem.static_agents:
                        if problem.graph.agents[r] == v1:
                            active.append(v1)

        dynamic_agents = [
            r for r in problem.graph.agents if r not in problem.static_agents
        ]

        for v, r in product(connected_nodes, dynamic_agents):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_z_idx(r, v, problem.T))
            A_iq_data.append(-1)
        b_iq.append(-1)
        constraint_idx += 1

    A_iq = sp.coo_matrix(
        (A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars)
    )
    return Constraint(A_iq=A_iq, b_iq=b_iq)
