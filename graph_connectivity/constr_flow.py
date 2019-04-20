from itertools import product

import numpy as np
import scipy.sparse as sp

from graph_connectivity.optimization_wrappers import Constraint

def generate_flow_constraints(problem):
    '''constraints on z, e, f, fbar'''

    c_48 = _dynamic_constraint_48(problem)
    c_49 = _dynamic_constraint_49(problem)
    c_50 = _dynamic_constraint_50(problem)
    c_52_53 = _generate_flow_constraint_52(problem)

    return c_48 & c_49 & c_50 & c_52_53

##########################################################
##########################################################

def _dynamic_constraint_48(problem):
    # Constructing A_eq and b_eq for equality (48) as sp.coo matrix
    A_iq_row  = []
    A_iq_col  = []
    A_iq_data = []

    N = len(problem.graph.agents)

    constraint_idx = 0
    for t, b, (v1, v2) in product(range(problem.T+1), range(problem.num_b),
                                  problem.graph.conn_edges()):
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_fbar_idx(b, v1, v2, t))
        A_iq_data.append(1)
        for r in range(len(problem.graph.agents)):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_z_idx(r, v1, t))
            A_iq_data.append(-N)
        constraint_idx += 1

    A_iq_48 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))

    return Constraint(A_iq=A_iq_48, b_iq=np.zeros(constraint_idx))

def _dynamic_constraint_49(problem):
    # Constructing A_eq and b_eq for equality (49) as sp.coo matrix
    A_iq_row  = []
    A_iq_col  = []
    A_iq_data = []

    N = len(problem.graph.agents)

    constraint_idx = 0
    for t, b, (v1, v2) in product(range(problem.T+1), range(problem.num_b),
                                  problem.graph.conn_edges()):
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_fbar_idx(b, v1, v2, t))
        A_iq_data.append(1)
        for r in range(len(problem.graph.agents)):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_z_idx(r, v2, t))
            A_iq_data.append(-N)
        constraint_idx += 1

    A_iq_49 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))

    return Constraint(A_iq=A_iq_49, b_iq=np.zeros(constraint_idx))

def _dynamic_constraint_50(problem):
    # Constructing A_eq and b_eq for equality (56) as sp.coo matrix
    A_iq_row  = []
    A_iq_col  = []
    A_iq_data = []

    N = len(problem.graph.agents)

    constraint_idx = 0
    for t, b, (v1, v2) in product(range(problem.T), range(problem.num_b),
                                  problem.graph.tran_edges()):
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_f_idx(b, v1, v2, t))
        A_iq_data.append(1)
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_e_idx(v1, v2, t))
        A_iq_data.append(-N)
        constraint_idx += 1

    A_iq_50 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))

    return Constraint(A_iq=A_iq_50, b_iq=np.zeros(constraint_idx))

def _generate_flow_constraint_52(problem):
    # Constructing A_eq and b_eq for equality (52) as sp.coo matrix
    A_eq_row  = []
    A_eq_col  = []
    A_eq_data = []

    constraint_idx = 0
    for t, v, (b, b_r) in product(range(problem.T+1), problem.graph.nodes, 
                                  enumerate(problem.b)):
        if t > 0:
            for edge in problem.graph.tran_in_edges(v):
                A_eq_row.append(constraint_idx)
                A_eq_col.append(problem.get_f_idx(b, edge[0], edge[1], t-1))
                A_eq_data.append(1)

        for edge in problem.graph.conn_in_edges(v):
                A_eq_row.append(constraint_idx)
                A_eq_col.append(problem.get_fbar_idx(b, edge[0], edge[1], t))
                A_eq_data.append(1)

        if t < problem.T:
            for edge in problem.graph.tran_out_edges(v):
                A_eq_row.append(constraint_idx)
                A_eq_col.append(problem.get_f_idx(b, edge[0], edge[1], t))
                A_eq_data.append(-1)

        for edge in problem.graph.conn_out_edges(v):
            A_eq_row.append(constraint_idx)
            A_eq_col.append(problem.get_fbar_idx(b, edge[0], edge[1], t))
            A_eq_data.append(-1)

        if len(problem.sources) <= len(problem.sinks):
            # case (52)
            if t == 0:
                A_eq_row.append(constraint_idx)
                A_eq_col.append(problem.get_z_idx(b_r, v, t))
                A_eq_data.append(len(problem.sinks))
            elif t == problem.T:
                for r in problem.sinks:
                    A_eq_row.append(constraint_idx)
                    A_eq_col.append(problem.get_z_idx(r, v, t))
                    A_eq_data.append(-1)
        else:
            # case (53)
            if t == 0:
                for r in problem.sources:
                    A_eq_row.append(constraint_idx)
                    A_eq_col.append(problem.get_z_idx(r, v, t))
                    A_eq_data.append(1)
            elif t == problem.T:
                A_eq_row.append(constraint_idx)
                A_eq_col.append(problem.get_z_idx(b_r, v, t))
                A_eq_data.append(-len(problem.sources))

        constraint_idx += 1
    A_eq_52 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))

    return Constraint(A_eq=A_eq_52, b_eq=np.zeros(constraint_idx))
