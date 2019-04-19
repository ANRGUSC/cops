from itertools import product

import numpy as np
import scipy.sparse as sp

from graph_connectivity.optimization_wrappers import Constraint

def generate_flow_contraints(problem):
    '''constraints on z, e, f, fbar'''

    c_48 = _dynamic_constraint_48(problem)
    c_49 = _dynamic_constraint_49(problem)
    c_50 = _dynamic_constraint_50(problem)
    c_52 = _generate_flow_constraint_52(problem)

    return c_48 & c_49 & c_50 & c_52

##########################################################
##########################################################

def _dynamic_constraint_48(problem):
    # Constructing A_eq and b_eq for equality (48) as sp.coo matrix
    A_iq_row  = []
    A_iq_col  = []
    A_iq_data = []

    N = len(problem.graph.agents)

    constraint_idx = 0
    for t, b, (v1, v2) in product(range(problem.T+1), problem.b,
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
    for t, b, (v1, v2) in product(range(problem.T+1), problem.b,
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
    for t, b, (v1, v2) in product(range(problem.T), problem.b,
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
    b_eq_flow = []

    N = len(problem.graph.agents)

    constraint_idx = 0
    for t, v, b in product(range(problem.T+1), problem.graph.nodes, problem.b):
        for edge in problem.graph.in_edges(v, data = True):
            if edge[2]['type'] == 'transition' and t > 0:
                A_eq_row.append(constraint_idx)
                A_eq_col.append(problem.get_f_idx(b, edge[0], edge[1], t-1))
                A_eq_data.append(1)
            elif edge[2]['type'] == 'connectivity':
                A_eq_row.append(constraint_idx)
                A_eq_col.append(problem.get_fbar_idx(b, edge[0], edge[1], t))
                A_eq_data.append(1)
        for edge in problem.graph.out_edges(v, data = True):
            if edge[2]['type'] == 'transition' and t < problem.T:
                A_eq_row.append(constraint_idx)
                A_eq_col.append(problem.get_f_idx(b, edge[0], edge[1], t))
                A_eq_data.append(-1)
            elif edge[2]['type'] == 'connectivity':
                A_eq_row.append(constraint_idx)
                A_eq_col.append(problem.get_fbar_idx(b, edge[0], edge[1], t))
                A_eq_data.append(-1)
        if t == 0 and v == problem.b[b]:
            b_eq_flow.append(-N)
        elif t == problem.T:
            for r in problem.graph.agents:
                A_eq_row.append(constraint_idx)
                A_eq_col.append(problem.get_z_idx(r, v, t))
                A_eq_data.append(-1)
            b_eq_flow.append(0)
        else:
            b_eq_flow.append(0)
        constraint_idx += 1
    A_eq_52 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))

    return Constraint(A_eq=A_eq_52, b_eq=b_eq_flow)