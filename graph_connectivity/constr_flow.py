from itertools import product

import numpy as np
import scipy.sparse as sp

from graph_connectivity.optimization_wrappers import Constraint

def generate_flow_bridge_constraints(problem):

    c_48 = _dynamic_constraint_48(problem)
    c_49 = _dynamic_constraint_49(problem)
    c_50 = _dynamic_constraint_50(problem)

    return c_48 & c_49 & c_50

def generate_flow_connectivity_constraints(problem):

    c_52_53 = _dynamic_constraint_52_53(problem)

    return c_52_53

def generate_flow_master_constraints(problem):

    c_48 = _dynamic_constraint_48_m(problem)
    c_49 = _dynamic_constraint_49_m(problem)
    c_54 = _dynamic_constraint_54(problem)
    c_55 = _dynamic_constraint_55(problem)
    c_59 = _dynamic_constraint_59(problem)

    return c_48 & c_49 & c_54 & c_55 #& c_59

##########################################################
##########################################################

def _dynamic_constraint_48(problem):
    # Constructing A_eq and b_eq for equality (48) as sp.coo matrix
    A_iq_row  = []
    A_iq_col  = []
    A_iq_data = []

    N = len(problem.graph.agents)

    constraint_idx = 0
    for t, b, (v1, v2) in product(range(problem.T+1), range(problem.num_min_src_snk),
                                  problem.graph.conn_edges()):
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_fbar_idx(b, v1, v2, t))
        A_iq_data.append(1)
        for r in problem.graph.agents:
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_z_idx(r, v1, t))
            A_iq_data.append(-N)
        constraint_idx += 1

    for t, b, (v1, v2) in product(range(problem.T+1), range(problem.num_min_src_snk),
                                  problem.graph.conn_edges()):
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_fbar_idx(b, v1, v2, t))
        A_iq_data.append(1)
        for r in problem.graph.agents:
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_z_idx(r, v2, t))
            A_iq_data.append(-N)
        constraint_idx += 1

    A_iq_48 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))
    # print("Constraint 48: {}".format(A_iq_48.shape[0]))
    return Constraint(A_iq=A_iq_48, b_iq=np.zeros(constraint_idx))

def _dynamic_constraint_48_m(problem):
    # Constructing A_eq and b_eq for equality (48) for master as sp.coo matrix
    A_iq_row  = []
    A_iq_col  = []
    A_iq_data = []

    N = len(problem.graph.agents)

    constraint_idx = 0
    for t, (v1, v2) in product(range(problem.T+1), problem.graph.conn_edges()):
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_mbar_idx(v1, v2, t))
        A_iq_data.append(1)
        for r in problem.graph.agents:
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_z_idx(r, v1, t))
            A_iq_data.append(-N)
        constraint_idx += 1

    for t, (v1, v2) in product(range(problem.T+1), problem.graph.conn_edges()):
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_mbar_idx(v1, v2, t))
        A_iq_data.append(1)
        for r in problem.graph.agents:
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_z_idx(r, v2, t))
            A_iq_data.append(-N)
        constraint_idx += 1

    A_iq_48 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))
    # print("Constraint 48m: {}".format(A_iq_48.shape[0]))
    return Constraint(A_iq=A_iq_48, b_iq=np.zeros(constraint_idx))

def _dynamic_constraint_49(problem):
    # Constructing A_eq and b_eq for equality (49) as sp.coo matrix
    A_iq_row  = []
    A_iq_col  = []
    A_iq_data = []

    N = len(problem.graph.agents)

    constraint_idx = 0
    for t, b, (v1, v2) in product(range(problem.T), range(problem.num_min_src_snk),
                                  problem.graph.tran_edges()):
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_f_idx(b, v1, v2, t))
        A_iq_data.append(1)
        for r in problem.graph.agents:
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_xf_idx(r, v1, v2, t))
            A_iq_data.append(-N)
        constraint_idx += 1

    A_iq_49 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))
    # print("Constraint 49: {}".format(A_iq_49.shape[0]))
    return Constraint(A_iq=A_iq_49, b_iq=np.zeros(constraint_idx))

def _dynamic_constraint_49_m(problem):
    # Constructing A_eq and b_eq for equality (49) as sp.coo matrix
    A_iq_row  = []
    A_iq_col  = []
    A_iq_data = []

    N = len(problem.graph.agents)

    constraint_idx = 0
    for t, (v1, v2) in product(range(problem.T), problem.graph.tran_edges()):
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_m_idx(v1, v2, t))
        A_iq_data.append(1)
        for r in problem.graph.agents:
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_xf_idx(r, v1, v2, t))
            A_iq_data.append(-N)
        constraint_idx += 1

    A_iq_49 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))
    # print("Constraint 49m: {}".format(A_iq_49.shape[0]))
    return Constraint(A_iq=A_iq_49, b_iq=np.zeros(constraint_idx))

def _dynamic_constraint_50(problem):
    '''constraint on z, y'''

    A_iq_row  = []
    A_iq_col  = []
    A_iq_data = []

    N = len(problem.graph.agents)

    constraint_idx = 0
    for v in problem.graph.nodes:
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_y_idx(v))
        A_iq_data.append(1)
        for r in problem.graph.agents:
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_z_idx(r, v, problem.T))
            A_iq_data.append(-1)
        constraint_idx += 1

    A_iq_50 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))
    # print("Constraint 50: {}".format(A_iq_50.shape[0]))
    return Constraint(A_iq=A_iq_50, b_iq=np.zeros(constraint_idx))

def _dynamic_constraint_52_53(problem):
    # Constructing A_eq and b_eq for equality (52,53) as sp.coo matrix
    A_eq_row  = []
    A_eq_col  = []
    A_eq_data = []

    constraint_idx = 0
    for t, v, (b, b_r) in product(range(problem.T+1), problem.graph.nodes,
                                  enumerate(problem.min_src_snk)):
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

        if len(problem.src) <= len(problem.snk):
            # case (52)
            if t == 0:
                A_eq_row.append(constraint_idx)
                A_eq_col.append(problem.get_z_idx(b_r, v, t))
                A_eq_data.append(len(problem.snk))
            elif t == problem.T:
                for r in problem.snk:
                    A_eq_row.append(constraint_idx)
                    A_eq_col.append(problem.get_z_idx(r, v, t))
                    A_eq_data.append(-1)
        else:
            # case (53)
            if t == 0:
                for r in problem.src:
                    A_eq_row.append(constraint_idx)
                    A_eq_col.append(problem.get_z_idx(r, v, t))
                    A_eq_data.append(1)
            elif t == problem.T:
                A_eq_row.append(constraint_idx)
                A_eq_col.append(problem.get_z_idx(b_r, v, t))
                A_eq_data.append(-len(problem.src))

        constraint_idx += 1
    A_eq_52 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))
    # print("Constraint 52: {}".format(A_eq_52.shape[0]))
    return Constraint(A_eq=A_eq_52, b_eq=np.zeros(constraint_idx))

def _dynamic_constraint_54(problem):
    # Constructing A_eq and b_eq for equality (55) as sp.coo matrix
    A_iq_row  = []
    A_iq_col  = []
    A_iq_data = []
    b_iq_54 = []

    v0 = problem.graph.agents[problem.master]

    constraint_idx = 0
    for t, v in product(range(problem.T+1), problem.graph.nodes):

        if t > 0:
            for edge in problem.graph.tran_in_edges(v):
                A_iq_row.append(constraint_idx)
                A_iq_col.append(problem.get_m_idx(edge[0], edge[1], t-1))
                A_iq_data.append(-1)

        for edge in problem.graph.conn_in_edges(v):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_mbar_idx(edge[0], edge[1], t))
            A_iq_data.append(-1)

        if t < problem.T:
            for edge in problem.graph.tran_out_edges(v):
                A_iq_row.append(constraint_idx)
                A_iq_col.append(problem.get_m_idx(edge[0], edge[1], t))
                A_iq_data.append(1)

        for edge in problem.graph.conn_out_edges(v):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_mbar_idx(edge[0], edge[1], t))
            A_iq_data.append(1)

        if t == 0 and v == v0:
            b_iq_54.append(len(problem.graph.agents) - 1)
        else:
            b_iq_54.append(0)
        constraint_idx += 1

    A_iq_54 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))
    # print("Constraint 54: {}".format(A_iq_54.shape[0]))
    return Constraint(A_iq=A_iq_54, b_iq=b_iq_54)

def _dynamic_constraint_55(problem):
    # Constructing A_eq and b_eq for equality (55) as sp.coo matrix
    A_iq_row  = []
    A_iq_col  = []
    A_iq_data = []
    b_iq_55 = []

    constraint_idx = 0
    m_v0 = problem.graph.agents[problem.master]
    for t, r in product(range(problem.T+1), problem.graph.agents):
        v0 = problem.graph.agents[r]
        if r != problem.master and v0 != m_v0:
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_z_idx(r, v0, t))
            A_iq_data.append(-1)

            for tau in range(t):
                # z_t is locked unless info arrived at some point before t-1
                if tau > 0:
                    for edge in problem.graph.tran_in_edges(v0):
                        A_iq_row.append(constraint_idx)
                        A_iq_col.append(problem.get_m_idx(edge[0], edge[1], tau-1))
                        A_iq_data.append(-1)

                for edge in problem.graph.conn_in_edges(v0):
                    A_iq_row.append(constraint_idx)
                    A_iq_col.append(problem.get_mbar_idx(edge[0], edge[1], tau))
                    A_iq_data.append(-1)

                if tau < problem.T:
                    for edge in problem.graph.tran_out_edges(v0):
                        A_iq_row.append(constraint_idx)
                        A_iq_col.append(problem.get_m_idx(edge[0], edge[1], tau))
                        A_iq_data.append(1)

                for edge in problem.graph.conn_out_edges(v0):
                    A_iq_row.append(constraint_idx)
                    A_iq_col.append(problem.get_mbar_idx(edge[0], edge[1], tau))
                    A_iq_data.append(1)

            b_iq_55.append(-1)

            constraint_idx += 1

    A_iq_55 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))
    # print("Constraint 55: {}".format(A_iq_55.shape[0]))
    return Constraint(A_iq=A_iq_55, b_iq=b_iq_55)


def _dynamic_constraint_59(problem):
    # Constructing A_eq and b_eq for equality (59) as sp.coo matrix
    A_iq_row  = []
    A_iq_col  = []
    A_iq_data = []

    constraint_idx = 0
    m_v0 = problem.graph.agents[problem.master]
    for v in problem.graph.nodes:

            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_y_idx(v))
            A_iq_data.append(1)

            for tau in range(problem.T+1):
                if tau > 0:
                    for edge in problem.graph.tran_in_edges(v):
                        A_iq_row.append(constraint_idx)
                        A_iq_col.append(problem.get_m_idx(edge[0], edge[1], tau-1))
                        A_iq_data.append(-1)

                for edge in problem.graph.conn_in_edges(v):
                    A_iq_row.append(constraint_idx)
                    A_iq_col.append(problem.get_mbar_idx(edge[0], edge[1], tau))
                    A_iq_data.append(-1)

                if tau < problem.T:
                    for edge in problem.graph.tran_out_edges(v):
                        A_iq_row.append(constraint_idx)
                        A_iq_col.append(problem.get_m_idx(edge[0], edge[1], tau))
                        A_iq_data.append(1)

                for edge in problem.graph.conn_out_edges(v):
                    A_iq_row.append(constraint_idx)
                    A_iq_col.append(problem.get_mbar_idx(edge[0], edge[1], tau))
                    A_iq_data.append(1)

            constraint_idx += 1

    A_iq_59 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))
    # print("Constraint 59: {}".format(A_iq_59.shape[0]))
    return Constraint(A_iq=A_iq_59, b_iq=np.zeros(constraint_idx))
