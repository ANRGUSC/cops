from itertools import product

import numpy as np
import scipy.sparse as sp

from cops.optimization_wrappers import Constraint


def generate_powerset_bridge_constraints(problem):
    """constraints on z, e, x, xbar, yb"""

    c_30 = _dynamic_constraint_30(problem)
    c_33 = _dynamic_constraint_33(problem)
    c_34 = _dynamic_constraint_34(problem)
    c_35 = _dynamic_constraint_35(problem)
    c_36 = _dynamic_constraint_36(problem)

    return c_30 & c_33 & c_34 & c_35 & c_36


def generate_connectivity_constraint_all(problem):
    """Generate connectivity constraints corresponding to all subsets
       in the graph, for all bases"""

    if problem.num_vars == None:
        problem.compute_num_var()

    ret = Constraint()

    # Iterator over all (v, t) subsets in the graph
    for b, b_r in enumerate(problem.src):
        # Convert each set in the iterator to (v,t) format
        add_S = map(
            lambda S: list(map(problem.get_time_augmented_n_t, S)),
            problem.powerset_exclude_agent(b_r),
        )
        ret &= generate_connectivity_constraint(problem, [b], add_S)

    return ret


def generate_connectivity_constraint(problem, b_list, add_S):
    """Generate connectivity constraints for the S subsets in add_S, for
       all bases in b_list

       NOTE: b_list are INDICES in problem.src, not robots"""

    # Constructing A_iq and b_iq for inequality (38) for all S in add_S as sp.coo matrix
    A_iq_row = []
    A_iq_col = []
    A_iq_data = []

    constraint_idx = 0
    # For each base
    for b, S_v_t in product(b_list, add_S):
        pre_S_transition = problem.graph.pre_tran_vt(S_v_t)
        pre_S_connectivity = problem.graph.pre_conn_vt(S_v_t)
        for v, t in S_v_t:
            # add y
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_yb_idx(b, v, t))
            A_iq_data.append(1)
            for v0, v1, t0 in pre_S_transition:
                A_iq_row.append(constraint_idx)
                A_iq_col.append(problem.get_x_idx(b, v0, v1, t0))
                A_iq_data.append(-1)
            for v0, v1, t1 in pre_S_connectivity:
                A_iq_row.append(constraint_idx)
                A_iq_col.append(problem.get_xbar_idx(b, v0, v1, t1))
                A_iq_data.append(-1)
            constraint_idx += 1
    A_iq_38 = sp.coo_matrix(
        (A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars)
    )

    return Constraint(A_iq=A_iq_38, b_iq=np.zeros(constraint_idx))


##########################################################
##########################################################


def _dynamic_constraint_30(problem):
    A_iq_row = []
    A_iq_col = []
    A_iq_data = []

    constraint_idx = 0
    for t, b, (v1, v2) in product(
        range(problem.T), range(problem.num_src), problem.graph.tran_edges()
    ):
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_x_idx(b, v1, v2, t))
        A_iq_data.append(1)

        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_e_idx(v1, v2, t))
        A_iq_data.append(-1)
        constraint_idx += 1
    A_iq_30 = sp.coo_matrix(
        (A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars)
    )

    return Constraint(A_iq=A_iq_30, b_iq=np.zeros(constraint_idx))


def _dynamic_constraint_33(problem):
    # Constructing A_eq and b_eq for equality (33) as sp.coo matrix
    A_iq_row = []
    A_iq_col = []
    A_iq_data = []

    constraint_idx = 0
    for t, (v1, v2), b in product(
        range(problem.T + 1), problem.graph.conn_edges(), range(problem.num_src)
    ):
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_xbar_idx(b, v1, v2, t))
        A_iq_data.append(1)
        for r in range(len(problem.graph.agents)):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_z_idx(r, v1, t))
            A_iq_data.append(-1)
        constraint_idx += 1

    A_iq_33 = sp.coo_matrix(
        (A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars)
    )

    return Constraint(A_iq=A_iq_33, b_iq=np.zeros(constraint_idx))


def _dynamic_constraint_34(problem):
    # Constructing A_eq and b_eq for equality (34) as sp.coo matrix
    A_iq_row = []
    A_iq_col = []
    A_iq_data = []

    constraint_idx = 0
    for t, (v1, v2), b in product(
        range(problem.T + 1), problem.graph.conn_edges(), range(problem.num_src)
    ):
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_xbar_idx(b, v1, v2, t))
        A_iq_data.append(1)
        for r in range(len(problem.graph.agents)):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_z_idx(r, v2, t))
            A_iq_data.append(-1)
        constraint_idx += 1

    A_iq_34 = sp.coo_matrix(
        (A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars)
    )

    return Constraint(A_iq=A_iq_34, b_iq=np.zeros(constraint_idx))


def _dynamic_constraint_35(problem):
    # Constructing A_eq and b_eq for equality (35) as sp.coo matrix
    A_iq_row = []
    A_iq_col = []
    A_iq_data = []

    constraint_idx = 0
    for t, v, b in product(
        range(problem.T + 1), problem.graph.nodes, range(problem.num_src)
    ):
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_yb_idx(b, v, t))
        A_iq_data.append(1)
        for r in range(len(problem.graph.agents)):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_z_idx(r, v, t))
            A_iq_data.append(-1)
        constraint_idx += 1

    A_iq_35 = sp.coo_matrix(
        (A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars)
    )

    return Constraint(A_iq=A_iq_35, b_iq=np.zeros(constraint_idx))


def _dynamic_constraint_36(problem):
    # Constructing A_iq and b_iq for equality (36) as sp.coo matrix
    A_iq_row = []
    A_iq_col = []
    A_iq_data = []

    N = len(problem.graph.agents)

    constraint_idx = 0
    for v, b in product(problem.graph.nodes, range(problem.num_src)):
        for r in problem.graph.agents:
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_z_idx(r, v, problem.T))
            A_iq_data.append(1)
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_yb_idx(b, v, problem.T))
        A_iq_data.append(-N)
        constraint_idx += 1

    A_iq_36 = sp.coo_matrix(
        (A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars)
    )

    return Constraint(A_iq=A_iq_36, b_iq=np.zeros(constraint_idx))
