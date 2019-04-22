from itertools import product

import numpy as np
import scipy.sparse as sp

from graph_connectivity.optimization_wrappers import Constraint

def generate_dynamic_constraints(problem):

    #Define number of variables
    if problem.num_vars == None: problem.compute_num_var()

    #Setup constraints
    c_44 = _dynamic_constraint_44(problem)
    c_45 = _dynamic_constraint_45(problem)
    c_46 = _dynamic_constraint_46(problem)
    c_static = _dynamic_constraint_static(problem)
    c_ex = _dynamic_constraint_ex(problem)

    return c_44 & c_45 & c_46 & c_static & c_ex

def generate_initial_constraints(problem):
    '''constraints on z'''

    # Constructing A_eq and b_eq for initial condition as sp.coo matrix
    A_init_row  = []
    A_init_col  = []
    A_init_data = []
    b_init = []

    constraint_idx = 0
    for r, v in product(problem.graph.agents, problem.graph.nodes):
        A_init_row.append(constraint_idx)
        A_init_col.append(problem.get_z_idx(r, v, 0))
        A_init_data.append(1)
        b_init.append(1 if problem.graph.agents[r] == v else 0)
        constraint_idx += 1
    A_init = sp.coo_matrix((A_init_data, (A_init_row, A_init_col)), shape=(constraint_idx, problem.num_vars))

    return Constraint(A_eq=A_init, b_eq=b_init)

def generate_optim_constraints(problem):
    '''constraints on z, y'''

    return _dynamic_constraint_51(problem)


##########################################################
##########################################################

def _dynamic_constraint_44(problem):
    A_eq_row  = []
    A_eq_col  = []
    A_eq_data = []

    constraint_idx = 0
    for t, v in product(range(problem.T), problem.graph.nodes):
        for r in problem.graph.agents:
            A_eq_row.append(constraint_idx)
            A_eq_col.append(problem.get_z_idx(r, v, t+1))
            A_eq_data.append(1)
        for edge in problem.graph.tran_in_edges(v):
            A_eq_row.append(constraint_idx)
            A_eq_col.append(problem.get_e_idx(edge[0], edge[1], t))
            A_eq_data.append(-1)
        constraint_idx += 1
    A_eq_44 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))

    return Constraint(A_eq=A_eq_44, b_eq=np.zeros(constraint_idx))

def _dynamic_constraint_45(problem):
    A_eq_row  = []
    A_eq_col  = []
    A_eq_data = []

    constraint_idx = 0
    for t, v in product(range(problem.T), problem.graph.nodes):
        for r in problem.graph.agents:
            A_eq_row.append(constraint_idx)
            A_eq_col.append(problem.get_z_idx(r, v, t))
            A_eq_data.append(1)
        for edge in problem.graph.tran_out_edges(v):
            A_eq_row.append(constraint_idx)
            A_eq_col.append(problem.get_e_idx(edge[0], edge[1], t))
            A_eq_data.append(-1)
        constraint_idx += 1
    A_eq_45 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))

    return Constraint(A_eq=A_eq_45, b_eq=np.zeros(constraint_idx))

def _dynamic_constraint_46(problem):
    # Constructing A_eq and b_eq for identity dynamics as sp.coo matrix
    A_iq_row  = []
    A_iq_col  = []
    A_iq_data = []

    constraint_idx = 0
    for t, v, r in product(range(problem.T), problem.graph.nodes, problem.graph.agents):
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_z_idx(r, v, t))
        A_iq_data.append(1)
        for edge in problem.graph.tran_out_edges(v):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_z_idx(r, edge[1], t+1))
            A_iq_data.append(-1)
        constraint_idx += 1
    A_iq_46 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))

    return Constraint(A_iq=A_iq_46, b_iq=np.zeros(constraint_idx))

def _dynamic_constraint_static(problem):
    # Constructing A_eq and b_eq for dynamic condition on static agents as sp.coo matrix
    A_stat_row  = []
    A_stat_col  = []
    A_stat_data = []
    b_stat = []

    constraint_idx = 0
    for t, r, v in product(range(problem.T+1), problem.static_agents,
                           problem.graph.nodes):
        A_stat_row.append(constraint_idx)
        A_stat_col.append(problem.get_z_idx(r, v, t))
        A_stat_data.append(1)
        b_stat.append(1 if problem.graph.agents[r] == v else 0)
        constraint_idx += 1
    A_stat = sp.coo_matrix((A_stat_data, (A_stat_row, A_stat_col)), shape=(constraint_idx, problem.num_vars))#.toarray(

    return Constraint(A_eq=A_stat, b_eq=b_stat)

def _dynamic_constraint_ex(problem):
    # Constructing A_eq and b_eq for equality for agent existence as sp.coo matrix
    A_eq_row  = []
    A_eq_col  = []
    A_eq_data = []

    constraint_idx = 0
    for t, r in product(range(problem.T+1), problem.graph.agents):
        for v in problem.graph.nodes:
            A_eq_row.append(constraint_idx)
            A_eq_col.append(problem.get_z_idx(r, v, t))
            A_eq_data.append(1)
        constraint_idx += 1
    A_eq_ex = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))

    return Constraint(A_eq=A_eq_ex, b_eq=np.ones(constraint_idx))

def _dynamic_constraint_51(problem):
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
        for r in range(len(problem.graph.agents)):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_z_idx(r, v, problem.T))
            A_iq_data.append(-1)
        constraint_idx += 1

    A_iq_51 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))

    return Constraint(A_iq=A_iq_51, b_iq=np.zeros(constraint_idx))
