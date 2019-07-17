from itertools import product, combinations

import numpy as np
import scipy.sparse as sp

from cops.optimization_wrappers import Constraint

def generate_powerset_dynamic_constraints(problem):

    #Define number of variables
    if problem.num_vars == None: problem.compute_num_var()

    #Setup constraints
    c_27 = _dynamic_constraint_27(problem)
    c_28 = _dynamic_constraint_28(problem)
    c_29 = _dynamic_constraint_29(problem)
    c_30 = _dynamic_constraint_30(problem)
    c_static = _dynamic_constraint_static(problem)

    return c_27 & c_28 & c_29 & c_30 & c_static

def generate_flow_dynamic_constraints(problem, static_master = True):

    #Define number of variables
    if problem.num_vars == None: problem.compute_num_var()

    #Setup constraints
    c_45 = _dynamic_constraint_45(problem)
    c_46 = _dynamic_constraint_46(problem)
    c_47 = _dynamic_constraint_47(problem)
    c_static = _dynamic_constraint_static(problem)
    c_agent_avoid = _dynamic_constraint_agent_avoidance(problem)

    return c_45 & c_46 & c_47 & c_static & c_agent_avoid

def generate_initial_constraints(problem):

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

#Powerset#################################################
##########################################################

def _dynamic_constraint_27(problem):
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
    A_eq_27 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))

    return Constraint(A_eq=A_eq_27, b_eq=np.ones(constraint_idx))

def _dynamic_constraint_28(problem):
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
    A_eq_27 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))

    return Constraint(A_eq=A_eq_27, b_eq=np.zeros(constraint_idx))

def _dynamic_constraint_29(problem):
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

def _dynamic_constraint_30(problem):
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

#Flow#####################################################
##########################################################
def _dynamic_constraint_45(problem):
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
    A_eq_45 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))
    return Constraint(A_eq=A_eq_45, b_eq=np.ones(constraint_idx))

def _dynamic_constraint_46(problem):
    A_eq_row  = []
    A_eq_col  = []
    A_eq_data = []

    constraint_idx = 0
    for t, v, r in product(range(problem.T), problem.graph.nodes, problem.graph.agents):
        A_eq_row.append(constraint_idx)
        A_eq_col.append(problem.get_z_idx(r, v, t+1))
        A_eq_data.append(1)
        for edge in problem.graph.tran_in_edges(v):
            A_eq_row.append(constraint_idx)
            A_eq_col.append(problem.get_xf_idx(r, edge[0], edge[1], t))
            A_eq_data.append(-1)
        constraint_idx += 1
    A_eq_46 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))
    return Constraint(A_eq=A_eq_46, b_eq=np.zeros(constraint_idx))

def _dynamic_constraint_47(problem):
    A_eq_row  = []
    A_eq_col  = []
    A_eq_data = []

    constraint_idx = 0
    for t, v, r in product(range(problem.T), problem.graph.nodes, problem.graph.agents):
        A_eq_row.append(constraint_idx)
        A_eq_col.append(problem.get_z_idx(r, v, t))
        A_eq_data.append(1)
        for edge in problem.graph.tran_out_edges(v):
            A_eq_row.append(constraint_idx)
            A_eq_col.append(problem.get_xf_idx(r, edge[0], edge[1], t))
            A_eq_data.append(-1)
        constraint_idx += 1
    A_eq_47 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))
    return Constraint(A_eq=A_eq_47, b_eq=np.zeros(constraint_idx))

#Features#################################################
##########################################################

def _dynamic_constraint_static(problem):
    # Constructing A_eq and b_eq for dynamic condition on static agents as sp.coo matrix
    A_stat_row  = []
    A_stat_col  = []
    A_stat_data = []
    b_stat = []

    constraint_idx = 0
    #Enforce static agents to be static
    for t, r, v in product(range(problem.T+1), problem.static_agents,
                           problem.graph.nodes):
        A_stat_row.append(constraint_idx)
        A_stat_col.append(problem.get_z_idx(r, v, t))
        A_stat_data.append(1)
        b_stat.append(1 if problem.graph.agents[r] == v else 0)
        constraint_idx += 1

    if problem.final_position:
        #Enforce final position for agents
        for r, v in problem.final_position.items():
            A_stat_row.append(constraint_idx)
            A_stat_col.append(problem.get_z_idx(r, v, problem.T))
            A_stat_data.append(1)
            b_stat.append(1)
            constraint_idx += 1
    A_stat = sp.coo_matrix((A_stat_data, (A_stat_row, A_stat_col)), shape=(constraint_idx, problem.num_vars))#.toarray(
    return Constraint(A_eq=A_stat, b_eq=b_stat)

def _dynamic_constraint_agent_avoidance(problem):
    # Constructing A_iq and b_iq for agent avoidance as sp.coo matrix
    A_row  = []
    A_col  = []
    A_data = []

    constraint_idx = 0

    # Constraint (56)
    for t, v, (r1, r2) in product(range(problem.T+1) , problem.graph.nodes , combinations(problem.big_agents, 2)):

        if problem.graph.nodes[v]['small'] and problem.graph.nodes[v]['frontiers']==0:

            if not (problem.graph.agents[r1] == problem.graph.agents[r2] and problem.graph.agents[r1] == v):

                A_row.append(constraint_idx)
                A_col.append(problem.get_z_idx(r1, v, t))
                A_data.append(1)

                A_row.append(constraint_idx)
                A_col.append(problem.get_z_idx(r2, v, t))
                A_data.append(1)

                constraint_idx += 1

    # Constraint (57)
    for t, (v1, v2), (r1, r2) in product(range(problem.T) , problem.graph.tran_edges() , combinations(problem.big_agents, 2)):

        if (problem.graph.nodes[v1]['small'] and problem.graph.nodes[v1]['frontiers']==0) or (problem.graph.nodes[v2]['small'] and problem.graph.nodes[v2]['frontiers']==0):

            A_row.append(constraint_idx)
            A_col.append(problem.get_xf_idx(r1, v1, v2, t))
            A_data.append(1)

            A_row.append(constraint_idx)
            A_col.append(problem.get_xf_idx(r2, v2, v1, t))
            A_data.append(1)

            constraint_idx += 1

            A_row.append(constraint_idx)
            A_col.append(problem.get_xf_idx(r1, v2, v1, t))
            A_data.append(1)

            A_row.append(constraint_idx)
            A_col.append(problem.get_xf_idx(r2, v1, v2, t))
            A_data.append(1)

            constraint_idx += 1

    A = sp.coo_matrix((A_data, (A_row, A_col)), shape=(constraint_idx, problem.num_vars))#.toarray(
    return Constraint(A_iq=A, b_iq=np.ones(constraint_idx))
