import time
from dataclasses import dataclass

import numpy as np
import networkx as nx
import scipy.sparse as sp

from graph_connectivity.optimization_wrappers import solve_ilp, Constraint
from graph_connectivity.graph import Graph

from itertools import chain, combinations, product
from networkx.drawing.nx_agraph import to_agraph

@dataclass
class Variable(object):
    start: int
    size: int
    binary: bool = False

#Dynamic constraints------------------------------------------------------------

def dynamic_constraint_44(problem):
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

def dynamic_constraint_45(problem):
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

def dynamic_constraint_46(problem):
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

def dynamic_constraint_48(problem):
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

def dynamic_constraint_49(problem):
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

def dynamic_constraint_50(problem):
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

def dynamic_constraint_51(problem):
    # Constructing A_eq and b_eq for equality (51) as sp.coo matrix
    A_iq_row  = []
    A_iq_col  = []
    A_iq_data = []

    N = len(problem.graph.agents)

    constraint_idx = 0
    for t, v in product(range(problem.T+1), problem.graph.nodes):
        A_iq_row.append(constraint_idx)
        A_iq_col.append(problem.get_y_idx(v))
        A_iq_data.append(1)
        for r in range(len(problem.graph.agents)):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(problem.get_z_idx(r, v, t))
            A_iq_data.append(-1)
        constraint_idx += 1

    A_iq_51 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))

    return Constraint(A_iq=A_iq_51, b_iq=np.zeros(constraint_idx))

def dynamic_constraint_static(problem):
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

def dynamic_constraint_ex(problem):
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

def generate_dynamic_contraints(problem):
    '''constraints on z, e, y'''

    #Define number of variables
    if problem.num_vars == None: problem.compute_num_var()

    #Setup constraints
    c_44 = dynamic_constraint_44(problem)
    c_45 = dynamic_constraint_45(problem)
    c_46 = dynamic_constraint_46(problem)
    c_51 = dynamic_constraint_51(problem)           # helper variable for objective
    c_static = dynamic_constraint_static(problem)
    c_ex = dynamic_constraint_ex(problem)

    return c_44 & c_45 & c_46 & c_51 & c_static & c_ex

#Flow constraints---------------------------------------------------------------

def generate_flow_constraint_52(problem):
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

def generate_flow_contraints(problem):
    '''constraints on z, e, f, fbar'''

    c_48 = dynamic_constraint_48(problem)
    c_49 = dynamic_constraint_49(problem)
    c_50 = dynamic_constraint_50(problem)
    c_52 = generate_flow_constraint_52(problem)

    return c_48 & c_49 & c_50 & c_52

#Initial constraints------------------------------------------------------------

def generate_initial_contraints(problem):
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
    A_init = sp.coo_matrix((A_init_data, (A_init_row, A_init_col)), shape=(constraint_idx, problem.num_vars))#.toarray(

    return Constraint(A_eq=A_init, b_eq=b_init)

#Connectivity problem-----------------------------------------------------------

class ConnectivityProblem(object):

    def __init__(self):
        # Problem definition
        self.graph = None                    #Graph
        self.T = None                        #Time horizon
        self.b = None                        #bases (agent positions)
        self.static_agents = None

        # ILP setup
        self.dict_tran = None
        self.dict_conn = None
        self.vars = None
        self.constraint = Constraint()

        # ILP solution
        self.solution = None

    ##INDEX HELPER FUNCTIONS##

    @property
    def num_b(self):
        return len(self.b)

    @property
    def num_r(self):
        return len(self.graph.agents)

    @property
    def num_v(self):
        return self.graph.number_of_nodes()

    @property
    def num_vars(self):
        return sum(var.size for var in self.vars.values())

    def generate_dicts(self):
        # Create dictionaries for (i,j)->k mapping for edges
        self.dict_tran = {(i,j): k for k, (i,j) in enumerate(self.graph.tran_edges())}
        self.dict_conn = {(i,j): k for k, (i,j) in enumerate(self.graph.conn_edges())}

    def get_z_idx(self, r, v, t):
        return self.vars['z'].start + np.ravel_multi_index((t,v,r), (self.T+1, self.num_v, self.num_r))

    def get_e_idx(self, i, j, t):
        k = self.dict_tran[(i, j)]
        idx = np.ravel_multi_index((t,k), (self.T, len(self.dict_tran)))
        return self.vars['e'].start + idx

    def get_y_idx(self, v):
        return self.vars['y'].start + v

    def get_f_idx(self, b, i, j, t):
        k = self.dict_tran[(i, j)]
        idx = np.ravel_multi_index((t,b,k), (self.T, self.num_b, len(self.dict_tran)))
        return self.vars['f'].start + idx

    def get_fbar_idx(self, b, i, j, t):
        k = self.dict_conn[(i, j)]       
        idx = np.ravel_multi_index((t,b,k), (self.T+1, self.num_b, len(self.dict_conn)))
        return self.vars['fbar'].start + idx

    ##GRAPH HELPER FUNCTIONS##

    def get_time_augmented_id(self, n, t):
        return np.ravel_multi_index((t,n), (self.T+1, self.num_v))

    def get_time_augmented_n_t(self, id):
        t, n = np.unravel_index(id, (self.T+1, self.num_v))
        return n,t

    ##SOLUTION HELPER FUNCTIONS##

    def plot_solution(self):
        if self.solution is None:
            print("No solution found: call solve() to obtain solution")
        else:
            g = self.graph
            #create time-augmented graph G
            G = Graph()
            #space between timesteps
            space = 2

            #loop of graph and add nodes to time-augmented graph
            for t in range(self.T+1):
                for n in g.nodes:
                    id = self.get_time_augmented_id(n, t)
                    G.add_node(id)
                    G.nodes[id]['x'] = g.nodes[n]['x'] + space*t
                    G.nodes[id]['y'] = g.nodes[n]['y']
                    G.nodes[id]['agents'] = []

            #loop of graph and add edges to time-augmented graph
            i = 0
            for n in g.nodes:
                for t in range(self.T+1):
                    for edge in g.out_edges(n, data = True):
                        if edge[2]['type'] == 'transition' and t < self.T:
                            out_id = self.get_time_augmented_id(n, t)
                            in_id = self.get_time_augmented_id(edge[1], t+1)
                            G.add_edge(out_id , in_id, type='transition')
                        elif edge[2]['type'] == 'connectivity':
                            out_id = self.get_time_augmented_id(n, t)
                            in_id = self.get_time_augmented_id(edge[1], t)
                            G.add_edge(out_id , in_id, type='connectivity')


            #set 'pos' attribute for accurate position in plot (use ! in end for accurate positioning)
            for n in G:
                G.nodes[n]['pos'] =  ",".join(map(str,[G.nodes[n]['x'],G.nodes[n]['y']])) + '!'

            #Set general edge-attributes
            G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}

            #Set individual attributes based on solution
            sol = self.solution
            z = sol['x'][self.vars['z'].start : self.vars['z'].start + self.vars['z'].size]

            for r in self.graph.agents:
                for v in self.graph.nodes:
                    for t in range(self.T+1):
                        if z[self.get_z_idx(r, v, t)] != 0:
                            id = self.get_time_augmented_id(v, t)
                            G.nodes[id]['agents'].append(r)

            for v in G.nodes:
                G.nodes[v]['number_of_agents'] = len(G.nodes[v]['agents'])
                G.nodes[v]['label'] = " ".join(map(str,(G.nodes[v]['agents'])))


            for n in G:
                if G.nodes[n]['number_of_agents']!=0:
                    G.nodes[n]['color'] = 'black'
                    G.nodes[n]['fillcolor'] = 'red'
                    G.nodes[n]['style'] = 'filled'
                else:
                    G.nodes[n]['color'] = 'black'
                    G.nodes[n]['fillcolor'] = 'white'
                    G.nodes[n]['style'] = 'filled'
                for nbr in G[n]:
                    for edge in G[n][nbr]:
                        if G[n][nbr][edge]['type']=='connectivity':
                            if len(G.nodes[n]['agents']) != 0 and len(G.nodes[nbr]['agents'])!= 0:
                                G[n][nbr][edge]['color']='black'
                                G[n][nbr][edge]['style']='dashed'
                            else:
                                G[n][nbr][edge]['color']='grey'
                                G[n][nbr][edge]['style']='dashed'
                        else:
                            G[n][nbr][edge]['color']='grey'
                            G[n][nbr][edge]['style']='solid'
            for n in G:
                for nbr in G[n]:
                    for edge in G[n][nbr]:
                        if G[n][nbr][edge]['type']=='transition':
                            for a in G.nodes[n]['agents']:
                                if a in G.nodes[nbr]['agents']:
                                    G[n][nbr][edge]['color']='black'
                                    G[n][nbr][edge]['style']='solid'


            #Plot/save graph
            A = to_agraph(G)
            A.layout()
            A.draw('solution.png')

    ##SOLVER FUNCTIONS##

    def solve(self, solver=None, output=False, integer=True):

        self.generate_dicts()

        zvar = Variable(size=(self.T+1) * self.num_r * self.num_v,
                        start=0,
                        binary=True)
        evar = Variable(size=self.T * len(self.dict_tran),
                        start=zvar.start + zvar.size,
                        binary=False)
        yvar = Variable(size=self.num_v,
                        start=evar.start + evar.size,
                        binary=True)
        fvar = Variable(size=self.T * self.num_b * len(self.dict_tran),
                        start=yvar.start + yvar.size,
                        binary=False)
        fbarvar = Variable(size=(self.T+1) * self.num_b * len(self.dict_conn),
                           start=fvar.start + fvar.size,
                           binary=False)

        self.vars = {'z': zvar, 'e': evar, 'y': yvar, 'f': fvar, 'fbar': fbarvar}

        print("Number of variables: {}".format(self.num_vars))

        # Initial Constraints on z
        self.constraint &= generate_initial_contraints(self)

        # Dynamic Constraints on z, e, y
        dct0 = time.time()
        self.constraint &= generate_dynamic_contraints(self)
        print("Dynamic constraints setup time {:.2f}s".format(time.time() - dct0))

        # Flow constraints on z, e, f, fbar
        cct0 = time.time()
        self.constraint &= generate_flow_contraints(self)
        print("Flow constraints setup time {:.2f}s".format(time.time() - cct0))

        print("Number of constraints: {}".format(self.constraint.A_eq.shape[0]+self.constraint.A_iq.shape[0]))

        self.solve_(solver=None, output=False, integer=True)

    def solve_(self, solver=None, output=False, integer=True):
        obj = np.zeros(self.num_vars)

        # Which variables are binary/integer
        J_int = sum([list(range(var.start, var.start + var.size)) 
                    for var in self.vars.values() if not var.binary], [])
        J_bin = sum([list(range(var.start, var.start + var.size))
                    for var in self.vars.values() if var.binary], [])

        # Solve it
        self.solution = solve_ilp(obj, self.constraint, J_int, J_bin, solver, output)

        if self.solution['status'] == 'infeasible':
            print("Problem infeasible")
