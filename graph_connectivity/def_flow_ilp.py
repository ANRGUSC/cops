import time

import numpy as np
import networkx as nx
import scipy.sparse as sp

from graph_connectivity.optimization_wrappers import solve_ilp, Constraint
from networkx.drawing.nx_agraph import to_agraph
from copy import deepcopy
from itertools import chain, combinations, product

#MultiDiGraph-------------------------------------------------------------------

class Graph(nx.MultiDiGraph):

    def __init__(self):
        super(Graph, self).__init__()
        self.agents = None

    def plot_graph(self):

        #copy graph to add plot attributes
        graph_copy = deepcopy(self)

        #set 'pos' attribute for accurate position in plot (use ! in end for accurate positioning)
        for n in graph_copy:
            graph_copy.nodes[n]['pos'] =  ",".join(map(str,[graph_copy.nodes[n]['x'],graph_copy.nodes[n]['y']])) + '!'

        #Set general edge-attributes
        graph_copy.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}

        #Set individual attributes
        for n in graph_copy:
            if graph_copy.nodes[n]['number_of_agents']!=0:
                graph_copy.nodes[n]['color'] = 'black'
                graph_copy.nodes[n]['fillcolor'] = 'red'
                graph_copy.nodes[n]['style'] = 'filled'
            else:
                graph_copy.nodes[n]['color'] = 'black'
                graph_copy.nodes[n]['fillcolor'] = 'white'
                graph_copy.nodes[n]['style'] = 'filled'
            for nbr in self[n]:
                for edge in graph_copy[n][nbr]:
                    if graph_copy[n][nbr][edge]['type']=='connectivity':
                        graph_copy[n][nbr][edge]['color']='grey'
                        graph_copy[n][nbr][edge]['style']='dashed'
                    else:
                        graph_copy[n][nbr][edge]['color']='grey'
                        graph_copy[n][nbr][edge]['style']='solid'

        #Change node label to agents in node
        for n in graph_copy:
            graph_copy.nodes[n]['label'] = " ".join(map(str,(graph_copy.nodes[n]['agents'])))


        #Plot/save graph
        A = to_agraph(graph_copy)
        A.layout()
        A.draw('graph.png')

    def add_transition_path(self, transition_list):
        self.add_path(transition_list, type='transition')
        self.add_path(transition_list[::-1], type='transition')
        for n in transition_list:
            if self.has_edge(n,n) == False:
                self.add_edge( n,n , type='transition', self_loop = True)

    def add_connectivity_path(self, connectivity_list):
        self.add_path(connectivity_list, type='connectivity')
        self.add_path(connectivity_list[::-1] , type='connectivity')
        for n in connectivity_list:
            if self.has_edge(n,n) == False:
                self.add_edge( n,n, type='connectivity')

    def set_node_positions(self, position_dictionary):
        self.add_nodes_from(position_dictionary.keys())
        for n, p in position_dictionary.items():
            self.node[n]['x'] = p[0]
            self.node[n]['y'] = p[1]

    def init_agents(self, agent_position_dictionary):

        for n in self:
            self.node[n]['number_of_agents']=0

        for agent in agent_position_dictionary:
            self.node[agent_position_dictionary[agent]]['number_of_agents']+=1

        self.agents = agent_position_dictionary
        for n in self.nodes:
            self.nodes[n]['agents'] = []
        for agent in self.agents:
            self.nodes[self.agents[agent]]['agents'].append(agent)

    def transition_adjacency_matrix(self):
        num_nodes = self.number_of_nodes()
        adj = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
        for n in self:
            for edge in self.in_edges(n, data = True):
                if edge[2]['type'] == 'transition':
                    adj[edge[0]][edge[1]] = 1
        return adj

    def connectivity_adjacency_matrix(self):
        num_nodes = self.number_of_nodes()
        adj = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
        for n in self:
            for edge in self.in_edges(n, data = True):
                if edge[2]['type'] == 'connectivity':
                    adj[edge[0]][edge[1]] = 1
        return adj

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
        for edge in problem.graph.in_edges(v, data = True):
            if edge[2]['type'] == 'transition':
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
        for edge in problem.graph.out_edges(v, data = True):
            if edge[2]['type'] == 'transition':
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
        for edge in problem.graph.out_edges(v, data = True):
            if edge[2]['type'] == 'transition':
                A_iq_row.append(constraint_idx)
                A_iq_col.append(problem.get_z_idx(r, edge[1], t+1))
                A_iq_data.append(-1)
        constraint_idx += 1
    A_iq_46 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))

    return Constraint(A_iq=A_iq_46, b_iq=np.zeros(constraint_idx))

def dynamic_constraint_47(problem):
    # Constructing A_eq and b_eq for equality (47) as sp.coo matrix
    A_eq_row  = []
    A_eq_col  = []
    A_eq_data = []

    connectivity_adj = problem.graph.connectivity_adjacency_matrix()

    constraint_idx = 0
    for t, b, v1, v2 in product(range(problem.T+1), problem.b,
                                problem.graph.nodes, problem.graph.nodes):
        if connectivity_adj[v1][v2] == 0:
            A_eq_row.append(constraint_idx)
            A_eq_col.append(problem.get_fbar_idx(b, v1, v2, t))
            A_eq_data.append(1)
            constraint_idx += 1
    A_eq_47 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))

    return Constraint(A_eq=A_eq_47, b_eq=np.zeros(constraint_idx))

def dynamic_constraint_48(problem):
    # Constructing A_eq and b_eq for equality (48) as sp.coo matrix
    A_iq_row  = []
    A_iq_col  = []
    A_iq_data = []

    N = len(problem.graph.agents)

    constraint_idx = 0
    for t, b, v1, v2 in product(range(problem.T+1), problem.b,
                                problem.graph.nodes, problem.graph.nodes):
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
    for t, b, v1, v2 in product(range(problem.T+1), problem.b,
                                problem.graph.nodes, problem.graph.nodes):
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
    for t, b, v1, v2 in product(range(problem.T), problem.b,
                                problem.graph.nodes, problem.graph.nodes):
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
        if problem.graph.agents[r] == v:
            b_stat.append(1)
        else:
            b_stat.append(0)
        constraint_idx += 1
    A_stat = sp.coo_matrix((A_stat_data, (A_stat_row, A_stat_col)), shape=(constraint_idx, problem.num_vars))#.toarray(

    return Constraint(A_eq=A_stat, b_eq=b_stat)

def dynamic_constraint_ex(problem):
    # Constructing A_eq and b_eq for equality for agent existance as sp.coo matrix
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

    #Define number of variables
    if problem.num_vars == None: problem.compute_num_var()

    #Setup constraints
    c_44 = dynamic_constraint_44(problem)
    c_45 = dynamic_constraint_45(problem)
    c_46 = dynamic_constraint_46(problem)
    c_47 = dynamic_constraint_47(problem)
    c_48 = dynamic_constraint_48(problem)
    c_49 = dynamic_constraint_49(problem)
    c_50 = dynamic_constraint_50(problem)
    c_51 = dynamic_constraint_51(problem)
    c_static = dynamic_constraint_static(problem)
    c_ex = dynamic_constraint_ex(problem)

    return c_44 & c_45 & c_46 & c_47 & c_48 & c_49 & c_50 & c_51 & c_static & c_ex

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

    #Define number of variables
    if problem.num_vars == None: problem.compute_num_var()

    #Setup constraints
    c_52 = generate_flow_constraint_52(problem)

    return c_52

#Initial constraints------------------------------------------------------------

def generate_initial_contraints(problem):

    if problem.num_vars == None: problem.compute_num_var()

    # Constructing A_eq and b_eq for initial condition as sp.coo matrix
    A_init_row  = []
    A_init_col  = []
    A_init_data = []
    b_init = []

    constraint_idx = 0
    for r in problem.graph.agents:
        for v in problem.graph.nodes:
            A_init_row.append(constraint_idx)
            A_init_col.append(problem.get_z_idx(r, v, 0))
            A_init_data.append(1)
            if problem.graph.agents[r] == v: b_init.append(1)
            else: b_init.append(0)
            constraint_idx += 1
    A_init = sp.coo_matrix((A_init_data, (A_init_row, A_init_col)), shape=(constraint_idx, problem.num_vars))#.toarray(

    return Constraint(A_eq=A_init, b_eq=b_init)

#Connectivity problem-----------------------------------------------------------

class ConnectivityProblem(object):

    def __init__(self):
        self.graph = None                    #Graph
        self.T = None                        #Time horizon
        self.b = None                        #bases (agent positions)
        self.static_agents = None
        self.num_b = None
        self.num_r = None
        self.num_v = None
        self.num_z = None
        self.num_e = None
        self.num_y = None
        self.num_x = None
        self.num_xbar = None
        self.num_vars = None

        self.constraint = Constraint()

        self.solution = None

    ##INDEX HELPER FUNCTIONS##

    def compute_num_var(self):
        self.num_b = len(self.b)
        self.num_r = len(self.graph.agents)
        self.num_v = self.graph.number_of_nodes()

        self.num_z = (self.T+1) * self.num_r * self.num_v
        self.num_e = self.T * self.num_v * self.num_v
        self.num_y = self.num_v
        self.num_f = self.T * self.num_b * self.num_v * self.num_v
        self.num_fbar = (self.T+1) * self.num_b * self.num_v * self.num_v

        self.num_vars = self.num_z + self.num_e + self.num_y + self.num_f + self.num_fbar
        print("Number of variables: {}".format(self.num_vars))

    def get_z_idx(self, r, v, t):
        return np.ravel_multi_index((t,v,r), (self.T+1, self.num_v, self.num_r))

    def get_e_idx(self, i, j, t):
        start = self.num_z
        idx = np.ravel_multi_index((t,i,j), (self.T, self.num_v, self.num_v))
        return start + idx

    def get_y_idx(self, v):
        start = self.num_z + self.num_e
        idx = v
        return start + idx

    def get_f_idx(self, b, i, j, t):
        start = self.num_z + self.num_e + self.num_y
        idx = np.ravel_multi_index((t,b,i,j), (self.T, self.num_b, self.num_v, self.num_v))
        return start + idx

    def get_fbar_idx(self, b, i, j, t):
        start = self.num_z + self.num_e + self.num_y + self.num_f
        idx = np.ravel_multi_index((t,b,i,j), (self.T+1, self.num_b, self.num_v, self.num_v))
        return start + idx

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
            z = sol['x'][0 : self.num_z]

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

        #Initial Constraints
        self.constraint &= generate_initial_contraints(self)

        #Set up Dynamic Constraints
        dct0 = time.time()
        self.constraint &= generate_dynamic_contraints(self)
        print("Dynamic constraints setup time {:.2f}s".format(time.time() - dct0))

        #Set up Flow contraint
        cct0 = time.time()
        self.constraint &= generate_flow_contraints(self)
        print("Flow constraints setup time {:.2f}s".format(time.time() - cct0))

        print("Number of constraints: {}".format(self.constraint.A_eq.shape[1]+self.constraint.A_iq.shape[1]))

        self.solve_(solver=None, output=False, integer=True)

    def solve_(self, solver=None, output=False, integer=True):
        obj = np.zeros(self.num_vars)

        # Which variables are binary/integer
        start = 0
        J_bin = list(range(start, start + self.num_z))   # z binary
        start += self.num_z
        J_int = list(range(start, start + self.num_e))   # e integer
        start += self.num_e
        J_bin = list(range(start, start + self.num_y))   # y binary
        start += self.num_y
        J_int += list(range(start, start + self.num_f))  # f integer
        start += self.num_f
        J_int += list(range(start, start + self.num_fbar))  # fbar integer

        # Solve it
        self.solution = solve_ilp(obj, self.constraint, J_int, J_bin, solver, output)

        if self.solution['status'] == 'infeasible':
            print("Problem infeasible")
