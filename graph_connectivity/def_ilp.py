import time
import numpy as np
import networkx as nx
import scipy.sparse as sp

from graph_connectivity.optimization_wrappers import *
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
        self.add_path(transition_list, type='transition', self_loop = False)
        self.add_path(transition_list[::-1], type='transition', self_loop = False)
        for n in self:
            self.add_edge( n,n , type='transition', self_loop = True)

    def add_connectivity_path(self, connectivity_list):
        self.add_path(connectivity_list, type='connectivity', self_loop = False)
        self.add_path(connectivity_list[::-1] , type='connectivity', self_loop = False)
        for n in self:
            self.add_edge( n,n, type='connectivity', self_loop = True)

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

    def get_pre_S_transition(self, S_v_t):
        pre_S = set()
        for v, t in S_v_t:
            if t > 0:
                for edge in self.in_edges(v, data = True):
                    if edge[2]['type'] == 'transition' and (edge[0], t - 1) not in S_v_t:
                        pre_S.add((edge[0], edge[1], t - 1))
        return pre_S

    def get_pre_S_connectivity(self, S_v_t):
        pre_S = set()
        for v, t in S_v_t:
            for edge in self.in_edges(v, data = True):
                if edge[2]['type'] == 'connectivity' and (edge[0], t) not in S_v_t:
                    pre_S.add((edge[0], edge[1], t))
        return pre_S

#Dynamic constraints------------------------------------------------------------

class DynamicConstraints(object):

    def __init__(self, problem):
        self.A_eq = None
        self.b_eq = None
        self.A_iq = None
        self.b_iq = None
        self.transition_adj = None
        self.connectivity_adj = None
        self.problem = problem
        self.generate_dynamic_contraints()

        # variables: z^b_rvt, e_ijt, y^b_vt, x^b_ijt, xbar^b_ijt

    def dynamic_constraint_27(self):

        A_eq_row  = []
        A_eq_col  = []
        A_eq_data = []

        constraint_idx = 0
        for t, v in product(range(self.problem.T), self.problem.graph.nodes):
            #left side of (27)
            for r in self.problem.graph.agents:
                A_eq_row.append(constraint_idx)
                A_eq_col.append(self.problem.get_z_idx(r, v, t+1))
                A_eq_data.append(1)
            #right side of (27)
            for edge in self.problem.graph.in_edges(v, data = True):
                if edge[2]['type'] == 'transition':
                    A_eq_row.append(constraint_idx)
                    A_eq_col.append(self.problem.get_e_idx(edge[0], edge[1], t))
                    A_eq_data.append(-1)
            constraint_idx += 1
        A_eq_27 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, self.problem.num_vars))
        b_eq_27 = np.zeros(constraint_idx)

        return A_eq_27, b_eq_27

    def dynamic_constraint_28(self):

        A_eq_row  = []
        A_eq_col  = []
        A_eq_data = []

        constraint_idx = 0
        for t, v in product(range(self.problem.T), self.problem.graph.nodes):
            #left side of (28)
            for r in self.problem.graph.agents:
                A_eq_row.append(constraint_idx)
                A_eq_col.append(self.problem.get_z_idx(r, v, t))
                A_eq_data.append(1)
            #right side of (28)
            for edge in self.problem.graph.out_edges(v, data = True):
                if edge[2]['type'] == 'transition':
                    A_eq_row.append(constraint_idx)
                    A_eq_col.append(self.problem.get_e_idx(edge[0], edge[1], t))
                    A_eq_data.append(-1)
            constraint_idx += 1
        A_eq_28 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, self.problem.num_vars))
        b_eq_28 = np.zeros(constraint_idx)

        return A_eq_28, b_eq_28

    def dynamic_constraint_30(self):
        A_iq_row  = []
        A_iq_col  = []
        A_iq_data = []

        constraint_idx = 0
        for t, b, v1, v2 in product(range(self.problem.T), self.problem.b,
                                    self.problem.graph.nodes, self.problem.graph.nodes):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(self.problem.get_x_idx(b, v1, v2, t))
            A_iq_data.append(1)

            A_iq_row.append(constraint_idx)
            A_iq_col.append(self.problem.get_e_idx(v1, v2, t))
            A_iq_data.append(-1)
            constraint_idx += 1
        A_iq_30 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, self.problem.num_vars))
        b_iq_30 = np.zeros(constraint_idx)

        return A_iq_30, b_iq_30

    def dynamic_constraint_31(self):
        # Constructing A_eq and b_eq for equality (31) as sp.coo matrix
        A_eq_row  = []
        A_eq_col  = []
        A_eq_data = []

        constraint_idx = 0
        for t, v1, v2, b in product(range(self.problem.T+1), self.problem.graph.nodes,
                                    self.problem.graph.nodes, self.problem.b):
            if self.connectivity_adj[v1][v2] == 0:
                A_eq_row.append(constraint_idx)
                A_eq_col.append(self.problem.get_xbar_idx(b, v1, v2, t))
                A_eq_data.append(1)
                constraint_idx += 1
        A_eq_31 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, self.problem.num_vars))
        b_eq_31 = np.zeros(constraint_idx)

        return A_eq_31, b_eq_31

    def dynamic_constraint_32(self):
        # Constructing A_eq and b_eq for equality (32) as sp.coo matrix (change x -> e in text)
        A_eq_row  = []
        A_eq_col  = []
        A_eq_data = []

        constraint_idx = 0
        for t, v1, v2 in product(range(self.problem.T), self.problem.graph.nodes, self.problem.graph.nodes):
            if self.transition_adj[v1][v2] == 0:
                A_eq_row.append(constraint_idx)
                A_eq_col.append(self.problem.get_e_idx(v1, v2, t))
                A_eq_data.append(1)
                constraint_idx += 1
        A_eq_32 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, self.problem.num_vars))
        b_eq_32 = np.zeros(constraint_idx)

        return A_eq_32, b_eq_32

    def dynamic_constraint_33(self):
        # Constructing A_eq and b_eq for equality (33) as sp.coo matrix
        A_iq_row  = []
        A_iq_col  = []
        A_iq_data = []

        constraint_idx = 0
        for t, v1, v2, b in product(range(self.problem.T+1), self.problem.graph.nodes,
                                    self.problem.graph.nodes, self.problem.b):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(self.problem.get_xbar_idx(b, v1, v2, t))
            A_iq_data.append(1)
            for r in range(len(self.problem.graph.agents)):
                A_iq_row.append(constraint_idx)
                A_iq_col.append(self.problem.get_z_idx(r, v1, t))
                A_iq_data.append(-1)
            constraint_idx += 1

        A_iq_33 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, self.problem.num_vars))
        b_iq_33 = np.zeros(constraint_idx)

        return A_iq_33, b_iq_33

    def dynamic_constraint_34(self):
        # Constructing A_eq and b_eq for equality (34) as sp.coo matrix
        A_iq_row  = []
        A_iq_col  = []
        A_iq_data = []

        constraint_idx = 0
        for t, v1, v2, b in product(range(self.problem.T+1), self.problem.graph.nodes,
                                    self.problem.graph.nodes, self.problem.b):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(self.problem.get_xbar_idx(b, v1, v2, t))
            A_iq_data.append(1)
            for r in range(len(self.problem.graph.agents)):
                A_iq_row.append(constraint_idx)
                A_iq_col.append(self.problem.get_z_idx(r, v2, t))
                A_iq_data.append(-1)
            constraint_idx += 1

        A_iq_34 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, self.problem.num_vars))
        b_iq_34 = np.zeros(constraint_idx)

        return A_iq_34, b_iq_34

    def dynamic_constraint_35(self):
        # Constructing A_eq and b_eq for equality (35) as sp.coo matrix
        A_iq_row  = []
        A_iq_col  = []
        A_iq_data = []


        constraint_idx = 0
        for t, v, b in product(range(self.problem.T+1), self.problem.graph.nodes, self.problem.b):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(self.problem.get_y_idx(b, v, t))
            A_iq_data.append(1)
            for r in range(len(self.problem.graph.agents)):
                A_iq_row.append(constraint_idx)
                A_iq_col.append(self.problem.get_z_idx(r, v, t))
                A_iq_data.append(-1)
            constraint_idx += 1

        A_iq_35 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, self.problem.num_vars))
        b_iq_35 = np.zeros(constraint_idx)

        return A_iq_35, b_iq_35

    def dynamic_constraint_36(self):
        # Constructing A_iq and b_iq for equality (36) as sp.coo matrix
        A_iq_row  = []
        A_iq_col  = []
        A_iq_data = []

        N = len(self.problem.graph.agents)

        constraint_idx = 0
        for v, b in product(self.problem.graph.nodes, self.problem.b):
            for r in self.problem.graph.agents:
                A_iq_row.append(constraint_idx)
                A_iq_col.append(self.problem.get_z_idx(r, v, self.problem.T))
                A_iq_data.append(1)
            A_iq_row.append(constraint_idx)
            A_iq_col.append(self.problem.get_y_idx(b, v, self.problem.T))
            A_iq_data.append(-N)
            constraint_idx += 1

        A_iq_36 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, self.problem.num_vars))
        b_iq_36 = np.zeros(constraint_idx)

        return A_iq_36, b_iq_36

    def dynamic_constraint_id(self):
        # Constructing A_iq and b_iq for identity dynamics as sp.coo matrix
        A_iq_row  = []
        A_iq_col  = []
        A_iq_data = []

        constraint_idx = 0
        for t, v, r in product(range(self.problem.T), self.problem.graph.nodes, self.problem.graph.agents):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(self.problem.get_z_idx(r, v, t+1))
            A_iq_data.append(1)
            for edge in self.problem.graph.in_edges(v, data = True):
                if edge[2]['type'] == 'transition':
                    A_iq_row.append(constraint_idx)
                    A_iq_col.append(self.problem.get_z_idx(r, edge[0], t))
                    A_iq_data.append(-1)
            constraint_idx += 1
        A_iq_id = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, self.problem.num_vars))
        b_iq_id = np.zeros(constraint_idx)

        return A_iq_id, b_iq_id

    def dynamic_constraint_ex(self):
        # Constructing A_eq and b_eq for equality for agent existance as sp.coo matrix
        A_eq_row  = []
        A_eq_col  = []
        A_eq_data = []

        constraint_idx = 0
        for t, r in product(range(self.problem.T+1), self.problem.graph.agents):
            for v in self.problem.graph.nodes:
                A_eq_row.append(constraint_idx)
                A_eq_col.append(self.problem.get_z_idx(r, v, t))
                A_eq_data.append(1)
            constraint_idx += 1
        A_eq_ex = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, self.problem.num_vars))
        b_eq_ex = np.ones(constraint_idx)

        return A_eq_ex, b_eq_ex

    def dynamic_constraint_ubz(self):
        # Constructing A_iq and b_iq for inequality as sp.coo matrix to bound z
        A_iq_ubz = sp.coo_matrix((np.ones(self.problem.num_z), (range(self.problem.num_z), range(self.problem.num_z))), shape=(self.problem.num_z, self.problem.num_vars))
        b_iq_ubz = np.ones(self.problem.num_z)

        return A_iq_ubz, b_iq_ubz

    def dynamic_constraint_uby(self):
        # Constructing A_iq and b_iq for inequality as sp.coo matrix to bound y
        A_iq_row  = []
        A_iq_col  = []
        A_iq_data = []

        constraint_idx = 0
        for t,b,v in product(range(self.problem.T+1), self.problem.b, self.problem.graph.nodes):
            A_iq_row.append(constraint_idx)
            A_iq_col.append(self.problem.get_y_idx(b, v, t))
            A_iq_data.append(1)
            constraint_idx += 1
        A_iq_uby = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, self.problem.num_vars))
        b_iq_uby = np.ones(constraint_idx)

        return A_iq_uby, b_iq_uby

    def dynamic_constraint_static(self):
        # Constructing A_eq and b_eq for dynamic condition on static agents as sp.coo matrix
        A_stat_row  = []
        A_stat_col  = []
        A_stat_data = []
        b_stat = []

        constraint_idx = 0
        for t, r, v in product(range(self.problem.T+1), self.problem.static_agents,
                               self.problem.graph.nodes):
            A_stat_row.append(constraint_idx)
            A_stat_col.append(self.problem.get_z_idx(r, v, t))
            A_stat_data.append(1)
            if self.problem.graph.agents[r] == v: b_stat.append(1)
            else: b_stat.append(0)
            constraint_idx += 1
        A_stat = sp.coo_matrix((A_stat_data, (A_stat_row, A_stat_col)), shape=(constraint_idx, self.problem.num_vars))#.toarray(

        return A_stat, b_stat

    def generate_dynamic_contraints(self):

        #Define number of variables
        if self.problem.num_vars == None: self.problem.compute_num_var()

        # Obtain adjacency matrix for transition/connectivity edges separately
        self.transition_adj = self.problem.graph.transition_adjacency_matrix()
        self.connectivity_adj = self.problem.graph.connectivity_adjacency_matrix()

        #Setup constraints
        A_eq_27, b_eq_27 = self.dynamic_constraint_27()
        A_eq_28, b_eq_28 = self.dynamic_constraint_28()
        A_iq_30, b_iq_30 = self.dynamic_constraint_30()
        A_eq_31, b_eq_31 = self.dynamic_constraint_31()
        A_eq_32, b_eq_32 = self.dynamic_constraint_32()
        A_iq_33, b_iq_33 = self.dynamic_constraint_33()
        A_iq_34, b_iq_34 = self.dynamic_constraint_34()
        A_iq_35, b_iq_35 = self.dynamic_constraint_35()
        A_iq_36, b_iq_36 = self.dynamic_constraint_36()
        A_iq_id, b_iq_id = self.dynamic_constraint_id()
        A_eq_ex, b_eq_ex = self.dynamic_constraint_ex()
        A_iq_ubz, b_iq_ubz = self.dynamic_constraint_ubz()
        A_iq_uby, b_iq_uby = self.dynamic_constraint_uby()
        A_stat, b_stat = self.dynamic_constraint_static()

        #Stack all equality constraints
        self.A_eq = sp.bmat([[A_eq_27], [A_eq_28], [A_eq_31], [A_eq_32], [A_eq_ex], [A_stat]])
        self.b_eq = np.hstack([b_eq_27, b_eq_28, b_eq_31, b_eq_32, b_eq_ex, b_stat])

        #Stack all inequality constraints
        self.A_iq = sp.bmat([[A_iq_30], [A_iq_33], [A_iq_34], [A_iq_35], [A_iq_36], [A_iq_ubz], [A_iq_uby], [A_iq_id]])
        self.b_iq = np.hstack([b_iq_30, b_iq_33, b_iq_34, b_iq_35, b_iq_36, b_iq_ubz, b_iq_uby, b_iq_id])

#Connectivity constraints-------------------------------------------------------

class ConnectivityConstraint(object):

    def __init__(self, problem):
        self.A_eq = None
        self.b_eq = None
        self.A_iq = None
        self.b_iq = None
        self.problem = problem
        self.generate_connectivity_contraints()

        # variables: z^b_rvt, e_ijt, y^b_vt, x^b_ijt, xbar^b_ijt

    def connectivity_constraint(self):
        # Constructing A_iq and b_iq for inequality (37) as sp.coo matrix
        A_iq_row  = []
        A_iq_col  = []
        A_iq_data = []

        constraint_idx = 0
        #For each base
        for b in self.problem.b:
            #Find all sets S exclude node where b is
            for S in self.problem.powerset_exclude_vertex(b):
                #Represent set S as list of tuples (vertex, time)
                S_v_t = [self.problem.get_time_augmented_n_t(id) for id in S]

                pre_S_transition = self.problem.graph.get_pre_S_transition(S_v_t)
                pre_S_connectivity = self.problem.graph.get_pre_S_connectivity(S_v_t)

                #For each v in S
                for v, t in S_v_t:
                    #add y
                    A_iq_row.append(constraint_idx)
                    A_iq_col.append(self.problem.get_y_idx(b, v, t))
                    A_iq_data.append(1)
                    for v0, v1, t0 in pre_S_transition:
                        A_iq_row.append(constraint_idx)
                        A_iq_col.append(self.problem.get_x_idx(b, v0, v1, t0))
                        A_iq_data.append(-1)
                    for v0, v1, t1 in pre_S_connectivity:
                        A_iq_row.append(constraint_idx)
                        A_iq_col.append(self.problem.get_xbar_idx(b, v0, v1, t1))
                        A_iq_data.append(-1)
                    constraint_idx += 1
        A_iq_37 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), 
                                shape=(constraint_idx, self.problem.num_vars))
        b_iq_37 = np.zeros(constraint_idx)
        A_eq_37 = sp.coo_matrix((0, self.problem.num_vars))  # zero matrix
        b_eq_37 = []

        return A_eq_37, b_eq_37, A_iq_37, b_iq_37

    def generate_connectivity_contraints(self):

        if self.problem.num_vars == None: self.problem.compute_num_var()

        #Setup constraints
        self.A_eq, self.b_eq, self.A_iq, self.b_iq = self.connectivity_constraint()

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

        self.equality_constraints = None       #equality contraints
        self.inequality_constraints = None     #inequality contraints

        self.solution = None

        # variables: z^b_rvt, e_ijt, y^b_vt, x^b_ijt, xbar^b_ijt

##INDEX HELPER FUNCTIONS##

    def compute_num_var(self):
        self.num_b = len(self.b)
        self.num_r = len(self.graph.agents)
        self.num_v = self.graph.number_of_nodes()

        self.num_z = (self.T+1) * self.num_r * self.num_v
        self.num_e = self.T * self.num_v * self.num_v
        self.num_y = (self.T+1) * self.num_b * self.num_v
        self.num_x = self.T * self.num_b * self.num_v * self.num_v
        self.num_xbar = (self.T+1) * self.num_b * self.num_v * self.num_v

        self.num_vars = self.num_z + self.num_e + self.num_y + self.num_x + self.num_xbar
        print("Number of variables: {}".format(self.num_vars))

    def get_z_idx(self, r, v, t):
        return np.ravel_multi_index((t,v,r), (self.T+1, self.num_v, self.num_r))

    def get_e_idx(self, i, j, t):
        start = self.num_z
        idx = np.ravel_multi_index((t,i,j), (self.T, self.num_v, self.num_v))
        return start + idx

    def get_y_idx(self, b, v, t):
        start = self.num_z + self.num_e
        idx = np.ravel_multi_index((t,b,v), (self.T+1, self.num_b, self.num_v))
        return start + idx

    def get_x_idx(self, b, i, j, t):
        start = self.num_z + self.num_e + self.num_y
        idx = np.ravel_multi_index((t,b,i,j), (self.T, self.num_b, self.num_v, self.num_v))
        return start + idx

    def get_xbar_idx(self, b, i, j, t):
        start = self.num_z + self.num_e + self.num_y + self.num_x
        idx = np.ravel_multi_index((t,b,i,j), (self.T+1, self.num_b, self.num_v, self.num_v))
        return start + idx

##GRAPH HELPER FUNCTIONS##

    def get_time_augmented_id(self, n, t):
        return np.ravel_multi_index((t,n), (self.T+1, self.num_v))

    def get_time_augmented_n_t(self, id):
        t, n = np.unravel_index(id, (self.T+1, self.num_v))
        return n,t

    def powerset_exclude_vertex(self, b):
        time_augmented_nodes = []
        for t in range(self.T+1):
            for v in self.graph.nodes:
                time_augmented_nodes.append(self.get_time_augmented_id(v,t))
        s = list(time_augmented_nodes)
        s.remove(self.get_time_augmented_id(self.b[b],0))
        return(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))

##CONSTRAINTS HELPER FUNCTIONS##

    def add_constraints(self, constraints):
        for constraint in constraints:
            if self.equality_constraints is not None:
                self.equality_constraints[0] = sp.bmat([[self.equality_constraints[0]], [constraint.A_eq]])
                self.equality_constraints[1] = np.hstack([self.equality_constraints[1], constraint.b_eq])

            else:
                self.equality_constraints = [constraint.A_eq, constraint.b_eq]

            if self.inequality_constraints is not None:
                self.inequality_constraints[0] = sp.bmat([[self.inequality_constraints[0]], [constraint.A_iq]])
                self.inequality_constraints[1] = np.hstack([self.inequality_constraints[1], constraint.b_iq])

            else:
                self.inequality_constraints = [constraint.A_iq, constraint.b_iq]
        print("Number of constraints: {}".format(len(self.equality_constraints[1])+len(self.inequality_constraints[1])))

    def add_connectivity_constraint(self, add_S):

        # Constructing A_iq and b_iq for inequality (37) for all S in add_S as sp.coo matrix
        A_iq_row  = []
        A_iq_col  = []
        A_iq_data = []

        constraint_idx = 0
        #For each base
        for b in self.b:
            #For each S to add
            for S_v_t in add_S:

                pre_S_transition = self.graph.get_pre_S_transition(S_v_t)
                pre_S_connectivity = self.graph.get_pre_S_connectivity(S_v_t)

                #For each v in S
                for v, t in S_v_t:
                    #add y
                    A_iq_row.append(constraint_idx)
                    A_iq_col.append(self.get_y_idx(b, v, t))
                    A_iq_data.append(1)
                    for v0, v1, t0 in pre_S_transition:
                        A_iq_row.append(constraint_idx)
                        A_iq_col.append(self.get_x_idx(b, v0, v1, t0))
                        A_iq_data.append(-1)
                    for v0, v1, t1 in pre_S_connectivity:
                        A_iq_row.append(constraint_idx)
                        A_iq_col.append(self.get_xbar_idx(b, v0, v1, t1))
                        A_iq_data.append(-1)
                    constraint_idx += 1
        A_iq_37 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, self.num_vars))
        b_iq_37 = np.zeros(constraint_idx)

        A_iq = A_iq_37
        b_iq = b_iq_37

        if self.inequality_constraints is not None:
            self.inequality_constraints[0] = sp.bmat([[self.inequality_constraints[0]], [A_iq]])
            self.inequality_constraints[1] = np.hstack([self.inequality_constraints[1], b_iq])

        else:
            self.inequality_constraints = [A_iq, b_iq]

        print("Number of constraints: {}".format(len(self.equality_constraints[1])+len(self.inequality_constraints[1])))

##PROBLEM SETUP HELPER FUNCTIONS##

    def compile(self):

        #Set up Dynamic Constraints
        dct0 = time.time()
        dc = DynamicConstraints(self)
        print("Dynamic constraints setup time", time.time() - dct0)

        #Set up Connectivity contraint
        cct0 = time.time()
        cc = ConnectivityConstraint(self)
        print("Connectivity constraints setup time", time.time() - cct0)

        self.add_constraints([dc, cc])

        # Constructing A_eq and b_eq for initial condition as sp.coo matrix
        A_init_row  = []
        A_init_col  = []
        A_init_data = []
        b_init = []

        constraint_idx = 0
        for r in self.graph.agents:
            for v in self.graph.nodes:
                A_init_row.append(constraint_idx)
                A_init_col.append(self.get_z_idx(r, v, 0))
                A_init_data.append(1)
                if self.graph.agents[r] == v: b_init.append(1)
                else: b_init.append(0)
                constraint_idx += 1
        A_init = sp.coo_matrix((A_init_data, (A_init_row, A_init_col)), shape=(constraint_idx, self.num_vars))#.toarray(

        if self.equality_constraints is not None:
            self.equality_constraints[0] = sp.bmat([[self.equality_constraints[0]], [A_init]])
            self.equality_constraints[1] = np.hstack([self.equality_constraints[1], b_init])

        else:
            self.equality_constraints = A_init
            self.equality_constraints = b_init

    def compile_adaptive(self):

        #Set up Dynamic Constraints
        dct0 = time.time()
        dc = DynamicConstraints(self)
        print("Dynamic constraints setup time", time.time() - dct0)

        self.add_constraints([dc])

        # Constructing A_eq and b_eq for initial condition as sp.coo matrix
        A_init_row  = []
        A_init_col  = []
        A_init_data = []
        b_init = []

        constraint_idx = 0
        for r in self.graph.agents:
            for v in self.graph.nodes:
                A_init_row.append(constraint_idx)
                A_init_col.append(self.get_z_idx(r, v, 0))
                A_init_data.append(1)
                if self.graph.agents[r] == v: b_init.append(1)
                else: b_init.append(0)
                constraint_idx += 1
        A_init = sp.coo_matrix((A_init_data, (A_init_row, A_init_col)), shape=(constraint_idx, self.num_vars))#.toarray(

        if self.equality_constraints is not None:
            self.equality_constraints[0] = sp.bmat([[self.equality_constraints[0]], [A_init]])
            self.equality_constraints[1] = np.hstack([self.equality_constraints[1], b_init])

        else:
            self.equality_constraints = A_init
            self.equality_constraints = b_init

    def check_well_defined(self):
        # Check input data
        if self.T is None:
            raise Exception("No time horizon 'T' specified")

        if self.graph is None:
            raise Exception("No graph 'G' specified")

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

    def test_solution(self):
        solution = self.solution['x']
        add_S = []
        for r in self.graph.agents:
            S = []
            #Find end position of agent r and set as V
            for v in self.graph.nodes:
                z_idx = self.get_z_idx(r, v, self.T)
                if solution[z_idx] == 1:
                    S.append((v, self.T))

            idx = 0
            while idx < len(S):
                pre_S_transition = self.graph.get_pre_S_transition([S[idx]])
                pre_S_connectivity = self.graph.get_pre_S_connectivity([S[idx]])
                for v0, v1, t in pre_S_transition:
                    occupied = False
                    for robot in self.graph.agents:
                        z_idx = self.get_z_idx(robot, v0, t)
                        if solution[z_idx] == 1:
                            occupied = True
                    if occupied == True and (v0, t) not in S:
                        S.append((v0, t))
                for v0, v1, t in pre_S_connectivity:
                    occupied = False
                    for robot in self.graph.agents:
                        z_idx = self.get_z_idx(robot, v0, t)
                        if solution[z_idx] == 1:
                            occupied = True
                    if occupied == True and (v0, t) not in S:
                        S.append((v0, t))
                idx +=1


            valid = True
            for b in self.b:
                if (self.b[b],0) not in S:
                    valid = False
            if valid == False:
                for b in self.b:
                    if (self.b[b]) in S:
                        s.remove((self.b[b],0))
                add_S.append(S)


        if len(add_S) != 0:
            return False, add_S
        else:
            return True, add_S

##SOLVER FUNCTIONS##

    def solve(self, solver=None, output=False, integer=True):

        # variables: z^b_rvt, e_ijt, y^b_vt, x^b_ijt, xbar^b_ijt

        #Equality Constraints
        try:
            A_eq = self.equality_constraints[0]
            b_eq = self.equality_constraints[1]
        except:
            A_eq = sp.coo_matrix(([], ([],[])), shape=(0, self.num_vars))
            b_eq = []

        #Inequality Constraints
        try:
            A_iq = self.inequality_constraints[0]
            b_iq = self.inequality_constraints[1]
        except:
            A_iq = sp.coo_matrix(([], ([],[])), shape=(0, self.num_vars))
            b_iq = []

        obj = np.zeros(self.num_vars)

        # Solve it
        if integer:
            sol = solve_ilp(obj, A_iq, b_iq, A_eq, b_eq,
                            None, solver, output);
        else:
            sol = solve_ilp(obj, A_iq, b_iq, A_eq, b_eq,
                            [], solver, output);

        self.solution = sol

        return sol['x'][0 : self.num_z], \
               sol['x'][self.num_z : self.num_z + self.num_e], \
               sol['x'][self.num_z + self.num_e : self.num_z + self.num_e + self.num_y], \

    def solve_adaptive(self, solver=None, output=False, integer=True):
        valid_solution = False
        while valid_solution == False:
            solution = self.solve(solver, output, integer)
            valid_solution, add_S = self.test_solution()
            self.add_connectivity_constraint(add_S)
        return(solution)
