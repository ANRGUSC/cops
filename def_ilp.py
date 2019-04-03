import numpy as np
import networkx as nx
import scipy.sparse as sp
from collections import deque

from optimization_wrappers import *

from networkx.drawing.nx_pydot import write_dot
from networkx.drawing.nx_agraph import to_agraph

from copy import deepcopy


#MultiDiGraph-------------------------------------------------------------------


class Graph(nx.MultiDiGraph):

    def __init__(self):
        super(Graph, self).__init__()
        self.num_nodes = None #Number of nodes
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
            #Change node label to number of agents in node
            #self.node[n]['label']=0

        for agent in agent_position_dictionary:
            self.node[agent_position_dictionary[agent]]['number_of_agents']+=1

        self.agents = agent_position_dictionary
        self.num_nodes = self.number_of_nodes()
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

class DynamicConstraints(object):
    def __init__(self, problem):
        self.A_eq = None
        self.b_eq = None
        self.A_iq = None
        self.b_iq = None
        self.problem = problem
        self.generate_dynamic_contraints(self.problem)

        # variables: z^b_rvt, e_ijt, y^b_vt, x^b_ijt, xbar^b_ijt

    def generate_dynamic_contraints(self, problem):
        """Generate equalities (27),(28)"""

        if problem.num_vars == None: problem.add_num_var()

        # Obtain adjacency matrix for transition/connectivity edges separately
        transition_adj = problem.graph.transition_adjacency_matrix()
        connectivity_adj = problem.graph.connectivity_adjacency_matrix()


        # Constructing A_eq and b_eq for equality (27) as sp.coo matrix
        A_eq_row  = np.array([])
        A_eq_col  = np.array([])
        A_eq_data = np.array([])


        constraint_idx = 0
        for t in range(problem.T):
            for v in problem.graph.nodes:
                #left side of (27)
                for r in problem.graph.agents:
                    A_eq_row = np.append(A_eq_row, constraint_idx)
                    A_eq_col = np.append(A_eq_col, problem.get_z_idx(r, v, t+1))
                    A_eq_data = np.append(A_eq_data, 1)
                #right side of (27)
                for edge in problem.graph.in_edges(v, data = True):
                    if edge[2]['type'] == 'transition':
                        A_eq_row = np.append(A_eq_row, constraint_idx)
                        A_eq_col = np.append(A_eq_col, problem.get_e_idx(edge[0], edge[1], t))
                        A_eq_data = np.append(A_eq_data, -1)
                constraint_idx += 1
        A_eq_35 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))#.toarray()
        b_eq_35 = np.zeros(constraint_idx)#.toarray()


        # Constructing A_eq and b_eq for equality (28) as sp.coo matrix
        A_eq_row  = np.array([])
        A_eq_col  = np.array([])
        A_eq_data = np.array([])

        constraint_idx = 0
        for t in range(problem.T):
            for v in problem.graph.nodes:
                #left side of (35)
                for r in problem.graph.agents:
                    A_eq_row = np.append(A_eq_row, constraint_idx)
                    A_eq_col = np.append(A_eq_col, problem.get_z_idx(r, v, t))
                    A_eq_data = np.append(A_eq_data, 1)
                #right side of (35)
                for edge in problem.graph.out_edges(v, data = True):
                    if edge[2]['type'] == 'transition':
                        A_eq_row = np.append(A_eq_row, constraint_idx)
                        A_eq_col = np.append(A_eq_col, problem.get_e_idx(edge[0], edge[1], t))
                        A_eq_data = np.append(A_eq_data, -1)
                constraint_idx += 1
        A_eq_36 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))#.toarray()
        b_eq_36 = np.zeros(constraint_idx)#.toarray()


        # Constructing A_eq and b_eq for equality for identity dynamics as sp.coo matrix
        A_iq_row  = np.array([])
        A_iq_col  = np.array([])
        A_iq_data = np.array([])

        constraint_idx = 0
        for t in range(problem.T):
            for v in problem.graph.nodes:
                for r in problem.graph.agents:
                    A_iq_row = np.append(A_iq_row, constraint_idx)
                    A_iq_col = np.append(A_iq_col, problem.get_z_idx(r, v, t+1))
                    A_iq_data = np.append(A_iq_data, 1)
                    for edge in problem.graph.in_edges(v, data = True):
                        if edge[2]['type'] == 'transition':
                            A_iq_row = np.append(A_iq_row, constraint_idx)
                            A_iq_col = np.append(A_iq_col, problem.get_z_idx(r, edge[0], t))
                            A_iq_data = np.append(A_iq_data, -1)
                    constraint_idx += 1
        A_iq_id = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))#.toarray()
        b_iq_id = np.zeros(constraint_idx)#.toarray()


        # Constructing A_eq and b_eq for equality (32) as sp.coo matrix (change x -> e in text)
        A_eq_row  = np.array([])
        A_eq_col  = np.array([])
        A_eq_data = np.array([])

        constraint_idx = 0
        for t in range(problem.T):
            for v1 in problem.graph.nodes:
                for v2 in problem.graph.nodes:
                    if transition_adj[v1][v2] == 0:
                        A_eq_row = np.append(A_eq_row, constraint_idx)
                        A_eq_col = np.append(A_eq_col, problem.get_e_idx(v1, v2, t))
                        A_eq_data = np.append(A_eq_data, 1)
                        constraint_idx += 1
        A_eq_32 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))#.toarray()
        b_eq_32 = np.zeros(constraint_idx)#.toarray()


        # Constructing A_eq and b_eq for equality for agent existance as sp.coo matrix (change x -> e in text)
        A_eq_row  = np.array([])
        A_eq_col  = np.array([])
        A_eq_data = np.array([])

        constraint_idx = 0
        for t in range(problem.T+1):
            for r in problem.graph.agents:
                for v in problem.graph.nodes:
                    A_eq_row = np.append(A_eq_row, constraint_idx)
                    A_eq_col = np.append(A_eq_col, problem.get_z_idx(r, v, t))
                    A_eq_data = np.append(A_eq_data, 1)
                constraint_idx += 1
        A_eq_ex = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))#.toarray()
        b_eq_ex = np.ones(constraint_idx)#.toarray()




        # Constructing A_iq and b_iq for inequality as sp.coo matrix to bound z
        A_iq_ubz = sp.coo_matrix((np.ones(problem.num_z), (range(problem.num_z), range(problem.num_z))), shape=(problem.num_z, problem.num_vars))#.toarray()
        b_iq_ubz = np.ones(problem.num_z)#.toarray()


        # Constructing A_iq and b_iq for inequality as sp.coo matrix to bound y
        A_iq_row  = np.array([])
        A_iq_col  = np.array([])
        A_iq_data = np.array([])

        constraint_idx = 0
        for t in range(problem.T+1):
            for b in problem.graph.agents:
                for v in problem.graph.nodes:
                    A_iq_row = np.append(A_iq_row, constraint_idx)
                    A_iq_col = np.append(A_iq_col, problem.get_y_idx(b, v, t))
                    A_iq_data = np.append(A_iq_data, 1)
                    constraint_idx += 1
        A_iq_uby = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))#.toarray()
        b_iq_uby = np.ones(constraint_idx)#.toarray()












        """Generate equations (30) - (36)"""

        if problem.num_vars == None: problem.add_num_var()


        # Constructing A_eq and b_eq for equality (30) as sp.coo matrix
        A_iq_row  = np.array([])
        A_iq_col  = np.array([])
        A_iq_data = np.array([])

        constraint_idx = 0
        for t in range(problem.T):
            for b in range(len(self.problem.graph.agents)):
                for v1 in problem.graph.nodes:
                    for v2 in problem.graph.nodes:
                        A_iq_row = np.append(A_iq_row, constraint_idx)
                        A_iq_col = np.append(A_iq_col, problem.get_x_idx(b, v1, v2, t))
                        A_iq_data = np.append(A_iq_data, 1)

                        A_iq_row = np.append(A_iq_row, constraint_idx)
                        A_iq_col = np.append(A_iq_col, problem.get_e_idx(v1, v2, t))
                        A_iq_data = np.append(A_iq_data, -1)
                        constraint_idx += 1
        A_iq_30 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))#.toarray()
        b_iq_30 = np.zeros(constraint_idx)#.toarray()



        # Constructing A_eq and b_eq for equality (31) as sp.coo matrix
        A_eq_row  = np.array([])
        A_eq_col  = np.array([])
        A_eq_data = np.array([])

        constraint_idx = 0
        for t in range(problem.T):
            for v1 in problem.graph.nodes:
                for v2 in problem.graph.nodes:
                    for b in range(len(self.problem.graph.agents)):
                        if connectivity_adj[v1][v2] == 0:
                            A_eq_row = np.append(A_eq_row, constraint_idx)
                            A_eq_col = np.append(A_eq_col, problem.get_xbar_idx(b, v1, v2, t))
                            A_eq_data = np.append(A_eq_data, 1)
                            constraint_idx += 1
        A_eq_31 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(constraint_idx, problem.num_vars))#.toarray()
        b_eq_31 = np.zeros(constraint_idx)#.toarray()



        # Constructing A_eq and b_eq for equality (33) as sp.coo matrix
        A_iq_row  = np.array([])
        A_iq_col  = np.array([])
        A_iq_data = np.array([])

        constraint_idx = 0
        for t in range(problem.T+1):
            for v1 in problem.graph.nodes:
                for v2 in problem.graph.nodes:
                    for b in range(len(self.problem.graph.agents)):
                        A_iq_row = np.append(A_iq_row, constraint_idx)
                        A_iq_col = np.append(A_iq_col, problem.get_xbar_idx(b, v1, v2, t))
                        A_iq_data = np.append(A_iq_data, 1)
                        for r in range(len(self.problem.graph.agents)):
                            A_iq_row = np.append(A_iq_row, constraint_idx)
                            A_iq_col = np.append(A_iq_col, problem.get_z_idx(r, v1, t))
                            A_iq_data = np.append(A_iq_data, -1)
                        constraint_idx += 1

        A_iq_33 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))#.toarray()
        b_iq_33 = np.zeros(constraint_idx)#.toarray()


        # Constructing A_eq and b_eq for equality (34) as sp.coo matrix
        A_iq_row  = np.array([])
        A_iq_col  = np.array([])
        A_iq_data = np.array([])

        constraint_idx = 0
        for t in range(problem.T+1):
            for v1 in problem.graph.nodes:
                for v2 in problem.graph.nodes:
                    for b in range(len(self.problem.graph.agents)):
                        A_iq_row = np.append(A_iq_row, constraint_idx)
                        A_iq_col = np.append(A_iq_col, problem.get_xbar_idx(b, v1, v2, t))
                        A_iq_data = np.append(A_iq_data, 1)
                        for r in range(len(self.problem.graph.agents)):
                            A_iq_row = np.append(A_iq_row, constraint_idx)
                            A_iq_col = np.append(A_iq_col, problem.get_z_idx(r, v2, t))
                            A_iq_data = np.append(A_iq_data, -1)
                        constraint_idx += 1

        A_iq_34 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))#.toarray()
        b_iq_34 = np.zeros(constraint_idx)#.toarray()



        # Constructing A_eq and b_eq for equality (35) as sp.coo matrix
        A_iq_row  = np.array([])
        A_iq_col  = np.array([])
        A_iq_data = np.array([])


        constraint_idx = 0
        for t in range(problem.T):
            for v in problem.graph.nodes:
                for b in range(len(self.problem.graph.agents)):
                    A_iq_row = np.append(A_iq_row, constraint_idx)
                    A_iq_col = np.append(A_iq_col, problem.get_y_idx(b, v, t))
                    A_iq_data = np.append(A_iq_data, 1)
                    for r in range(len(self.problem.graph.agents)):
                        A_iq_row = np.append(A_iq_row, constraint_idx)
                        A_iq_col = np.append(A_iq_col, problem.get_z_idx(r, v, t))
                        A_iq_data = np.append(A_iq_data, -1)
                    constraint_idx += 1

        A_iq_35 = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))#.toarray()
        b_iq_35 = np.zeros(constraint_idx)#.toarray()



        # Constructing A_eq and b_eq for equality (36 left) as sp.coo matrix
        A_iq_row  = np.array([])
        A_iq_col  = np.array([])
        A_iq_data = np.array([])


        constraint_idx = 0
        for v in problem.graph.nodes:
            for b in range(len(self.problem.graph.agents)):
                A_iq_row = np.append(A_iq_row, constraint_idx)
                A_iq_col = np.append(A_iq_col, problem.get_y_idx(b, v, self.problem.T))
                A_iq_data = np.append(A_iq_data, 1)
                for r in range(len(self.problem.graph.agents)):
                    A_iq_row = np.append(A_iq_row, constraint_idx)
                    A_iq_col = np.append(A_iq_col, problem.get_z_idx(r, v, self.problem.T))
                    A_iq_data = np.append(A_iq_data, -1)
                constraint_idx += 1

        A_iq_36l = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))#.toarray()
        b_iq_36l = np.zeros(constraint_idx)#.toarray()



        # Constructing A_eq and b_eq for equality (36 right) as sp.coo matrix
        A_iq_row  = np.array([])
        A_iq_col  = np.array([])
        A_iq_data = np.array([])

        N = len(self.problem.graph.agents)

        constraint_idx = 0
        for v in problem.graph.nodes:
            for b in range(len(self.problem.graph.agents)):
                for r in range(len(self.problem.graph.agents)):
                    A_iq_row = np.append(A_iq_row, constraint_idx)
                    A_iq_col = np.append(A_iq_col, problem.get_z_idx(r, v, self.problem.T))
                    A_iq_data = np.append(A_iq_data, 1)
                A_iq_row = np.append(A_iq_row, constraint_idx)
                A_iq_col = np.append(A_iq_col, problem.get_y_idx(b, v, self.problem.T))
                A_iq_data = np.append(A_iq_data, -N)
                constraint_idx += 1

        A_iq_36r = sp.coo_matrix((A_iq_data, (A_iq_row, A_iq_col)), shape=(constraint_idx, problem.num_vars))#.toarray()
        b_iq_36r = np.zeros(constraint_idx)#.toarray()



        #Stack all (equality/inequality) constraints vertically
        self.A_eq = sp.bmat([[A_eq_35], [A_eq_36], [A_eq_32], [A_eq_ex], [A_eq_31]])
        self.b_eq = np.hstack([b_eq_35, b_eq_36, b_eq_32, b_eq_ex, b_eq_31])

        self.A_iq = sp.bmat([[A_iq_ubz], [A_iq_uby], [A_iq_id], [A_iq_30], [A_iq_33], [A_iq_34], [A_iq_35], [A_iq_36l], [A_iq_36r]])
        self.b_iq = np.hstack([b_iq_ubz, b_iq_uby, b_iq_id, b_iq_30, b_iq_33, b_iq_34, b_iq_35, b_iq_36l, b_iq_36r])


#Connectivity constraints-------------------------------------------------------

class ConnectivityConstraint(object):
    def __init__(self):
        self.A_eq = None
        self.b_eq = None
        self.A_iq = None
        self.b_iq = None


#Connectivity problem-----=-----------------------------------------------------

class ConnectivityProblem(object):
    """For multiple classes"""
    def __init__(self):
        self.graph = None                    #Graph
        self.equality_constraints = None       #equality contraints
        self.inequality_constraints = None     #inequality contraints
        self.T = None                        #Time horizon
        self.b = None                        #bases (agent positions)
        self.num_b = None
        self.num_r = None
        self.num_v = None
        self.num_i = None
        self.num_j = None
        self.num_z = None
        self.num_e = None
        self.num_y = None
        self.num_x = None
        self.num_xbar = None
        self.num_vars = None

        self.solution = None

        # variables: z^b_rvt, e_ijt, y^b_vt, x^b_ijt, xbar^b_ijt

    def add_num_var(self):
        self.num_b = len(self.b)
        self.num_r = len(self.graph.agents)
        self.num_v = self.graph.num_nodes
        self.num_i = self.num_v
        self.num_j = self.num_v

        self.num_z = (self.T+1) * self.num_r * self.num_v
        self.num_e = self.T * self.num_i * self.num_j
        self.num_y = (self.T+1) * self.num_b * self.num_v
        self.num_x = self.T * self.num_b * self.num_i * self.num_j
        self.num_xbar = (self.T+1) * self.num_b * self.num_i * self.num_j

        self.num_vars = self.num_z + self.num_e + self.num_y + self.num_x + self.num_xbar
        print("Number of variables: {}".format(self.num_vars))


    def get_z_idx(self, r, v, t):
        start = 0
        z_t = (self.num_r * self.num_v) * t
        z_v = (self.num_r) * v
        z_r = r
        idx = start + z_t + z_v + z_r
        return(idx)

    def get_e_idx(self, i, j, t):
        start = self.num_z
        e_t = (self.num_i * self.num_j) * t
        e_i = (self.num_j) * i
        e_j = j
        idx = start + e_t + e_i + e_j
        return(idx)


    def get_y_idx(self, b, v, t):
        start = self.num_z + self.num_e
        y_t = (self.num_b * self.num_v) * t
        y_b = (self.num_v) * b
        y_v = v
        idx = start + y_t + y_b + y_v
        return(idx)

    def get_x_idx(self, b, i, j, t):
        start = self.num_z + self.num_e + self.num_y
        x_t = (self.num_b * self.num_i * self.num_j) * t
        x_b = (self.num_i * self.num_j) * b
        x_i = (self.num_j) * i
        x_j = j
        idx = start + x_t + x_b +x_i + x_j
        return(idx)


    def get_xbar_idx(self, b, i, j, t):
        start = self.num_z + self.num_e + self.num_y + self.num_x
        xbar_t = (self.num_b * self.num_i * self.num_j) * t
        xbar_b = (self.num_i * self.num_j) * b
        xbar_i = (self.num_j) * i
        xbar_j = j
        idx = start + xbar_t + xbar_b +xbar_i + xbar_j
        return(idx)

    def get_time_augmented_id(self, n, t):
        idx = (self.graph.num_nodes) * t + n
        return(idx)

    def get_time_augmented_n_t(self, id):
        n = id % self.graph.num_nodes
        t = id // self.graph.num_nodes
        return(n,t)


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


    def compile(self):

        # Constructing A_eq and b_eq for initial condition as sp.coo matrix
        A_init_row  = np.array([])
        A_init_col  = np.array([])
        A_init_data = np.array([])
        b_init = []

        constraint_idx = 0
        for r in self.graph.agents:
            for v in self.graph.nodes:
                A_init_row = np.append(A_init_row, constraint_idx)
                A_init_col = np.append(A_init_col, self.get_z_idx(r, v, 0))
                A_init_data = np.append(A_init_data, 1)
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
                            if len(G.nodes[n]['agents']) != 0 and len(G.nodes[nbr]['agents'])!= 0:
                                G[n][nbr][edge]['color']='black'
                                G[n][nbr][edge]['style']='solid'
                            else:
                                G[n][nbr][edge]['color']='grey'
                                G[n][nbr][edge]['style']='solid'


            #Plot/save graph
            A = to_agraph(G)
            A.layout()
            A.draw('solution.png')



    def check_well_defined(self):
        # Check input data
        if self.T is None:
            raise Exception("No time horizon 'T' specified")

        if self.graph is None:
            raise Exception("No graph 'G' specified")



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
















































    def test_solution(self):
        for g in range(len(self.graphs)):
            # Check dynamics
            np.testing.assert_almost_equal(
                self.x[g][:, 1:],
                self.graphs[g].system_matrix().dot(self.u[g])
            )

            # Check control constraints
            np.testing.assert_almost_equal(
                self.x[g][:, :-1],
                _id_stacked(self.graphs[g].K(), self.graphs[g].M())
                    .dot(self.u[g])
            )

            # Check prefix-suffix connection
            assgn_sum_g = np.zeros(self.graphs[g].K())
            for C, a in zip(self.cycle_sets[g], self.assignments[g]):
                for (Ci, ai) in zip(C, a):
                    assgn_sum_g[self.graphs[g].order_fcn(Ci[0])] += ai

            np.testing.assert_almost_equal(
                self.x[g][:, -1],
                assgn_sum_g
            )

        # Check counting constraints
        for cc in self.constraints:
            for t in range(1000):  # enough to do T + LCM(cycle_lengths)
                assert(self.mode_count(cc.X, t) <= cc.R)

    def mode_count(self, X, t):
        """When a solution has been found, return the `X`-count
        at time `t`"""
        X_count = 0
        for g in range(len(self.graphs)):
            G = self.graphs[g]
            if t < self.T:
                u_t_g = self.u[g][:, t]
                X_count += sum(u_t_g[G.order_fcn(v) +
                                     G.index_of_mode(m) * G.K()]
                               for (v, m) in X[g])
            else:
                ass_rot = [deque(a) for a in self.assignments[g]]
                for a in ass_rot:
                    a.rotate(t - self.T)
                X_count += sum(np.inner(_cycle_row(C, X[g]), a)
                               for C, a in zip(self.cycle_sets[g],
                                               ass_rot))
        return X_count

    def get_aggregate_input(self, t):
        """When an integer solution has been found, return aggregate inputs `u`
            at time t"""
        if self.u is None:
            raise Exception("No solution available")

        agg_u = []

        for g in range(len(self.graphs)):
            G = self.graphs[g]

            if t < self.T:
                # We are in prefix; states are available
                u_g = self.u[g][:, t]

            else:
                u_g = np.zeros(G.K() * G.M())
                # Rotate assignments
                ass_rot = [deque(a) for a in self.assignments[g]]
                for a in ass_rot:
                    a.rotate(t - self.T)

                for assgn, c in zip(ass_rot,
                                    self.cycle_sets[g]):
                    for ai, ci in zip(assgn, c):
                        u_g[G.order_fcn(ci[0]) +
                            G.index_of_mode(ci[1]) * G.K()] += ai
            agg_u.append(u_g)

        return agg_u


    def get_input(self, xi_list, t):
        """When an integer solution has been found, given individual states
        `xi_list` at time t, return individual inputs `sigma`"""
        if self.u is None:
            raise Exception("No solution available")

        actions = []

        for g in range(len(self.graphs)):
            G = self.graphs[g]
            N_g = np.sum(self.inits[g])

            actions_g = [None] * N_g

            if t < self.T:
                # We are in prefix; states are available
                x_g = self.x[g][:, t]
                u_g = self.u[g][:, t]

            else:
                # We are in suffix, must build x_g, u_g from cycle assignments
                u_g = np.zeros(G.K() * G.M())
                x_g = np.zeros(len(self.graphs[g]))

                # Rotate assignments
                ass_rot = [deque(a) for a in self.assignments[g]]
                for a in ass_rot:
                    a.rotate(t - self.T)

                for assgn, c in zip(ass_rot,
                                    self.cycle_sets[g]):
                    x_g += self.graphs[g].index_matrix(c) \
                               .dot(assgn).flatten()
                    for ai, ci in zip(assgn, c):
                        u_g[G.order_fcn(ci[0]) +
                            G.index_of_mode(ci[1]) * G.K()] += ai

            # Assert that xi_list agrees with aggregate solution
            xi_sum = np.zeros(len(self.graphs[g]))
            for xi in xi_list[g]:
                xi_sum[self.graphs[g].order_fcn(xi)] += 1
            print(xi_sum)
            try:
                np.testing.assert_almost_equal(xi_sum,
                                               x_g)
            except:
                raise Exception("States don't agree with aggregate" +
                                " at time " + str(t))

            for n in range(N_g):
                k = G.order_fcn(xi_list[g][n])
                u_state = [u_g[k + G.K() * m] for m in range(G.M())]
                m = next(i for i in range(len(u_state))
                         if u_state[i] >= 1)
                actions_g[n] = G.mode(m)
                u_g[k + G.K() * m] -= 1
            actions.append(actions_g)

        return actions
























    def solve_prefix_suffix(self, solver=None, output=False, integer=True):

        # self.check_well_defined()

        # Variables for each g in self.graphs:
        #  v_g := u_g[0] ... u_g[T-1] x_g[0] ... x_g[T-1]
        #         a_g[0] ... a_g[C1-1] b[0] ... b[LJ-1]
        # These are stacked horizontally as
        #  v_0 v_1 ... v_G-1

        L = len(self.constraints)
        T = self.T

        # Variable counts for each class g
        N_u = T * self.graph.K() * self.graph.M()   # input vars
        N_x = T * self.graph.K()   # state vars

        N_tot = sum(N_u) + sum(N_x)

        # Add dynamic constraints
        A_eq_list = []
        b_eq_list = []

        A_eq1_u, A_eq1_x, b_eq1 = \
            generate_prefix_dyn_cstr(self.graph, T)
        A_eq2_x, A_eq2_a, b_eq2 = \
            generate_prefix_suffix_cstr(self.graph, T)
        A_eq1_b = sp.coo_matrix((A_eq1_u.shape[0], N_b_list[g]))

        A_eq_list.append(
            sp.bmat([[A_eq1_u, A_eq1_x, None, A_eq1_b],
                     [None, A_eq2_x, A_eq2_a, None]])
        )
        b_eq_list.append(
            np.hstack([b_eq1, b_eq2])
        )

        A_eq = sp.block_diag(A_eq_list)
        b_eq = np.hstack(b_eq_list)

        # Add counting constraints
        A_iq_list = []
        b_iq_list = []
        for l in range(len(self.constraints)):
            cc = self.constraints[l]

            # Count over classes: Should be stacked horizontally
            A_iq1_list = []

            # Bounded by slack vars: Should be block diagonalized
            A_iq2_list = []
            b_iq2_list = []

            # Count over bound vars for each class: Should be stacked
            # horizontally
            A_iq3_list = []

            for g in range(len(self.graphs)):
                # Prefix counting
                A_iq1_u, b_iq1 = \
                    generate_prefix_counting_cstr(self.graphs[g], T,
                                                  cc.X[g], cc.R)
                A_iq1_list.append(
                    sp.bmat([[A_iq1_u, sp.coo_matrix((T, N_x_list[g] +
                                                      N_a_list[g] +
                                                      N_b_list[g]))]])
                )

                # Suffix counting
                A_iq2_a, A_iq2_b, b_iq2, A_iq3_a, A_iq3_b, b_iq3 = \
                    generate_suffix_counting_cstr(self.cycle_sets[g],
                                                  cc.X[g], cc.R)

                b_head2 = sp.coo_matrix((N_a_list[g], l * J_list[g]))
                b_tail2 = sp.coo_matrix((N_a_list[g], (L - 1 - l) * J_list[g]))
                A_iq2_b = sp.bmat([[b_head2, A_iq2_b, b_tail2]])
                b_head3 = sp.coo_matrix((1, l * J_list[g]))
                b_tail3 = sp.coo_matrix((1, (L - 1 - l) * J_list[g]))
                A_iq3_b = sp.bmat([[b_head3, A_iq3_b, b_tail3]])

                A_iq2_u = sp.coo_matrix((N_a_list[g], N_u_list[g]))
                A_iq2_x = sp.coo_matrix((N_a_list[g], N_x_list[g]))

                A_iq2_list.append(
                    sp.bmat([[A_iq2_u, A_iq2_x, A_iq2_a, A_iq2_b]])
                )
                b_iq2_list.append(b_iq2)

                A_iq3_list.append(
                    sp.bmat([[sp.coo_matrix((1, N_u_list[g] + N_x_list[g])),
                              A_iq3_a, A_iq3_b]])
                )

            # Stack horizontally
            A_iq_list.append(sp.bmat([A_iq1_list]))
            b_iq_list.append(b_iq1)

            # Stack by block
            A_iq_list.append(sp.block_diag(A_iq2_list))
            b_iq_list.append(np.hstack(b_iq2_list))

            # Stack horizontally
            A_iq_list.append(sp.bmat([A_iq3_list]))
            b_iq_list.append(b_iq3)

        # Stack everything vertically
        if len(A_iq_list) > 0:
            A_iq = sp.bmat([[A] for A in A_iq_list])
            b_iq = np.hstack(b_iq_list)
        else:
            A_iq = sp.coo_matrix((0, N_tot))
            b_iq = np.zeros(0)

        # Solve it
        if integer:
            sol = solve_mip(np.zeros(N_tot), A_iq, b_iq, A_eq, b_eq,
                            range(N_tot), solver, output);
        else:
            sol = solve_mip(np.zeros(N_tot), A_iq, b_iq, A_eq, b_eq,
                            [], solver, output);

        # Extract solution (if valid)
        if sol['status'] == 2:
            self.u = []
            self.x = []
            self.assignments = []

            idx0 = 0
            for g in range(len(self.graphs)):
                self.u.append(
                    np.array(sol['x'][idx0:idx0 + N_u_list[g]], dtype=int)
                      .reshape(T, self.graphs[g].K() * self.graphs[g].M())
                      .transpose()
                )
                self.x.append(
                    np.hstack([
                        np.array(self.inits[g]).reshape(len(self.inits[g]), 1),
                        np.array(sol['x'][idx0 + N_u_list[g]:
                                          idx0 + N_u_list[g] + N_x_list[g]])
                          .reshape(T, self.graphs[g].K()).transpose()
                    ])
                )

                cycle_lengths = [len(C) for C in self.cycle_sets[g]]
                self.assignments.append(
                    [sol['x'][idx0 + N_u_list[g] + N_x_list[g] + an:
                              idx0 + N_u_list[g] + N_x_list[g] + an + dn]
                     for an, dn in zip(np.cumsum([0] +
                                       cycle_lengths[:-1]),
                                       cycle_lengths)]
                )
                idx0 += N_u_list[g] + N_x_list[g] + N_a_list[g] + N_b_list[g]
        return sol['status']


    def test_solution(self):
        for g in range(len(self.graphs)):
            # Check dynamics
            np.testing.assert_almost_equal(
                self.x[g][:, 1:],
                self.graphs[g].system_matrix().dot(self.u[g])
            )

            # Check control constraints
            np.testing.assert_almost_equal(
                self.x[g][:, :-1],
                _id_stacked(self.graphs[g].K(), self.graphs[g].M())
                    .dot(self.u[g])
            )

            # Check prefix-suffix connection
            assgn_sum_g = np.zeros(self.graphs[g].K())
            for C, a in zip(self.cycle_sets[g], self.assignments[g]):
                for (Ci, ai) in zip(C, a):
                    assgn_sum_g[self.graphs[g].order_fcn(Ci[0])] += ai

            np.testing.assert_almost_equal(
                self.x[g][:, -1],
                assgn_sum_g
            )

        # Check counting constraints
        for cc in self.constraints:
            for t in range(1000):  # enough to do T + LCM(cycle_lengths)
                assert(self.mode_count(cc.X, t) <= cc.R)

    def mode_count(self, X, t):
        """When a solution has been found, return the `X`-count
        at time `t`"""
        X_count = 0
        for g in range(len(self.graphs)):
            G = self.graphs[g]
            if t < self.T:
                u_t_g = self.u[g][:, t]
                X_count += sum(u_t_g[G.order_fcn(v) +
                                     G.index_of_mode(m) * G.K()]
                               for (v, m) in X[g])
            else:
                ass_rot = [deque(a) for a in self.assignments[g]]
                for a in ass_rot:
                    a.rotate(t - self.T)
                X_count += sum(np.inner(_cycle_row(C, X[g]), a)
                               for C, a in zip(self.cycle_sets[g],
                                               ass_rot))
        return X_count

    def get_aggregate_input(self, t):
        """When an integer solution has been found, return aggregate inputs `u`
            at time t"""
        if self.u is None:
            raise Exception("No solution available")

        agg_u = []

        for g in range(len(self.graphs)):
            G = self.graphs[g]

            if t < self.T:
                # We are in prefix; states are available
                u_g = self.u[g][:, t]

            else:
                u_g = np.zeros(G.K() * G.M())
                # Rotate assignments
                ass_rot = [deque(a) for a in self.assignments[g]]
                for a in ass_rot:
                    a.rotate(t - self.T)

                for assgn, c in zip(ass_rot,
                                    self.cycle_sets[g]):
                    for ai, ci in zip(assgn, c):
                        u_g[G.order_fcn(ci[0]) +
                            G.index_of_mode(ci[1]) * G.K()] += ai
            agg_u.append(u_g)

        return agg_u


    def get_input(self, xi_list, t):
        """When an integer solution has been found, given individual states
        `xi_list` at time t, return individual inputs `sigma`"""
        if self.u is None:
            raise Exception("No solution available")

        actions = []

        for g in range(len(self.graphs)):
            G = self.graphs[g]
            N_g = np.sum(self.inits[g])

            actions_g = [None] * N_g

            if t < self.T:
                # We are in prefix; states are available
                x_g = self.x[g][:, t]
                u_g = self.u[g][:, t]

            else:
                # We are in suffix, must build x_g, u_g from cycle assignments
                u_g = np.zeros(G.K() * G.M())
                x_g = np.zeros(len(self.graphs[g]))

                # Rotate assignments
                ass_rot = [deque(a) for a in self.assignments[g]]
                for a in ass_rot:
                    a.rotate(t - self.T)

                for assgn, c in zip(ass_rot,
                                    self.cycle_sets[g]):
                    x_g += self.graphs[g].index_matrix(c) \
                               .dot(assgn).flatten()
                    for ai, ci in zip(assgn, c):
                        u_g[G.order_fcn(ci[0]) +
                            G.index_of_mode(ci[1]) * G.K()] += ai

            # Assert that xi_list agrees with aggregate solution
            xi_sum = np.zeros(len(self.graphs[g]))
            for xi in xi_list[g]:
                xi_sum[self.graphs[g].order_fcn(xi)] += 1
            print(xi_sum)
            try:
                np.testing.assert_almost_equal(xi_sum,
                                               x_g)
            except:
                raise Exception("States don't agree with aggregate" +
                                " at time " + str(t))

            for n in range(N_g):
                k = G.order_fcn(xi_list[g][n])
                u_state = [u_g[k + G.K() * m] for m in range(G.M())]
                m = next(i for i in range(len(u_state))
                         if u_state[i] >= 1)
                actions_g[n] = G.mode(m)
                u_g[k + G.K() * m] -= 1
            actions.append(actions_g)

        return actions
