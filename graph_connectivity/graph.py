import networkx as nx

from networkx.drawing.nx_agraph import to_agraph
from copy import deepcopy

class Graph(nx.MultiDiGraph):

    def __init__(self):
        super(Graph, self).__init__()
        self.agents = None
        self.std_tran_weight = 1
        self.std_con_weight = 0.01

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

    def add_transition_path(self, transition_list, w = None):
        if w == None:
            w = self.std_tran_weight
        self.add_path(transition_list, type = 'transition', weight = w)
        self.add_path(transition_list[::-1], type='transition', weight = w)
        self.add_self_loops()

    def add_connectivity_path(self, connectivity_list, w = None):
        if w == None:
            w = self.std_con_weight
        self.add_path(connectivity_list, type='connectivity', weight = w)
        self.add_path(connectivity_list[::-1] , type='connectivity', weight = w)

    def add_self_loops(self):
        for n in self:
            add_transition = True
            for edge in self.edges(n, data = True):
                if edge[0] == edge[1]:
                    if edge[2]['type'] == 'transition':
                        add_transition = False
            if add_transition:
                self.add_edge(n, n, type='transition', weight = 0)

    def set_frontiers(self, frontiers):
        for v in self:
            if v in frontiers:
                self.nodes[v]['frontiers'] = frontiers[v]
            else:
                self.nodes[v]['frontiers'] = 0

    def is_frontier(self, v):
        frontier = False
        if self.node[v]['known']:
            for edge in self.tran_out_edges(v):
                if not self.node[edge[1]]['known']:
                    frontier = True
        return frontier

    def is_local_frontier(self, v):
        if self.node[v]['frontiers'] != 0:
            return True
        return False

    def is_known(self):
        for v in self:
            if not self.node[v]['known']:
                return False
        return True


    def set_node_positions(self, position_dictionary):
        self.add_nodes_from(position_dictionary.keys())
        for n, p in position_dictionary.items():
            self.node[n]['x'] = p[0]
            self.node[n]['y'] = p[1]

    def conn_edges(self):
        for (i, j, data) in self.edges(data=True):
            if data['type'] == 'connectivity':
                yield (i,j)

    def tran_edges(self):
        for (i, j, data) in self.edges(data=True):
            if data['type'] == 'transition':
                yield (i,j)

    def conn_in_edges(self, k):
        for (i, j, data) in self.in_edges(k, data=True):
            if data['type'] == 'connectivity':
                yield (i,j)

    def tran_in_edges(self, k):
        for (i, j, data) in self.in_edges(k, data=True):
            if data['type'] == 'transition':
                yield (i,j)

    def conn_out_edges(self, k):
        for (i, j, data) in self.out_edges(k, data=True):
            if data['type'] == 'connectivity':
                yield (i,j)

    def tran_out_edges(self, k):
        for (i, j, data) in self.out_edges(k, data=True):
            if data['type'] == 'transition':
                yield (i,j)

    def init_agents(self, agent_dictionary):

        self.agents = agent_dictionary

        for n in self:
            self.node[n]['number_of_agents']=0
            self.node[n]['agents'] = []
            self.node[n]['known'] = False

        for r, n in self.agents.items():
            self.node[n]['known'] = True

        for agent, position in agent_dictionary.items():
            self.node[position]['number_of_agents'] += 1
            self.node[position]['agents'].append(agent)

    def transition_adjacency_matrix(self):
        num_nodes = self.number_of_nodes()
        adj = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
        for n in self:
            for edge in self.tran_in_edges(n):
                adj[edge[0]][edge[1]] = 1
        return adj

    def connectivity_adjacency_matrix(self):
        num_nodes = self.number_of_nodes()
        adj = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
        for n in self:
            for edge in self.conn_in_edges(n):
                adj[edge[0]][edge[1]] = 1
        return adj

    def get_pre_S_transition(self, S_v_t):
        pre_S = set()
        for v, t in S_v_t:
            if t > 0:
                for edge in self.tran_in_edges(v):
                    if (edge[0], t - 1) not in S_v_t:
                        pre_S.add((edge[0], edge[1], t - 1))
        return pre_S

    def get_pre_S_connectivity(self, S_v_t):
        pre_S = set()
        for v, t in S_v_t:
            for edge in self.conn_in_edges(v):
                if (edge[0], t) not in S_v_t:
                    pre_S.add((edge[0], edge[1], t))
        return pre_S
