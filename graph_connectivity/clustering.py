import networkx as nx
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product
from sklearn.cluster import SpectralClustering
from sklearn import metrics

from graph import *

class ClusterProblem(object):

    def __init__(self, graph, k, master):
        # Problem definition
        self.G = graph                   #Graph
        self.G_tran = None               #Transition-only graph
        self.k = k                       #number of clusters
        self.master = master
        self.agent_clusters = None
        self.subgraphs = None
        self.submasters = None
        self.subsinks = None
        self.priority_list = None
        self.free_nodes = None

    def create_G_tran(self):

        self.G_tran = deepcopy(self.G)
        self.G_tran.remove_edges_from(list(self.G.conn_edges()))

    def create_agent_graph(self):

        g = nx.Graph()
        for r in self.G.agents:
            g.add_node(r)

        agent_nodes = list(self.G.agents.values())

        for (r1,v1), (r2,v2) in product(self.G.agents.items(), self.G.agents.items()):
            if r1 == r2:
                add_path = False
            else:
                add_path = True
                shortest_path = nx.shortest_path(self.G_tran, source=v1, target=v2, weight='weight')
                for v in shortest_path:
                    if v in agent_nodes:
                        if v != v1 and v != v2:
                            add_path = False
            if add_path:
                w = nx.shortest_path_length(self.G_tran, source=v1, target=v2, weight='weight')
                g.add_edge(r1, r2, weight=w, type = 'transition')

        #add small weight to every edge to prevent divide by zero. (w += 0.1 -> agents in same node has 10 similarity)
        for edge in g.edges:
            g.edges[edge]['weight'] += 0.1

        #inverting edge weights as spectral_clustering use them as similarity measure
        for edge in g.edges:
            g.edges[edge]['weight'] = 1/g.edges[edge]['weight']

        return g

    def spectral_clustering(self):

        g = self.create_agent_graph()

        # Adjacency matrix
        adj = nx.to_numpy_matrix(g)

        # Cluster (uses weights as similarity measure)
        sc = SpectralClustering(self.k, affinity='precomputed')
        sc.fit(adj)

        self.agent_clusters = {}
        c_group = []

        for c in range(self.k):
            for (i,r) in enumerate(self.G.agents):
                if sc.labels_[i] == c:
                    c_group.append(r)
            self.agent_clusters['cluster' + str(c)] = c_group
            c_group = []

    def expand_cluster(self, c):

        agent_cluster = self.agent_clusters[c]
        node_cluster = self.subgraphs[c]

        agent_nodes = [self.G.agents[r] for r in agent_cluster]
        closest_node = None
        closest_distance = None
        for n in node_cluster:
            for nbr in nx.neighbors(self.G_tran, n):
                if nbr in self.free_nodes:
                    dist, path = nx.multi_source_dijkstra(self.G_tran, sources = agent_nodes, target=nbr)
                    add_path = True
                    for v in path[1:-1]:
                        if v in self.G.agents.values():
                            add_path = False
                    if add_path:
                        if closest_node == None:
                            closest_node = nbr
                            closest_distance = dist
                        elif dist < closest_distance:
                            closest_node = nbr
                            closest_distance = dist
                elif nbr in self.G.agents.values() and nbr not in node_cluster:
                    add_nbr = True
                    for cluster in self.subgraphs:
                        if nbr in self.subgraphs[cluster]:
                            for v in node_cluster:
                                if v in self.subgraphs[cluster]:
                                    add_nbr = False
                    if add_nbr:
                        dist, path = nx.multi_source_dijkstra(self.G_tran, sources = agent_nodes, target=nbr)
                        add_path = True
                        for v in path[1:-1]:
                            if v in self.G.agents.values():
                                add_path = False
                        if add_path:
                            if closest_node == None:
                                closest_node = nbr
                                closest_distance = dist
                            elif dist < closest_distance:
                                closest_node = nbr
                                closest_distance = dist


        return closest_node

    def create_cluster_priority_list(self):
        master_node = self.G.agents[self.master]
        G_tran = deepcopy(self.G)
        G_tran.remove_edges_from(list(self.G.conn_edges()))

        self.priority_list = []
        for c in self.subgraphs:
            dist, path = nx.multi_source_dijkstra(self.G_tran, sources = self.subgraphs[c], target=master_node)
            self.priority_list.append((c,dist))
        sorted_priority_list = sorted(self.priority_list, key=lambda x: x[1])
        self.priority_list = [c[0] for c in sorted_priority_list]

    def create_clusters(self):

        #create subgraphs corresponding to agent clusters
        self.subgraphs = {}
        for c in self.agent_clusters:
            self.subgraphs[c] = []
            for r in self.agent_clusters[c]:
                self.subgraphs[c].append(self.G.agents[r])

        #list of nodes available to be added into a cluster
        self.free_nodes = [n for n in self.G.nodes if n not in self.G.agents.values()]

        #assign priority to clusters
        self.create_cluster_priority_list()

        for c in self.priority_list:
            cluster_full = False
            while not cluster_full:
                expand_node = self.expand_cluster(c)
                if expand_node == None:
                    cluster_full = True
                else:
                    self.subgraphs[c].append(expand_node)
                    if expand_node in self.free_nodes:
                        self.free_nodes.remove(expand_node)

    def create_submaster_dictionary(self):
        self.submasters = {self.priority_list[0]: self.master}
        self.subsinks = {}
        for sub1 in range(len(self.priority_list)):
            for sub2 in range(sub1 + 1, len(self.priority_list)):
                n = list(set(self.subgraphs[self.priority_list[sub1]]) & set(self.subgraphs[self.priority_list[sub2]]))
                if n:
                    for r in self.G.agents:
                        if self.G.agents[r] in n:
                            self.submasters[self.priority_list[sub2]] = r
                            if self.priority_list[sub1] in self.subsinks:
                                self.subsinks[self.priority_list[sub1]].append(r)
                            else:
                                self.subsinks[self.priority_list[sub1]] = [r]

    def create_subgraphs(self):

        #create transition-only graph
        self.create_G_tran()

        #detect agent clusters
        self.spectral_clustering()

        #dictionary mapping cluster to nodes
        self.create_clusters()

        #dictionary mapping cluster to submaster, subsinks
        self.create_submaster_dictionary()




################################################################################


# Define a connectivity graph
G = Graph()
#Add edges to graph
connectivity_edges = [0, 1, 2, 3, 4, 5]
transition_edges = [0, 1, 2, 3, 4, 5]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [4, 6, 7, 8]
transition_edges = [4, 6, 7, 8]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [1, 9, 10, 11, 12, 13]
transition_edges = [1, 9, 10, 11, 12, 13]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [9, 14, 15, 12]
transition_edges = [9, 14, 15, 12]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [1, 16, 17, 18, 19, 20, 21, 22, 23, 24]
transition_edges = [1, 16, 17, 18, 19, 20, 21, 22, 23, 24]
G.add_transition_path(transition_edges)
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [8, 10]
G.add_connectivity_path(connectivity_edges)
connectivity_edges = [15, 24]
G.add_connectivity_path(connectivity_edges)
#Set node positions
node_positions = {0: (0,0), 1: (0,1), 2: (-2,1), 3: (-2,2), 4: (-3,2), 5: (-3,1),
                6: (-3,3), 7: (-3.5,3.5), 8: (-2.5,3.5), 9: (0,3), 10: (-1.8,3),
                11: (-1.6,4), 12: (0,4), 13: (0,5), 14: (1,3), 15: (1,4),
                16: (1.5,1), 17: (2.5,1.3), 18: (4,1.3), 19: (5,1.3), 20: (5,2),
                21: (4,3), 22: (5,3), 23: (5,4), 24: (3.5,4)}

G.set_node_positions(node_positions)

frontiers = {1: 1, 2: 1}
G.set_frontiers(frontiers)

#Set initial position of agents
agent_positions = {0: 0, 1: 1, 2: 4, 3: 7, 4: 12, 5: 14, 6: 19, 7: 24}    #agent:position
G.init_agents(agent_positions)

#Plot Graph (saves image as graph.png)
G.plot_graph()








#CLUSTERING
k = 4
master = 0
cp = ClusterProblem(G, k, master)

cp.create_subgraphs()

print("Agent clusters: {}".format(cp.agent_clusters))
print("Subgraphs: {}".format(cp.subgraphs))
print("Submasters: {}".format(cp.submasters))
print("Subsinks: {}".format(cp.subsinks))
