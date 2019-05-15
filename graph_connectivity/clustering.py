import networkx as nx
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product
from sklearn.cluster import SpectralClustering
from sklearn import metrics

from graph import *

def create_agent_graph(G):

    G_tran = deepcopy(G)
    G_tran.remove_edges_from(list(G.conn_edges()))

    g = nx.Graph()
    for r in G.agents:
        g.add_node(r)

    agent_nodes = list(G.agents.values())

    for (r1,v1), (r2,v2) in product(G.agents.items(), G.agents.items()):
        if r1 == r2:
            add_path = False
        else:
            add_path = True
            shortest_path = nx.shortest_path(G_tran, source=v1, target=v2, weight='weight')
            for v in shortest_path:
                if v in agent_nodes:
                    if v != v1 and v != v2:
                        add_path = False
        if add_path:
            w = nx.shortest_path_length(G_tran, source=v1, target=v2, weight='weight')
            g.add_edge(r1, r2, weight=w, type = 'transition')

    #add small weight to every edge to prevent divide by zero. (w += 0.1 -> agents in same node has 10 similarity)
    for edge in g.edges:
        g.edges[edge]['weight'] += 0.1

    #inverting edge weights as spectral_clustering use them as similarity measure
    for edge in g.edges:
        g.edges[edge]['weight'] = 1/g.edges[edge]['weight']

    return g

def spectral_clustering(G, k):

    g = create_agent_graph(G)

    # Adjacency matrix
    adj = nx.to_numpy_matrix(g)

    # Cluster (uses weights as similarity measure)
    sc = SpectralClustering(k, affinity='precomputed')
    sc.fit(adj)

    print(sc.labels_)

    clusters = {}
    c_group = []

    for c in range(k):
        for (i,r) in enumerate(G.agents):
            if sc.labels_[i] == c:
                c_group.append(r)
        clusters['cluster' + str(c)] = c_group
        c_group = []

    return clusters

def inflate_subgraphs(G, clusters):






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

clusters = spectral_clustering(G,4)
print(clusters)
