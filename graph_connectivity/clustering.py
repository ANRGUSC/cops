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

    clusters = {}
    c_group = []

    for c in range(k):
        for (i,r) in enumerate(G.agents):
            if sc.labels_[i] == c:
                c_group.append(r)
        clusters['cluster' + str(c)] = c_group
        c_group = []

    return clusters

def expand_cluster(G, c, agent_clusters, node_clusters, free_nodes):
    G_tran = deepcopy(G)
    G_tran.remove_edges_from(list(G.conn_edges()))

    agent_cluster = agent_clusters[c]
    node_cluster = node_clusters[c]

    agent_nodes = [G.agents[r] for r in agent_cluster]
    closest_node = None
    closest_distance = None
    for n in node_cluster:
        for nbr in nx.neighbors(G_tran, n):
            if nbr in free_nodes:
                dist, path = nx.multi_source_dijkstra(G_tran, sources = agent_nodes, target=nbr)
                add_path = True
                for v in path[1:-1]:
                    if v in G.agents.values():
                        add_path = False
                if add_path:
                    if closest_node == None:
                        closest_node = nbr
                        closest_distance = dist
                    elif dist < closest_distance:
                        closest_node = nbr
                        closest_distance = dist
            elif nbr in G.agents.values() and nbr not in node_cluster:
                add_nbr = True
                for cluster in node_clusters:
                    if nbr in node_clusters[cluster]:
                        for v in node_cluster:
                            if v in node_clusters[cluster]:
                                add_nbr = False
                if add_nbr:
                    dist, path = nx.multi_source_dijkstra(G_tran, sources = agent_nodes, target=nbr)
                    add_path = True
                    for v in path[1:-1]:
                        if v in G.agents.values():
                            add_path = False
                    if add_path:
                        if closest_node == None:
                            closest_node = nbr
                            closest_distance = dist
                        elif dist < closest_distance:
                            closest_node = nbr
                            closest_distance = dist


    return closest_node

def create_cluster_priority_list(G, master_node, node_clusters):
    G_tran = deepcopy(G)
    G_tran.remove_edges_from(list(G.conn_edges()))

    priority_list = []
    for c in node_clusters:
        dist, path = nx.multi_source_dijkstra(G_tran, sources = node_clusters[c], target=master_node)
        priority_list.append((c,dist))
    sorted_priority_list = sorted(priority_list, key=lambda x: x[1])
    priority_list = [c[0] for c in sorted_priority_list]
    return priority_list

def inflate_clusters(G, agent_clusters, master):

    #create node_clusters corresponding to agent clusters
    node_clusters = {}
    for c in agent_clusters:
        node_clusters[c] = []
        for r in agent_clusters[c]:
            node_clusters[c].append(G.agents[r])

    #list of nodes available to be added into a cluster
    free_nodes = [n for n in G.nodes if n not in G.agents.values()]

    #assign priority to clusters
    priority_list = create_cluster_priority_list(G, G.agents[master], node_clusters)

    for c in priority_list:
        cluster_full = False
        while not cluster_full:
            expand_node = expand_cluster(G, c, agent_clusters, node_clusters, free_nodes)
            if expand_node == None:
                cluster_full = True
            else:
                node_clusters[c].append(expand_node)
                if expand_node in free_nodes:
                    free_nodes.remove(expand_node)

    #dictionary mappeing cluster to submaster
    submasters = create_submaster_dictionary(G, node_clusters, priority_list, master)

    return node_clusters, submasters

def create_submaster_dictionary(G, subgraphs, priority_list, master):
    submasters = {priority_list[0]: master}
    for sub1 in range(len(priority_list)):
        for sub2 in range(sub1 + 1, len(priority_list)):
            n = list(set(subgraphs[priority_list[sub1]]) & set(subgraphs[priority_list[sub2]]))
            if n:
                for r in G.agents:
                    if G.agents[r] in n:
                        submasters[priority_list[sub2]] = r
    return submasters


def create_clusters(G,k):
    master = 0

    agent_clusters = spectral_clustering(G,k)
    print("Agent clusters: {}".format(agent_clusters))

    subgraphs, submasters = inflate_clusters(G, agent_clusters, master)
    print("Subgraphs: {}".format(subgraphs))
    print("Submasters: {}".format(submasters))






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


create_clusters(G,4)
