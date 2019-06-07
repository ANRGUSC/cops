import networkx as nx
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy
from itertools import product
from sklearn.cluster import SpectralClustering

from graph_connectivity.problem import *

class ClusterProblem(object):

    def __init__(self):
        # Problem definition
        self.graph = None                   #Graph
        self.graph_tran = None              #Transition-only graph
        self.k = None                       #number of clusters
        self.master = None
        self.static_agents = None
        self.T = None
        self.max_problem_size = 70
        self.agent_clusters = None
        self.subgraphs = None
        self.submasters = None
        self.subsinks = None
        self.priority_list = None
        self.free_nodes = None
        self.hier_ascend = None
        self.hier_descend = None
        self.problems = {}
        self.trajectories = {}
        self.conn = {}
        self.com_in = {}
        self.com_out = {}
        self.start_time = {}
        self.end_time = {}

    #===GRAPH HELPER FUNCTIONS==================================================

    def create_graph_tran(self):

        self.graph_tran = deepcopy(self.graph)
        self.graph_tran.remove_edges_from(list(self.graph.conn_edges()))

    def create_agent_graph(self):

        g = nx.Graph()
        for r in self.graph.agents:
            g.add_node(r)

        agent_nodes = list(self.graph.agents.values())

        for (r1,v1), (r2,v2) in product(self.graph.agents.items(), self.graph.agents.items()):
            if r1 == r2:
                add_path = False
            else:
                add_path = True
                shortest_path = nx.shortest_path(self.graph_tran, source=v1, target=v2, weight='weight')
                for v in shortest_path:
                    if v in agent_nodes:
                        if v != v1 and v != v2:
                            add_path = False
            if add_path:
                w = nx.shortest_path_length(self.graph_tran, source=v1, target=v2, weight='weight')
                g.add_edge(r1, r2, weight=w, type = 'transition')

        #add small weight to every edge to prevent divide by zero. (w += 0.1 -> agents in same node has 10 similarity)
        for edge in g.edges:
            g.edges[edge]['weight'] += 0.1

        #inverting edge weights as spectral_clustering use them as similarity measure
        for edge in g.edges:
            g.edges[edge]['weight'] = 1/g.edges[edge]['weight']

        return g

    #===CLUSTER FUNCTIONS=======================================================

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
            for (i,r) in enumerate(self.graph.agents):
                if sc.labels_[i] == c:
                    c_group.append(r)
            self.agent_clusters['cluster' + str(c)] = c_group
            c_group = []

    def expand_cluster(self, c):

        agent_cluster = self.agent_clusters[c]
        node_cluster = self.subgraphs[c]

        agent_nodes = [self.graph.agents[r] for r in agent_cluster]
        closest_node = None
        closest_distance = None
        for n in node_cluster:
            for nbr in nx.neighbors(self.graph_tran, n):
                if nbr in self.free_nodes:
                    dist, path = nx.multi_source_dijkstra(self.graph_tran, sources = agent_nodes, target=nbr)
                    add_path = True
                    for v in path[1:-1]:
                        if v in self.graph.agents.values():
                            add_path = False
                    if add_path:
                        if closest_node == None:
                            closest_node = nbr
                            closest_distance = dist
                        elif dist < closest_distance:
                            closest_node = nbr
                            closest_distance = dist
                elif nbr in self.graph.agents.values() and nbr not in node_cluster:
                    add_nbr = True
                    for cluster in self.subgraphs:
                        if nbr in self.subgraphs[cluster]:
                            for v in node_cluster:
                                if v in self.subgraphs[cluster]:
                                    add_nbr = False
                    if add_nbr:
                        dist, path = nx.multi_source_dijkstra(self.graph_tran, sources = agent_nodes, target=nbr)
                        add_path = True
                        for v in path[1:-1]:
                            if v in self.graph.agents.values():
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
        master_node = self.graph.agents[self.master]

        self.priority_list = []
        for c in self.subgraphs:
            dist, path = nx.multi_source_dijkstra(self.graph_tran, sources = self.subgraphs[c], target=master_node)
            self.priority_list.append((c,dist))
        sorted_priority_list = sorted(self.priority_list, key=lambda x: x[1])
        self.priority_list = [c[0] for c in sorted_priority_list]

    def create_clusters(self):

        #nodes free to add to a subgraph
        self.free_nodes = [n for n in self.graph.nodes]

        while len(self.free_nodes)>0:

            #nodes free to add to a subgraph
            self.free_nodes = [n for n in self.graph.nodes]

            #create subgraphs corresponding to agent clusters
            self.subgraphs = {}
            for c in self.agent_clusters:
                self.subgraphs[c] = []
                for r in self.agent_clusters[c]:
                    v = self.graph.agents[r]
                    if v not in self.subgraphs[c]:
                        self.subgraphs[c].append(v)
                        if v in self.free_nodes:
                            self.free_nodes.remove(v)

            #assign priority to clusters
            self.create_cluster_priority_list()

            active = [self.priority_list[0]]   #subgraphs that expands each iteration
            connected_components = True        #indicates if subgraphs are connected

            while len(active) > 0 and connected_components:

                for c in reversed(active):

                    #test if subgraph c is connected
                    g = deepcopy(self.graph_tran).to_undirected()
                    for C in self.priority_list:
                        if C == c:
                            print('hej')
                            disconnected_subgraphs = []
                            components = nx.connected_components(g)
                            agent_components = {}
                            for i, comp in enumerate(components):
                                print(comp)
                                for r in self.agent_clusters[c]:
                                    if self.graph.agents[r] in comp:
                                        if i in agent_components:
                                            agent_components[i].append(r)
                                        else:
                                            agent_components[i] = [r]
                            print(agent_components)
                            #split agent cluster if disconnected
                            if len(agent_components.keys()) > 1:
                                del self.agent_clusters[c]
                                for i in agent_components.keys():
                                    self.agent_clusters[c + '_' + str(i)] = agent_components[i]
                                connected_components = False
                                print(c + ' not connected!!!')
                            else:
                                connected_components = True

                                #add agent-pairwise path to subgraph
                                for r1, r2 in product(self.agent_clusters[c],self.agent_clusters[c]):
                                    path = nx.shortest_path(g, source=self.graph.agents[r1], target=self.graph.agents[r2], weight='weight', method='dijkstra')
                                    for v in path:
                                        if v not in self.subgraphs[c] and v in self.free_nodes:
                                            self.subgraphs[c].append(v)
                                            self.free_nodes.remove(v)
                                #inflate subgraphs using a free node
                                expand_node = self.expand_cluster(c)
                                if expand_node == None:
                                    active.remove(c)
                                else:
                                    self.subgraphs[c].append(expand_node)
                                    if expand_node in self.free_nodes:
                                        self.free_nodes.remove(expand_node)
                                    for c_act in self.subgraphs:
                                        if expand_node in self.subgraphs[c_act] and c_act not in active:
                                            active.append(c_act)

                        else:
                            #remove nodes in subgraph except agent nodes
                            remove_nodes = []
                            for v in self.subgraphs[C]:
                                remove_nodes.append(v)
                            g.remove_nodes_from(remove_nodes)

                    if not connected_components:
                        break

    def create_clusters_2(self):

        connected_components = False
        while not connected_components:

            #create subgraphs corresponding to agent clusters
            self.subgraphs = {}
            for c in self.agent_clusters:
                self.subgraphs[c] = []
                for r in self.agent_clusters[c]:
                    v = self.graph.agents[r]
                    if v not in self.subgraphs[c]:
                        self.subgraphs[c].append(v)

            #assign priority to clusters
            self.create_cluster_priority_list()


            #inflate subgraphs corresponding to agent clusters
            g = deepcopy(self.graph_tran).to_undirected()
            for c in self.priority_list:

                #test if subgraphs are connected
                disconnected_subgraphs = []
                g = deepcopy(self.graph_tran).to_undirected()
                components = nx.connected_components(g)
                agent_components = {}
                for i, comp in enumerate(components):
                    for r in self.agent_clusters[c]:
                        if self.graph.agents[r] in comp:
                            if i in agent_components:
                                agent_components[i].append(r)
                            else:
                                agent_components[i] = [r]

                if len(agent_components.keys()) > 1:
                    del self.agent_clusters[c]
                    for i in agent_components.keys():
                        self.agent_clusters[c + '_' + str(i)] = agent_components[i]
                    connected_components = False
                else:
                    connected_components = True

                    g.remove_nodes_from(self.subgraphs[c])
                    for r1, r2 in product(self.agent_clusters[c],self.agent_clusters[c]):
                        path = nx.shortest_path(g, source=self.graph.agents[r1], target=self.graph.agents[r2], weight='weight', method='dijkstra')
                        for v in path:
                            if v not in self.subgraphs[c]:
                                add = True
                                for C in self.priority_list:
                                    if v not in self.subgraphs[C]:
                                        add = False
                                if add:
                                    self.subgraphs[c].append(v)
                    g.remove_nodes_from(self.subgraphs[c])


        '''
        #list of nodes available to be added into a cluster
        self.free_nodes = [n for n in self.graph.nodes if n not in self.graph.agents.values()]

        #inflate subgraphs using free nodes
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
        '''


        #create subgraphs corresponding to agent clusters
        self.subgraphs = {}
        for c in self.agent_clusters:
            self.subgraphs[c] = []
            for r in self.agent_clusters[c]:
                v = self.graph.agents[r]
                if v not in self.subgraphs[c]:
                    self.subgraphs[c].append(v)

        #assign priority to clusters
        self.create_cluster_priority_list()

        #inflate subgraphs corresponding to agent clusters
        g = deepcopy(self.graph_tran).to_undirected()
        for c in self.priority_list:
            for r1, r2 in product(self.agent_clusters[c],self.agent_clusters[c]):
                path = nx.shortest_path(g, source=self.graph.agents[r1], target=self.graph.agents[r2], weight='weight', method='dijkstra')
                for v in path:
                    if v not in self.subgraphs[c]:
                        add = True
                        for C in self.priority_list:
                            if v not in self.subgraphs[C]:
                                add = False
                        if add:
                            self.subgraphs[c].append(v)
            g.remove_nodes_from(self.subgraphs[c])


        #list of nodes available to be added into a cluster
        self.free_nodes = [n for n in self.graph.nodes if n not in self.graph.agents.values()]

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

    def create_clusters_3(self):

        # node clusters with agent positions
        clusters = {c : set(self.graph.agents[r] for r in r_list)  
                    for c, r_list in self.agent_clusters.items()}
        agent_node_clusters = {c : [self.graph.agents[r] for r in r_list]  
                               for c, r_list in self.agent_clusters.items()}

        child_clusters = {c : [] for c in self.agent_clusters.keys()}
        parent_clusters = {c : [] for c in self.agent_clusters.keys()}

        # start with master cluster active 
        active_clusters = [c for c, r_list in self.agent_clusters.items() 
                           if self.master in r_list]

        # nodes free to add to a subgraph
        free_nodes = set(self.graph.nodes) \
                     - set.union(*[set(v) for v in clusters.values()])

        while len(free_nodes) > 0:

            # make sure clusters are connected via transitions, if not split
            for c, v_set in clusters.items():
                c_subg = self.graph_tran.subgraph(v_set | free_nodes).to_undirected()
                
                comps = nx.connected_components(c_subg)
                comp_dict = { i: [r for r in self.agent_clusters[c] 
                                  if self.graph.agents[r] in comp] 
                              for i, comp in enumerate(comps)}

                comp_dict = {i:l for i,l in comp_dict.items() if len(l) > 0}

                if len(comp_dict) > 1:
                    # split and restart
                    del self.agent_clusters[c]
                    for i, r_list in comp_dict.items():
                        self.agent_clusters["{}_{}".format(c, i+1)] = r_list
                    return self.create_clusters_3()


            # find neighbors as candidate new nodes
            neighbors = [n for v in self.graph.nodes
                         for n in nx.neighbors(self.graph_tran, v) 
                         if n in free_nodes and v not in free_nodes]

            # find closest neighbor to any cluster
            new_cluster, new_node, min_dist = -1, -1, 99999
            for c in active_clusters:
                c_neighbors = set(n for v in clusters[c]
                                  for n in nx.neighbors(self.graph_tran, v) 
                                  if n not in clusters[c])

                for n in neighbors:
                    if n in c_neighbors:
                        dist = nx.multi_source_dijkstra(self.graph_tran, 
                                                   sources=agent_node_clusters[c], 
                                                   target=n)[0] 
                        if dist < min_dist:
                            min_dist, new_node, new_cluster = dist, n, c

            clusters[new_cluster].add(new_node)
            free_nodes.remove(new_node)

            cv_list = self.check(new_node, new_cluster, clusters, active_clusters)

            while len(cv_list) > 0:
                v0, c0, v1, c1 = cv_list.pop(0)
                active_clusters.append(c1)

                child_clusters[c0].append((c1, v1))
                parent_clusters[c1].append((c0, v0))

                cv_list += self.check(v1, c1, clusters, active_clusters)
                
        return clusters, child_clusters, parent_clusters


    def check(self, new_node, new_cluster, clusters, active_clusters):

        ret = []

        # check if new node adjacent to non-active cluster
        for c in clusters.keys():
            if c not in active_clusters:

                for v, edge in product(clusters[c], 
                                       self.graph.out_edges(new_node, data=True)):
                    if edge[1] == v:
                        ret.append((new_node, new_cluster, v, c))
                        break
        return ret

    def create_submaster_subsinks_dictionary(self):
        self.submasters = {self.priority_list[0]: self.master}
        self.subsinks = {}
        for sub1 in range(len(self.priority_list)):
            for sub2 in range(sub1 + 1, len(self.priority_list)):
                n = list(set(self.subgraphs[self.priority_list[sub1]]) & set(self.subgraphs[self.priority_list[sub2]]))
                if n:
                    for r in self.graph.agents:
                        if self.graph.agents[r] in n:
                            self.submasters[self.priority_list[sub2]] = r
                            if self.priority_list[sub1] in self.subsinks:
                                self.subsinks[self.priority_list[sub1]].append(r)
                            else:
                                self.subsinks[self.priority_list[sub1]] = [r]

    def create_hierarchical_dictionary(self):
        self.hier_ascend = {}
        self.hier_descend = {}
        for sub1, sub2 in product(self.subgraphs, self.subgraphs):
            if sub1 in self.submasters and sub2 in self.subsinks:
                if self.submasters[sub1] in self.subsinks[sub2]:
                    if sub2 in self.hier_ascend:
                        self.hier_ascend[sub2].append(sub1)
                    else:
                        self.hier_ascend[sub2] = [sub1]
                    if sub1 in self.hier_descend:
                        self.hier_descend[sub1].append(sub2)
                    else:
                        self.hier_descend[sub1] = [sub2]

    def create_subgraphs(self):

        #create transition-only graph
        self.create_graph_tran()

        if self.k == None:
            small_problem_size = False
            self.k = 1
            while not small_problem_size:

                #detect agent clusters
                self.spectral_clustering()

                #dictionary mapping cluster to nodes
                self.create_clusters_3()
                '''
                print(self.k)
                print(self.graph.agents)
                print(self.agent_clusters)
                print(self.subgraphs)
                '''

                #dictionary mapping cluster to submaster, subsinks
                self.create_submaster_subsinks_dictionary()

                small_problem_size = True
                for c in self.subgraphs:
                    g = deepcopy(self.graph)
                    out_of_cluster_nodes = set(self.graph.nodes) - set(self.subgraphs[c])
                    g.remove_nodes_from(out_of_cluster_nodes)
                    if c in self.subsinks:
                        size = (len(self.subgraphs[c])-len(self.subsinks[c])) * nx.diameter(g)
                    else:
                        size = len(self.subgraphs[c]) * nx.diameter(g)
                    if size > self.max_problem_size:
                        small_problem_size = False
                        self.k += 1
                        break

                if len(self.agent_clusters) == len(self.graph.agents):
                    raise Exception("Too large graph, add more agents!")

        else:
            #detect agent clusters
            self.spectral_clustering()

            #dictionary mapping cluster to nodes
            clusters, parents, children = self.create_clusters_3()

            print(clusters)
            print(parents)
            print(children)

        #dictionary mapping cluster to submaster, subsinks
        self.create_submaster_subsinks_dictionary()

        #dictionary mapping cluster to submaster, subsinks
        self.create_hierarchical_dictionary()

    def frontiers_in_cluster(self, c):
        cluster_frontiers = []
        for v in self.graph.nodes:
            if self.graph.nodes[v]['frontiers'] != 0 and v in self.subgraphs[c] and v != self.subsinks[c]:
                cluster_frontiers.append(v)
        return cluster_frontiers

    def active_subgraphs(self):
        active_subgraphs = []
        for c in reversed(self.priority_list):
            include_frontier = False
            for v in self.graph.nodes:
                if self.graph.node[v]['frontiers'] != 0 and v in self.subgraphs[c]:
                    include_frontier = True
            if include_frontier:
                active_subgraphs.append(c)
            elif c in self.hier_ascend and c not in active_subgraphs:
                for pre_c in self.hier_ascend[c]:
                    if pre_c in active_subgraphs and c not in active_subgraphs:
                        active_subgraphs.append(c)
        return active_subgraphs

    #===SOLVE FUNCTIONS=========================================================

    def augment_solutions(self):

        #Construct communication dictionaries
        for c in self.priority_list:
            if c in self.problems:
                for t, (b, br), (v1, v2) in product(range(self.problems[c].T+1), enumerate(self.problems[c].min_src_snk), self.problems[c].graph.conn_edges()):
                    if self.problems[c].solution['x'][self.problems[c].get_fbar_idx(b, v1, v2, t)] > 0.5:
                        for r in self.problems[c].graph.agents:
                            if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v1, t)] > 0.5:
                                r_out = r
                            if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v2, t)] > 0.5:
                                r_in = r
                        self.com_out[r_out,t] = (r_in, v2, br)
                        self.com_in[r_in,t] = (r_out, v1, br)
                        self.conn[(br, v1, v2, t)] = 1

                for t, (v1, v2) in product(range(self.problems[c].T+1), self.problems[c].graph.conn_edges()):
                    if self.problems[c].solution['x'][self.problems[c].get_mbar_idx(v1, v2, t)] > 0.5:
                        for r in self.problems[c].graph.agents:
                            if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v1, t)] > 0.5:
                                r_out = r
                            if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v2, t)] > 0.5:
                                r_in = r
                        self.com_out[r_out,t] = (r_in, v2, self.problems[c].master)
                        self.com_in[r_in,t] = (r_out, v1, self.problems[c].master)
                        self.conn[(self.problems[c].master, v1, v2, t)] = 1

        #Find start time for each cluster
        for c in self.priority_list:
            if c in self.problems:
                if self.problems[c].master == self.master:
                    self.start_time[c] = 0
                if c in self.hier_ascend:
                    for c_asc in self.hier_ascend[c]:
                        submaster = self.submasters[c_asc]
                        #find time when submaster in upper subgraph is no longer updated
                        t = self.problems[c].T
                        cut = True
                        while cut and t>0:
                            if (self.problems[c].trajectories[(submaster, t)] != self.problems[c].trajectories[(submaster, t - 1)]
                            or (submaster, t) in self.com_in or (submaster, t) in self.com_out):
                                cut = False
                            if cut:
                                t -= 1
                        self.start_time[c_asc] = t

        #Re-construct communication dictionaries with new starting times
        self.com_out = {}
        self.com_in = {}
        self.conn = {}
        for c in self.priority_list:
            if c in self.problems:
                for t, (b, br), (v1, v2) in product(range(self.problems[c].T+1), enumerate(self.problems[c].min_src_snk), self.problems[c].graph.conn_edges()):
                    if self.problems[c].solution['x'][self.problems[c].get_fbar_idx(b, v1, v2, t)] > 0.5:
                        for r in self.problems[c].graph.agents:
                            if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v1, t)] > 0.5:
                                r_out = r
                            if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v2, t)] > 0.5:
                                r_in = r
                        self.com_out[r_out,t+self.start_time[c]] = (r_in, v2, br)
                        self.com_in[r_in,t+self.start_time[c]] = (r_out, v1, br)
                        self.conn[(br, v1, v2, t+self.start_time[c])] = 1

                for t, (v1, v2) in product(range(self.problems[c].T+1), self.problems[c].graph.conn_edges()):
                    if self.problems[c].solution['x'][self.problems[c].get_mbar_idx(v1, v2, t)] > 0.5:
                        for r in self.problems[c].graph.agents:
                            if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v1, t)] > 0.5:
                                r_out = r
                            if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v2, t)] > 0.5:
                                r_in = r
                        self.com_out[r_out,t+self.start_time[c]] = (r_in, v2, self.problems[c].master)
                        self.com_in[r_in,t+self.start_time[c]] = (r_out, v1, self.problems[c].master)
                        self.conn[(self.problems[c].master, v1, v2, t+self.start_time[c])] = 1


        #Construct trajectories
        self.T = 0
        for r, v in self.graph.agents.items():
            self.trajectories[(r,0)] = v
        for c in self.priority_list:
            if c in self.problems:
                for r, v, t in product(self.agent_clusters[c], self.problems[c].graph.nodes, range(self.problems[c].T + 1)):
                    if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v, t)] > 0.5:
                        self.trajectories[(r,t + self.start_time[c])] = v
                        if t + self.start_time[c] > self.T:
                            self.T = t + self.start_time[c]

        #Fill out trajectories
        for r, v, t in product(self.graph.agents, self.graph.nodes, range(self.T + 1)):
            if (r,t) not in self.trajectories:
                self.trajectories[(r,t)] = self.trajectories[(r,t-1)]

    def augment_solutions_reversed(self):

        #Construct communication dictionaries
        for c in self.priority_list:
            if c in self.problems:
                for t, (b, br), (v1, v2) in product(range(self.problems[c].T+1), enumerate(self.problems[c].min_src_snk), self.problems[c].graph.conn_edges()):
                    if self.problems[c].solution['x'][self.problems[c].get_fbar_idx(b, v1, v2, t)] > 0.5:
                        for r in self.problems[c].graph.agents:
                            if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v1, t)] > 0.5:
                                r_out = r
                            if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v2, t)] > 0.5:
                                r_in = r
                        self.com_out[r_out,t] = (r_in, v2, br)
                        self.com_in[r_in,t] = (r_out, v1, br)
                        self.conn[(br, v1, v2, t)] = 1

                for t, (v1, v2) in product(range(self.problems[c].T+1), self.problems[c].graph.conn_edges()):
                    if self.problems[c].solution['x'][self.problems[c].get_mbar_idx(v1, v2, t)] > 0.5:
                        for r in self.problems[c].graph.agents:
                            if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v1, t)] > 0.5:
                                r_out = r
                            if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v2, t)] > 0.5:
                                r_in = r
                        self.com_out[r_out,t] = (r_in, v2, self.problems[c].master)
                        self.com_in[r_in,t] = (r_out, v1, self.problems[c].master)
                        self.conn[(self.problems[c].master, v1, v2, t)] = 1

        #Find end time for each cluster
        min_time = None
        for c in self.priority_list:
            if min_time == None:
                min_time = self.problems[c].T
            if c in self.problems:
                if self.problems[c].master == self.master:
                    self.end_time[c] = self.problems[c].T
                if c in self.hier_ascend:
                    for c_asc in self.hier_ascend[c]:
                        submaster = self.submasters[c_asc]
                        #find time when submaster in upper subgraph no longer updated
                        t = self.problems[c].T
                        cut = True
                        while cut and t>0:
                            if (self.problems[c].trajectories[(submaster, t)] != self.problems[c].trajectories[(submaster, t - 1)]
                            or (submaster, t) in self.com_in or (submaster, t) in self.com_out):
                                cut = False
                            if cut:
                                t -= 1
                        self.end_time[c_asc] = t
                        if t < min_time:
                            min_time = t


        #Re-construct communication dictionaries with new starting times
        self.com_out = {}
        self.com_in = {}
        self.conn = {}
        for c in self.priority_list:
            if c in self.problems:
                for t, (b, br), (v1, v2) in product(range(self.problems[c].T+1), enumerate(self.problems[c].min_src_snk), self.problems[c].graph.conn_edges()):
                    if self.problems[c].solution['x'][self.problems[c].get_fbar_idx(b, v1, v2, t)] > 0.5:
                        for r in self.problems[c].graph.agents:
                            if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v1, t)] > 0.5:
                                r_out = r
                            if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v2, t)] > 0.5:
                                r_in = r
                        self.com_out[r_out,self.end_time[c] - (self.problems[c].T - t)] = (r_in, v2, br)
                        self.com_in[r_in,self.end_time[c] - (self.problems[c].T - t)] = (r_out, v1, br)
                        self.conn[(br, v1, v2, self.end_time[c] - (self.problems[c].T - t))] = 1

                for t, (v1, v2) in product(range(self.problems[c].T+1), self.problems[c].graph.conn_edges()):
                    if self.problems[c].solution['x'][self.problems[c].get_mbar_idx(v1, v2, t)] > 0.5:
                        for r in self.problems[c].graph.agents:
                            if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v1, t)] > 0.5:
                                r_out = r
                            if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v2, t)] > 0.5:
                                r_in = r
                        self.com_out[r_out,self.end_time[c] - (self.problems[c].T - t)] = (r_in, v2, self.problems[c].master)
                        self.com_in[r_in,self.end_time[c] - (self.problems[c].T - t)] = (r_out, v1, self.problems[c].master)
                        self.conn[(self.problems[c].master, v1, v2, self.end_time[c] - (self.problems[c].T - t))] = 1

        #Construct trajectories
        self.T = 0
        for r, v in self.graph.agents.items():
            self.trajectories[(r,0)] = v
        for c in self.priority_list:
            if c in self.problems:
                for r, v, t in product(self.agent_clusters[c], self.problems[c].graph.nodes, range(self.problems[c].T + 1)):
                    if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v, t)] > 0.5:
                        self.trajectories[(r, self.end_time[c] - (self.problems[c].T - t))] = v
                        if self.end_time[c] - (self.problems[c].T - t) > self.T:
                            self.T = self.end_time[c] - (self.problems[c].T - t)

        #Fill out trajectories
        for r, v, t in product(self.graph.agents, self.graph.nodes, range(self.T + 1)):
            if (r,t) not in self.trajectories:
                self.trajectories[(r,t)] = self.trajectories[(r,t-1)]

    def solve_to_frontier_problem(self):

        self.create_subgraphs()

        active_subgraphs = self.active_subgraphs()

        for c in active_subgraphs:

            #Setup connectivity problem
            cp = ConnectivityProblem()

            #Master is submaster of cluster
            cp.master = self.submasters[c]

            #Setup agents and static agents in subgraph
            agents = {}
            static_agents = []
            sinks = []
            for r in self.agent_clusters[c]:
                agents[r] = self.graph.agents[r]
                if r in self.static_agents:
                    static_agents.append(r)
            if c in self.hier_ascend:
                for C in self.hier_ascend[c]:
                    agents[self.submasters[C]] = self.graph.agents[self.submasters[C]]
                    static_agents.append(self.submasters[C])
                    if C in active_subgraphs:
                        sinks.append(self.submasters[C])

            cp.static_agents = static_agents

            g = deepcopy(self.graph)
            nodes_not_in_subset = [v for v in self.graph.nodes if v not in self.subgraphs[c]]
            g.remove_nodes_from(nodes_not_in_subset)
            g.init_agents(agents)
            cp.graph = g

            #Source is submaster
            cp.src = [self.submasters[c]]
            #Sinks are submaster in active higger ranked subgraphs
            cp.snk = sinks

            cp.diameter_solve_flow(master = True, connectivity = True, optimal = True)
            self.problems[c] = cp

        self.augment_solutions()

    def solve_to_base_problem(self):

        self.create_subgraphs()

        active_subgraphs = self.active_subgraphs()

        for c in active_subgraphs:

            #Setup connectivity problem
            cp = ConnectivityProblem()

            #Master is submaster of cluster
            cp.master = self.submasters[c]

            #Setup agents and static agents in subgraph
            agents = {}
            static_agents = []
            sources = []
            for r in self.agent_clusters[c]:
                agents[r] = self.graph.agents[r]
                if r in self.static_agents:
                    static_agents.append(r)
                if self.graph.nodes[self.graph.agents[r]]['frontiers'] != 0:
                    sources.append(r)
            if c in self.hier_ascend:
                for C in self.hier_ascend[c]:
                    agents[self.submasters[C]] = self.graph.agents[self.submasters[C]]
                    static_agents.append(self.submasters[C])
                    if C in active_subgraphs:
                        sources.append(self.submasters[C])
            sources = list(set(sources))

            cp.static_agents = static_agents

            g = deepcopy(self.graph)
            nodes_not_in_subset = [v for v in self.graph.nodes if v not in self.subgraphs[c]]
            g.remove_nodes_from(nodes_not_in_subset)
            g.init_agents(agents)
            cp.graph = g

            cp.final_position = {r: v for r,v in self.graph.agents.items() if r == self.submasters[c]}

            #Sources are submaster in active higger ranked subgraphs
            cp.src = sources
            #Submaster is sink
            cp.snk = [self.submasters[c]]

            cp.diameter_solve_flow(master = False, connectivity = True, optimal = True, frontier_reward = False)
            self.problems[c] = cp

        self.augment_solutions_reversed()

    #===ANIMATE=====================================================================

    def animate_solution(self, ANIM_STEP=30, filename='cluster_animation.mp4', labels=False):

        # Initiate plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.axis('off')

        #Setup position dictionary for node positions
        dict_pos = {n: (self.graph.nodes[n]['x'], self.graph.nodes[n]['y']) for n in self.graph}

        # Build dictionary robot,time -> position
        traj_x = {(r,t): np.array([self.graph.nodes[v]['x'], self.graph.nodes[v]['y']])
                  for (r,t), v in self.trajectories.items()}

        # FIXED STUFF
        cluster_colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(self.subgraphs)))
        cluster_color_dict = {c : cluster_colors[i] for i, c in enumerate(self.subgraphs)}

        frontier_colors = {}
        if 'frontiers' in self.graph.nodes[0]:
            frontiers = [v for v in self.graph if self.graph.nodes[v]['frontiers'] != 0]
            for v in frontiers:
                for c in self.priority_list:
                    if v in self.subgraphs[c]:
                        frontier_colors[v] = cluster_color_dict[c]
            fcolors = []
            for v in frontiers:
                fcolors.append(frontier_colors[v])
        else:
            frontiers = []
        nx.draw_networkx_nodes(self.graph, dict_pos, ax=ax, nodelist = frontiers,
                               node_shape = "D", node_color = fcolors, edgecolors='black',
                               linewidths=1.0, alpha=0.5)

        nodes = []
        node_colors = {}
        for c in self.priority_list:
            for v in self.subgraphs[c]:
                if v not in frontiers:
                    nodes.append(v)
                    node_colors[v] = cluster_color_dict[c]
        colors = []
        for v in nodes:
            colors.append(node_colors[v])

        nx.draw_networkx_nodes(self.graph, dict_pos, ax=ax, nodelist = nodes,
                               node_color=colors, edgecolors='black', linewidths=1.0, alpha=0.5)
        nx.draw_networkx_edges(self.graph, dict_pos, ax=ax, edgelist=list(self.graph.tran_edges()),
                               connectionstyle='arc', edge_color='black')

        if labels:
            nx.draw_networkx_labels(self.graph, dict_pos)

        # VARIABLE STUFF
        # connectivity edges
        coll_cedge = nx.draw_networkx_edges(self.graph, dict_pos, ax=ax, edgelist=list(self.graph.conn_edges()),
                                            edge_color='black')
        if coll_cedge is not None:
            for cedge in coll_cedge:
                cedge.set_connectionstyle("arc3,rad=0.25")
                cedge.set_linestyle('dashed')

        # robot nodes
        pos = np.array([traj_x[(r, 0)] for r in self.graph.agents])
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.graph.agents)))
        coll_rpos = ax.scatter(pos[:,0], pos[:,1], s=140, marker='o',
                               c=colors, zorder=5, alpha=0.7,
                               linewidths=2, edgecolors='black')

        # robot labels
        coll_text = [ax.text(pos[i,0], pos[i,1], str(r),
                             horizontalalignment='center',
                             verticalalignment='center',
                             zorder=10, size=8, color='k',
                             family='sans-serif', weight='bold', alpha=1.0)
                     for i, r in enumerate(self.graph.agents)]

        def animate(i):
            t = int(i / ANIM_STEP)
            anim_idx = i % ANIM_STEP
            alpha = anim_idx / ANIM_STEP

            # Update connectivity edge colors if there is flow information
            for i, (v1, v2) in enumerate(self.graph.conn_edges()):
                coll_cedge[i].set_color('black')
                col_list = [colors[b_r] for b, b_r in enumerate(self.graph.agents)
                            if (b_r, v1, v2, t) in self.conn]
                if len(col_list):
                    coll_cedge[i].set_color(col_list[int(10 * alpha) % len(col_list)])

            # Update robot node and label positions
            pos = (1-alpha) * np.array([traj_x[(r, min(self.T, t))] for r in self.graph.agents]) \
                  + alpha * np.array([traj_x[(r, min(self.T, t+1))] for r in self.graph.agents])

            coll_rpos.set_offsets(pos)
            for i in range(len(self.graph.agents)):
                coll_text[i].set_x(pos[i, 0])
                coll_text[i].set_y(pos[i, 1])

        ani = animation.FuncAnimation(fig, animate, range((self.T+2) * ANIM_STEP), blit=False)

        writer = animation.writers['ffmpeg'](fps = 0.5*ANIM_STEP)
        ani.save(filename, writer=writer,dpi=100)
