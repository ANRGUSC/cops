import networkx as nx
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
from networkx.algorithms.centrality import betweenness_centrality
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
        self.max_problem_size = 1500

        #Clusters
        self.agent_clusters = None
        self.child_clusters = None
        self.parent_clusters = None
        self.subgraphs = None
        self.submasters = None
        self.subsinks = None

        #Problems/Solutions
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

    def create_dynamic_agent_graph(self):

        g = nx.Graph()

        dynamic_agent_nodes = set(v for r, v in self.graph.agents.items() if r not in self.static_agents)
        agents = [(tuple([r for r in self.graph.agents if self.graph.agents[r] == v and r not in self.static_agents]), v) for v in dynamic_agent_nodes]
        for r,v in agents:
            g.add_node(r)

        for (r1,v1), (r2,v2) in product(agents, agents):
            if r1 == r2:
                add_path = False
            else:
                add_path = True
                shortest_path = nx.shortest_path(self.graph_tran, source=v1, target=v2, weight='weight')
                for v in shortest_path:
                    if v in dynamic_agent_nodes and v != v1 and v != v2:
                        add_path = False
            if add_path:
                w = nx.shortest_path_length(self.graph_tran, source=v1, target=v2, weight='weight')
                g.add_edge(r1, r2, weight=w)

        #add small weight to every edge to prevent divide by zero. (w += 0.1 -> agents in same node has 10 similarity)
        for edge in g.edges:
            g.edges[edge]['weight'] += 0.1

        #inverting edge weights as spectral_clustering use them as similarity measure
        for edge in g.edges:
            g.edges[edge]['weight'] = 1/g.edges[edge]['weight']
        return g

    #===CLUSTER FUNCTIONS=======================================================

    def problem_size(self):

        cluster_size = {}
        for c in self.subgraphs:

            R  = len(set(self.agent_clusters[c]) - set(self.static_agents))
            V = len(self.subgraphs[c])
            Rp = dynamic_agent_nodes = len(set(v for r, v in self.graph.agents.items()
                                               if r in self.agent_clusters[c]))

            g = deepcopy(self.graph)
            nodes_not_in_subset = [v for v in self.graph.nodes if v not in self.subgraphs[c]]
            g.remove_nodes_from(nodes_not_in_subset)

            E = g.number_of_edges()/2
            D = nx.diameter(g)

            T = int(max(D/2, D - Rp))

            size4 = R * E * T
            print("Calculating size with R={}, E={}, D={}, Rp={}, size={}".format(R, E, D, Rp, size4 ))

            cluster_size[c] = size4

        return cluster_size

    def spectral_clustering(self, dynamic_agent_graph):

        # Adjacency matrix
        adj = nx.to_numpy_matrix(dynamic_agent_graph)

        # Cluster (uses weights as similarity measure)
        sc = SpectralClustering(self.k, affinity='precomputed')
        sc.fit(adj)

        #construct dynamic agent clusters
        self.agent_clusters = {}
        c_group = []

        dynamic_agents = [r for r in self.graph.agents if r not in self.static_agents]
        for c in range(self.k):
            for i, r_list in enumerate(dynamic_agent_graph):
                if sc.labels_[i] == c:
                    c_group += r_list
            self.agent_clusters['cluster' + str(c)] = c_group
            c_group = []

        dynamic_agent_nodes = set(v for r, v in self.graph.agents.items() if r not in self.static_agents)
        #add static agents to nearest cluster
        for (r,v) in self.graph.agents.items():
            added = False
            if r in self.static_agents:
                start_node = nx.multi_source_dijkstra(self.graph_tran, sources = dynamic_agent_nodes, target = v )[1][0]
                for c in self.agent_clusters:
                    for rc in self.agent_clusters[c]:
                        if start_node == self.graph.agents[rc] and not added:
                            self.agent_clusters[c].append(r)
                            added = True

    def create_clusters(self):

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

        # check if master cluster can activate other clusters straight away
        master_cluster_nodes = []
        for c, r_list in self.agent_clusters.items():
            if self.master in r_list:
                for r in r_list:
                    master_cluster_nodes.append(self.graph.agents[r])
        for v in master_cluster_nodes:
            cv_list = self.check(v, active_clusters[0], clusters, active_clusters)
            while len(cv_list) > 0:
                v0, c0, v1, c1 = cv_list.pop(0)
                active_clusters.append(c1)
                child_clusters[c0].append((c1, v1))
                parent_clusters[c1].append((c0, v0))
                cv_list += self.check(v1, c1, clusters, active_clusters)

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
                    return self.create_clusters()

            active_nodes = []
            for c in active_clusters:
                for v in clusters[c]:
                    active_nodes.append(v)
            # find neighbors as candidate new nodes
            neighbors = set(n for v in self.graph.nodes
                         for n in nx.neighbors(self.graph_tran, v)
                         if n in free_nodes and v in active_nodes)

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


            # check if active cluster can activate other clusters directly
            for c in active_clusters:
                cluster_nodes = []
                for v in clusters[c]:
                    cluster_nodes.append(v)
                for v in cluster_nodes:
                    cv_list = self.check(v, c, clusters, active_clusters)
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

    def create_sub_dict(self):

        # create dictionaries mappeing clusters to sub-master/sub-sinks
        self.subsinks = {}
        for c in self.subgraphs:
            self.subsinks[c] = []
            if len(self.parent_clusters[c]) == 0:
                self.submasters = {c: self.master}
                active = [c]

        while len(active)>0:
            cluster = active.pop(0)
            for child in self.child_clusters[cluster]:
                for r in self.graph.agents:
                    if self.graph.agents[r] == child[1]:
                        if child[0] not in self.submasters:
                            self.submasters[child[0]] = r
                        if r not in self.subsinks[cluster]:
                            self.subsinks[cluster].append(r)
                active.append(child[0])
        self.subsinks = {c : [r for r in self.subsinks[c]] for c in self.subsinks
                        if len(self.subsinks[c])>0}

    def create_subgraphs(self):

        print("Entering create_subgraphs")
        small_problem_size = False

        # create transition-only graph
        self.create_graph_tran()

        dynamic_agent_graph = self.create_dynamic_agent_graph()

        if self.k == None:
            self.k = 1
            self.agent_clusters = {'cluster0': [r for r in self.graph.agents]}
            self.subgraphs, self.child_clusters, self.parent_clusters = self.create_clusters()
            self.create_sub_dict()

            small_problem_size = max(self.problem_size().values()) < self.max_problem_size

            print("Initial problem size is {}".format(self.problem_size()))

            # Strategy 1: cluster
            while not small_problem_size and len(dynamic_agent_graph) > 1:
                self.k += 1

                print('Solving create_subgraphs for k={}'.format(self.k))
                #detect agent clusters
                self.spectral_clustering(dynamic_agent_graph)
                #dictionary mapping cluster to nodes
                self.subgraphs, self.child_clusters, self.parent_clusters = self.create_clusters()
                #dictionary mapping cluster to submaster, subsinks
                self.create_sub_dict()

                small_problem_size = max(self.problem_size().values()) < self.max_problem_size

            # Strategy 2: deactivate agents problem
            while not small_problem_size:
                print("Trying strategy 2: Making robots static")
                #find agent populated nodes and corresponding agents
                cluster_size = self.problem_size()
                for c, val in cluster_size.items():
                    if val >= self.max_problem_size:
                        population = {v: [r for r in self.agent_clusters[c]
                                        if r not in self.static_agents
                                        and self.graph.agents[r]==v]
                                        for v in self.subgraphs[c]}
                        max_pop = max(population.values(), key = len)
                        for r in max_pop:
                            if r != self.master:
                                self.static_agents.append(r)
                                break

                print('static_agents:', self.static_agents)
                small_problem_size = max(self.problem_size().values()) < self.max_problem_size

            print("Finished create_subgraphs with clusters", self.subgraphs, "and agent clusters", self.agent_clusters)
        else:
            #detect agent clusters
            self.spectral_clustering()

            #dictionary mapping cluster to nodes
            self.subgraphs, self.child_clusters, self.parent_clusters = self.create_clusters()

            #dictionary mapping cluster to submaster, subsinks
            self.create_sub_dict()

    def frontiers_in_cluster(self, c):
        cluster_frontiers = []
        for v in self.graph.nodes:
            if self.graph.nodes[v]['frontiers'] != 0 and v in self.subgraphs[c] and v != self.subsinks[c]:
                cluster_frontiers.append(v)
        return cluster_frontiers

    def active_clusters(self):
        active_clusters = []
        #set clusters with frontiers as active
        for c in self.subgraphs:
            active = False
            for v in self.graph.nodes:
                if self.graph.node[v]['frontiers'] != 0 and v in self.subgraphs[c]:
                    active = True
            if active:
                active_clusters.append(c)
        #set children of active clusters to active
        for c in active_clusters:
            children = self.find_all_children_clusters(c)
            for child in children:
                if child not in active_clusters:
                    active_clusters.append(child)
        return active_clusters

    def find_all_children_clusters(self, c):
        children = []
        active = [c]
        while len(active)>0:
            cluster = active.pop(0)
            children.append(cluster)
            if cluster in self.parent_clusters:
                for child in self.parent_clusters[cluster]:
                    active.append(child[0])
                    children.append(child[0])
        children.remove(c)
        return children

    #===SOLVE FUNCTIONS=========================================================

    def augment_solutions(self):

        #Construct communication dictionaries
        for c in self.problems:
            for t, (v1, v2) in product(range(self.problems[c].T+1), self.problems[c].graph.conn_edges()):

                # add communication between agents
                for (b, br) in enumerate(self.problems[c].min_src_snk)
                    if self.problems[c].solution['x'][self.problems[c].get_fbar_idx(b, v1, v2, t)] > 0.5:
                        for r_in, r_out in product(self.problems[c].graph.agents, self.problems[c].graph.agents):
                            self.com_out[r_out,t] = []
                            self.com_in[r_in,t] = []
                            if (self.problems[c].solution['x'][self.problems[c].get_z_idx(r_in, v1, t)] > 0.5
                                and self.problems[c].solution['x'][self.problems[c].get_z_idx(r_out, v2, t)] > 0.5):
                                    self.com_out[r_out,t] += (r_in, v2, br)
                                    self.com_in[r_in,t] += (r_out, v1, br)
                                    self.conn[(br, v1, v2, t)] = 1

                # add master communication between agents
                if self.problems[c].solution['x'][self.problems[c].get_mbar_idx(v1, v2, t)] > 0.5:
                    for r_in, r_out in product(self.problems[c].graph.agents, self.problems[c].graph.agents):
                        if (self.problems[c].solution['x'][self.problems[c].get_z_idx(r_in, v1, t)] > 0.5
                            and self.problems[c].solution['x'][self.problems[c].get_z_idx(r_out, v2, t)] > 0.5):
                                self.com_out[r_out,t] += (r_in, v2, self.problems[c].master)
                                self.com_in[r_in,t] += (r_out, v1, self.problems[c].master)
                                self.conn[(self.problems[c].master, v1, v2, t)] = 1



        #Find start time for each cluster
        for c in self.problems:
            if self.problems[c].master == self.master:
                self.start_time[c] = 0
            for child in self.child_clusters[c]:
                submaster = self.submasters[child[0]]
                #find time when submaster in child graph is no longer updated
                t = self.problems[c].T
                cut = True
                while cut and t>0:
                    if (self.problems[c].trajectories[(submaster, t)] != self.problems[c].trajectories[(submaster, t - 1)]:
                            cut = False
                    else:
                        for com_in in self.com_in[(submaster, t)]
                            if com_in[2] in self.problems[c].agents:
                                    cut = False
                        for com_out in self.com_out[(submaster, t)]
                            if com_out[2] in self.problems[c].agents:
                                    cut = False
                    if cut:
                        t -= 1
                self.start_time[child[0]] = t


        #Re-construct communication dictionaries with new starting times
        self.com_out = {}
        self.com_in = {}
        self.conn = {}
        for c in self.subgraphs:
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
        for c in self.subgraphs:
            if self.master in self.agent_clusters[c]:
                active = [c]        # construct active in parent-first order
        while len(active)>0:
            c = active.pop(0)
            print(c)
            if c in self.child_clusters:
                children = [child for child in self.child_clusters[c]]
                for child in children:
                    active.append(child[0])
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
        for c in self.subgraphs:
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
        for c in self.problems:
            if min_time == None:
                min_time = self.problems[c].T
            if c in self.problems:
                if self.problems[c].master == self.master:
                    self.end_time[c] = self.problems[c].T
                for child in self.child_clusters[c]:
                    submaster = self.submasters[child[0]]
                    #find time when submaster in upper subgraph no longer updated
                    t = self.problems[c].T
                    cut = True
                    while cut and t>0:
                        if (self.problems[c].trajectories[(submaster, t)] != self.problems[c].trajectories[(submaster, t - 1)]
                        or (submaster, t) in self.com_in or (submaster, t) in self.com_out):
                            cut = False
                        if cut:
                            t -= 1
                    self.end_time[child[0]] = t
                    if t < min_time:
                        min_time = t


        #Re-construct communication dictionaries with new starting times
        self.com_out = {}
        self.com_in = {}
        self.conn = {}
        for c in self.subgraphs:
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
        for c in self.subgraphs:
            if self.master in self.agent_clusters[c]:
                active = [c]        # construct active in parent-first order
        while len(active)>0:
            c = active.pop(0)
            if c in self.child_clusters:
                children = [child for child in self.child_clusters[c]]
                for child in children:
                    active.append(child[0])
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

        active_subgraphs = self.active_clusters()

        for c in active_subgraphs:

            #Setup connectivity problem
            cp = ConnectivityProblem()

            #Master is submaster of cluster
            cp.master = self.submasters[c]

            #Setup agents and static agents in subgraph
            agents = {}
            static_agents = []
            sinks = []
            additional_nodes = []

            for r in self.agent_clusters[c]:
                agents[r] = self.graph.agents[r]
                if r in self.static_agents:
                    static_agents.append(r)

            for C in self.child_clusters[c]:
                agents[self.submasters[C[0]]] = C[1]
                static_agents.append(self.submasters[C[0]])
                additional_nodes.append(C[1])
                if C[0] in active_subgraphs:
                    sinks.append(self.submasters[C[0]])

            cp.static_agents = static_agents

            g = deepcopy(self.graph)
            g.remove_nodes_from(set(self.graph.nodes) - set(self.subgraphs[c]) - set(additional_nodes))
            g.init_agents(agents)
            cp.graph = g

            cp.reward_dict = betweenness_centrality(g)
            norm = max(cp.reward_dict.values())
            if norm == 0:
                norm = 1
            cp.reward_dict = {v: 10*val/norm for v, val in cp.reward_dict.items()}

            #Source is submaster
            cp.src = [self.submasters[c]]
            #Sinks are submaster in active higger ranked subgraphs
            cp.snk = sinks

            cp.diameter_solve_flow(master = True, connectivity = True, optimal = True)
            self.problems[c] = cp

        self.augment_solutions()

    def solve_to_base_problem(self):

        self.create_subgraphs()

        active_subgraphs = self.active_clusters()

        for c in active_subgraphs:

            #Setup connectivity problem
            cp = ConnectivityProblem()

            #Master is submaster of cluster
            cp.master = self.submasters[c]

            #Setup agents and static agents in subgraph
            agents = {}
            static_agents = []
            sources = []
            additional_nodes = []

            for r in self.agent_clusters[c]:
                agents[r] = self.graph.agents[r]
                if r in self.static_agents:
                    static_agents.append(r)
                if self.graph.nodes[self.graph.agents[r]]['frontiers'] != 0:
                    sources.append(r)

            for C in self.child_clusters[c]:
                agents[self.submasters[C[0]]] = C[1]
                static_agents.append(self.submasters[C[0]])
                additional_nodes.append(C[1])
                if C[0] in active_subgraphs:
                    sources.append(self.submasters[C[0]])
            sources = list(set(sources))

            cp.static_agents = static_agents
            g = deepcopy(self.graph)
            g.remove_nodes_from(set(self.graph.nodes) - set(self.subgraphs[c]) - set(additional_nodes))
            g.init_agents(agents)
            cp.graph = g

            cp.reward_dict = betweenness_centrality(g)
            norm = max(cp.reward_dict.values())
            if norm == 0:
                norm = 1
            cp.reward_dict = {v: 10*val/norm for v, val in cp.reward_dict.items()}

            cp.final_position = {r: v for r,v in self.graph.agents.items() if r == self.submasters[c]}

            #Sources are submaster in active higger ranked subgraphs
            cp.src = sources
            #Submaster is sink
            cp.snk = [self.submasters[c]]

            cp.diameter_solve_flow(master = False, connectivity = True, optimal = True, frontier_reward = False)
            self.problems[c] = cp


        self.augment_solutions_reversed()

    #===ANIMATE=====================================================================

    def animate_solution(self, ANIM_STEP=30, filename='animation.mp4', labels=False):

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
                for c in self.subgraphs:
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
        for c in self.subgraphs:
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
