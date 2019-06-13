import networkx as nx
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
from networkx.algorithms.centrality import betweenness_centrality
import time
from copy import deepcopy
from itertools import product
from sklearn.cluster import SpectralClustering
from colorama import Fore, Style

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
        self.max_problem_size = 4000

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

    def parent_first_iter(self):
        for c in self.problems:
            if self.problems[c].master == self.master:
                active =[c]
        while len(active) > 0:
            c = active.pop(0)
            if c in self.child_clusters:
                for child in self.child_clusters[c]:
                    if child[0] in self.problems:
                        active.append(child[0])
            yield c

    #===CLUSTER FUNCTIONS=======================================================

    def problem_size(self, verbose=False):

        cluster_size = {}
        for c in self.subgraphs:

            # number of robots
            R  = len(self.agent_clusters[c]) + len(self.child_clusters[c])
            # number of nodes
            V = len(self.subgraphs[c]) + len(self.child_clusters[c])
            # number of occupied nodes
            Rp = len(set(v for r, v in self.graph.agents.items()
                         if r in self.agent_clusters[c]))
            # number of transition edges
            Et = self.graph.number_of_tran_edges(self.subgraphs[c])
            # number of connectivity edges
            Ec = self.graph.number_of_conn_edges(self.subgraphs[c])
            # graph diameter
            D = nx.diameter(nx.subgraph(self.graph, self.subgraphs[c]))

            T = int(max(D/2, D - Rp))

            size = R * Et * T

            if verbose:
                print("{} size={} [R={}, V={}, Et={}, Ec={}, D={}, Rp={}]"\
                      .format(c, size, R, V, Et, Ec, D, Rp))

            cluster_size[c] = size

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
        '''
        Requires: self.graph
                  self.agent_clusters { 'c' : [r0, r1] ...}
                  self.master

        Produces: clusters, child_clusters, parent_clusters

        '''

        if self.graph_tran is None:
            self.create_graph_tran()

        # node clusters with agent positions
        clusters = {c : set(self.graph.agents[r] for r in r_list)
                    for c, r_list in self.agent_clusters.items()}

        child_clusters = {c : [] for c in self.agent_clusters.keys()}
        parent_clusters = {c : [] for c in self.agent_clusters.keys()}

        # start with master cluster active
        active_clusters = [c for c, r_list in self.agent_clusters.items()
                           if self.master in r_list]

        while True:

            # check if active cluster can activate other clusters directly
            found_new = True
            while (found_new):
                found_new = False
                for c0, c1 in product(active_clusters, clusters.keys()):
                    if c0 == c1 or c1 in active_clusters:
                        continue
                    for v0, v1 in product(clusters[c0], clusters[c1]):
                        if self.graph.has_conn_edge(v0, v1):
                            active_clusters.append(c1)
                            child_clusters[c0].append((c1, v1))
                            parent_clusters[c1].append((c0, v0))
                            found_new = True

            # all active nodes
            active_nodes = set.union(*[clusters[c] for c in active_clusters])
            free_nodes = set(self.graph.nodes) - set.union(*clusters.values())

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

            # find closest neighbor and activate it
            neighbors = self.graph.post_tran(active_nodes) - active_nodes

            if (len(neighbors) == 0):
                break  # nothing more to do

            new_cluster, new_node, min_dist = -1, -1, 99999
            for c in active_clusters:
                c_neighbors = self.graph.post_tran(clusters[c])

                agent_positions = [self.graph.agents[r] for r in self.agent_clusters[c]]

                for n in neighbors & c_neighbors:

                    dist = nx.multi_source_dijkstra(self.graph_tran,
                                                    sources=agent_positions,
                                                    target=n)[0]
                    if dist < min_dist:
                        min_dist, new_node, new_cluster = dist, n, c

            clusters[new_cluster].add(new_node)

        return clusters, child_clusters, parent_clusters

    def create_sub_dict(self):

        # create dictionaries mapping clusters to sub-master/sub-sinks
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

    def create_subgraphs(self, verbose=False):

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

            # Strategy 1: cluster
            while not small_problem_size and self.k < len(dynamic_agent_graph)/2:
                self.k += 1

                print('Strategy 1: clustering with k={}'.format(self.k))
                #detect agent clusters
                self.spectral_clustering(dynamic_agent_graph)
                #dictionary mapping cluster to nodes
                self.subgraphs, self.child_clusters, self.parent_clusters = self.create_clusters()
                #dictionary mapping cluster to submaster, subsinks
                self.create_sub_dict()

                small_problem_size = max(self.problem_size(verbose=verbose).values()) < self.max_problem_size

            # Strategy 2: deactivate agents problem
            while not small_problem_size:
                print("Strategy 2: Make robot static")

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

                small_problem_size = max(self.problem_size(verbose=verbose).values()) < self.max_problem_size

        else:
            #detect agent clusters
            self.spectral_clustering(dynamic_agent_graph)

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

    def merge_solutions(self, order = 'forward'):

        if order == 'forward':
            #Find start time for each cluster
            start_time = {}
            for c in self.problems:
                if self.problems[c].master == self.master:
                    start_time[c] = 0
                    active = [c]
            while len(active) > 0:
                c = active.pop(0)
                for child in self.child_clusters[c]:
                    if c not in self.problems:
                        continue
                    active.append(child[0])
                    #find time when submaster in child graph is no longer updated
                    submaster = self.submasters[child[0]]
                    submaster_node = self.problems[c].graph.agents[submaster]
                    t = self.problems[c].T
                    cut = True
                    while cut and t>0:
                        if (self.problems[c].trajectories[(submaster, t)]
                            != self.problems[c].trajectories[(submaster, t - 1)]):
                                cut = False
                        else:
                            for v0, v1, b in self.problems[c].conn[t]:
                                if (v0 == submaster_node or v1 == submaster_node):
                                    cut = False
                        if cut:
                            t -= 1

                    start_time[child[0]] = start_time[c] + t

        else:
            #Find start time for each cluster
            end_time = {}
            for c in self.problems:
                if self.problems[c].master == self.master:
                    end_time[c] = 0
                    active = [c]
            while len(active) > 0:
                c = active.pop(0)
                for child in self.child_clusters[c]:
                    if c not in self.problems:
                        continue
                    active.append(child[0])
                    #find time when submaster in child graph is no longer updated
                    submaster = self.submasters[child[0]]
                    submaster_node = self.problems[c].graph.agents[submaster]
                    t = self.problems[c].T
                    cut = True
                    while cut and t>0:
                        if (self.problems[c].trajectories[(submaster, t)]
                            != self.problems[c].trajectories[(submaster, t - 1)]):
                                cut = False
                        else:
                            for v0, v1, b in self.problems[c].conn[t]:
                                if (v0 == submaster_node or v1 == submaster_node):
                                    cut = False
                        if cut:
                            t -= 1
                    end_time[child[0]] = end_time[c] - self.problems[c].T + t

            start_time = {c: end_time[c] - self.problems[c].T for c in self.problems}
            start_time = {c: t - min(start_time.values()) for c, t in start_time.items()}

        # Construct communication dictionary for cluster problem
        self.conn = {start_time[c] + t: conn_t for c in self.problems for t, conn_t in self.problems[c].conn.items()}
        # Construct trajectories for cluster problem
        self.trajectories = {(r, start_time[c] + t): v for c in self.parent_first_iter() for (r, t), v in self.problems[c].trajectories.items()}

        self.T = max(start_time[c] + self.problems[c].T for c in self.problems)

        for r, t in product(self.graph.agents, range(self.T+1)):
            if t == 0:
                self.trajectories[(r,t)] = self.graph.agents[r]
            if (r,t) not in self.trajectories:
                self.trajectories[(r,t)] = self.trajectories[(r,t-1)]

    def solve_to_frontier_problem(self, verbose=False):

        self.create_subgraphs(verbose=verbose)

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

            cp.diameter_solve_flow(master = True, connectivity = True,
                                   optimal = True
                                   , verbose = verbose)
            self.problems[c] = cp

        self.merge_solutions(order = 'forward')

    def solve_to_base_problem(self, verbose=False):

        self.create_subgraphs(verbose=verbose)

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

            cp.diameter_solve_flow(master = False, connectivity = True,
                                   optimal = True, frontier_reward = False,
                                   verbose = verbose)
            self.problems[c] = cp


        self.merge_solutions(order = 'reversed')
