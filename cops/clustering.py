import networkx as nx
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
from networkx.algorithms.centrality import betweenness_centrality
import time
from copy import deepcopy
from itertools import product
from sklearn.cluster import SpectralClustering
from colorama import Fore, Style

from cops.problem import *

class ClusterProblem(object):

    def __init__(self):
        # Problem definition
        self.original_graph = None
        self.graph = None                   #Graph
        self.graph_tran = None              #Transition-only graph
        self.k = None                       #number of clusters
        self.master = None
        self.static_agents = None
        self.eagents = None
        self.T = None
        self.max_problem_size = 4000
        self.to_frontier_problem = None
        self.max_centrality_reward = 20
        self.evac_reward = 100

        #Clusters
        self.cluster_builup = None
        self.agent_clusters = None
        self.child_clusters = None
        self.parent_clusters = None
        self.subgraphs = None
        self.submasters = None
        self.subsinks = None
        self.active_agents = None
        self.active_subgraphs = None

        #Graph
        self.node_children_dict = None
        self.node_parent_dict = None

        #Problems/Solutions
        self.problems = {}
        self.evac = {}
        self.traj = {}
        self.conn = {}

    #=== HELPER FUNCTIONS======================================================

    @property
    def master_cluster(self):
        for c in self.agent_clusters:
            if self.master in self.agent_clusters[c]:
                return c
        print("Warning: no master cluster found")
        return None

    def parent_first_iter(self):
        '''iterate over clusters s.t. parents come before children'''
        active = [self.master_cluster]
        while len(active) > 0:
            c = active.pop(0)
            if c in self.child_clusters:
                for child in self.child_clusters[c]:
                    active.append(child[0])
            yield c

    def children_first_iter(self):
        '''iterate over clusters s.t. children come before parents'''
        for c in reversed(list(self.parent_first_iter())):
            yield c

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

    def create_node_children_parent_dict(self):

        length_to_master = nx.shortest_path_length(self.graph_tran, source=None, target=self.graph.agents[self.master], weight=None)

        self.node_children_dict = {v: [] for v in self.graph.nodes}
        self.node_parent_dict = {v: [] for v in self.graph.nodes}

        for v in self.graph.nodes:
            nbrs = self.graph_tran.neighbors(v)
            for nbr in nbrs:
                if length_to_master[nbr]>length_to_master[v]:
                    self.node_children_dict[v].append(nbr)
                    self.node_parent_dict[nbr].append(v)

    def find_dead_nodes(self):

        if self.graph_tran is None:
            self.create_graph_tran()

        #create node children dict
        if self.node_children_dict == None or self.node_parent_dict == None:
            self.create_node_children_parent_dict()

        #revive
        for v in self.graph.nodes:
            self.graph.nodes[v]['dead'] = False

        # kill dead end nodes
        for v in self.graph.nodes:
            if (len(self.node_children_dict[v]) == 0
            and self.graph.nodes[v]['frontiers'] == 0
            and not self.agent_in_node(v)):
                self.graph.nodes[v]['dead'] = True

        # kill nodes with only dead children and no agents in node
        for v in self.node_children_first_iter():
            dead = True
            # check if node is frontier
            if self.graph.nodes[v]['frontiers'] != 0:
                dead = False

            # check if all children are dead
            for child in self.node_children_dict[v]:
                if self.graph.nodes[child]['dead'] == False:
                    dead = False

            # check if exist agent in node
            if self.agent_in_node(v):
                    dead = False

            if dead:
                self.graph.nodes[v]['dead'] = True

    def node_children_first_iter(self):
        length_to_master = nx.shortest_path_length(self.graph_tran, source=None, target=self.graph.agents[self.master], weight=None)
        children_first = sorted(length_to_master.items(), key=lambda v: v[1])
        for c in reversed(children_first):
            yield c[0]

    def agent_in_node(self, v):
        exist = False
        for r in self.graph.agents:
            if self.graph.agents[r] == v:
                exist = True
        return(exist)

    def find_evac_path(self, c):

        master_nodes = set(v for r, v in self.graph.agents.items() if r == self.master)
        active_nodes = set(v for C in self.active_subgraphs for v in self.subgraphs[C]).union(master_nodes)

        evac = {}

        for r, v in self.graph.agents.items():
            if r in self.agent_clusters[c]:
                length, path = nx.multi_source_dijkstra(self.graph_tran, sources = active_nodes, target = v, weight='weight')
                path = path[::-1]
                evac[r] = path

        return evac

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

            T = int(max(D/2, D - int(Rp/2)))

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

    def inflate_clusters(self, save_buildup = False):
        '''
        Requires: self.graph
                  self.agent_clusters { 'c' : [r0, r1] ...}
                  self.master
        '''

        if save_buildup and self.cluster_builup == None:
            self.cluster_builup = []

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

        if save_buildup:
            self.cluster_builup.append((deepcopy(clusters), deepcopy(active_clusters), None, None))

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
                    return self.inflate_clusters() # recursive call

            # find closest neighbor and activate it
            neighbors = self.graph.post_tran(active_nodes) - active_nodes

            if (len(neighbors) == 0):
                break  # nothing more to do

            new_cluster, new_node, min_dist, min_path = -1, -1, 99999, None
            for c in active_clusters:
                c_neighbors = self.graph.post_tran(clusters[c])

                agent_positions = [self.graph.agents[r] for r in self.agent_clusters[c]]

                for n in neighbors & c_neighbors:

                    dist, path = nx.multi_source_dijkstra(self.graph_tran,
                                                    sources=agent_positions,
                                                    target=n)
                    if dist < min_dist:
                        min_dist, new_node, new_cluster, min_path = dist, n, c, path

            clusters[new_cluster].add(new_node)

            if save_buildup:
                self.cluster_builup.append((deepcopy(clusters), deepcopy(active_clusters), deepcopy(new_node), min_path))

        return clusters, child_clusters, parent_clusters

    def create_subgraphs(self, verbose=False, save_buildup = False):

        small_problem_size = False

        # create transition-only graph
        self.create_graph_tran()

        dynamic_agent_graph = self.create_dynamic_agent_graph()

        if self.k == None:
            self.k = 1
            self.agent_clusters = {'cluster0': [r for r in self.graph.agents]}
            self.subgraphs, self.child_clusters, self.parent_clusters = self.inflate_clusters(save_buildup)
            small_problem_size = max(self.problem_size().values()) < self.max_problem_size

            # Strategy 1: cluster
            while not small_problem_size and self.k < len(dynamic_agent_graph):
                self.k += 1

                print('Strategy 1: clustering with k={}'.format(self.k))
                #detect agent clusters
                self.spectral_clustering(dynamic_agent_graph)
                #dictionary mapping cluster to nodes
                self.subgraphs, self.child_clusters, self.parent_clusters = self.inflate_clusters(save_buildup)

                small_problem_size = max(self.problem_size(verbose=verbose).values()) < self.max_problem_size

            # Strategy 2: kill parts of graph
            while not small_problem_size:
                print("Strategy 2: Kill frontiers")

                #find frontier furthest away from master for large problems
                cluster_size = self.problem_size()
                for c, val in cluster_size.items():
                    if val >= self.max_problem_size:

                        #revive
                        for v in self.graph.nodes:
                            self.graph.nodes[v]['dead'] = False

                        #find not occupied frontiers in large subgraphs
                        frontiers = [v for v in self.subgraphs[c] if self.graph.nodes[v]['frontiers']!=0 and not self.graph.nodes[v]['dead']]
                        for r in self.agent_clusters[c]:
                            if self.graph.agents[r] in frontiers:
                                frontiers.remove(self.graph.agents[r])

                        if len(self.parent_clusters[c])>0:
                            master_node = self.parent_clusters[c][0][1]
                        else:
                            master_node = self.graph.agents[self.master]

                        max_length = None
                        max_frontier = None
                        for f in frontiers:
                            length, path = nx.single_source_dijkstra(self.graph_tran, source=master_node, target=f)
                            if max_length == None:
                                max_length = length
                                max_frontier = f
                            elif length > max_length:
                                max_length = length
                                max_frontier = f

                        self.graph.nodes[f]['dead'] = True

                        # kill max_frontier
                        self.graph.nodes[max_frontier]['dead'] = True

                        # kill nodes with only dead children and no agents in node
                        for v in self.node_children_first_iter():
                            if v in self.subgraphs[c]:

                                dead = True
                                # check if node is frontier
                                if self.graph.nodes[v]['frontiers'] != 0:
                                    dead = False

                                # check if all children are dead
                                for child in self.node_children_dict[v]:
                                    if child in self.subgraphs[c]:
                                        if self.graph.nodes[child]['dead'] == False:
                                            dead = False

                                # check if exist agent in node
                                if self.agent_in_node(v):
                                        dead = False

                                if dead:
                                    self.graph.nodes[v]['dead'] = True

                        # remove dead nodes from subgraph
                        for v in list(self.subgraphs[c]):
                            if self.graph.nodes[v]['dead']:
                                self.subgraphs[c].remove(v)

                small_problem_size = max(self.problem_size(verbose=verbose).values()) < self.max_problem_size

        else:
            #detect agent clusters
            self.spectral_clustering(dynamic_agent_graph)

            #dictionary mapping cluster to nodes
            self.subgraphs, self.child_clusters, self.parent_clusters = self.inflate_clusters(save_buildup)

        # create dictionaries mapping cluster to submaster, subsinks
        self.submasters = {self.master_cluster: self.master}
        self.subsinks = {c : [] for c in self.subgraphs}

        for c in self.parent_first_iter():
            for child, r in product(self.child_clusters[c], self.graph.agents):
                if self.graph.agents[r] == child[1]:
                    if child[0] not in self.submasters:
                        self.submasters[child[0]] = r
                    if r not in self.subsinks[c]:
                        self.subsinks[c].append(r)

    #===SOLVE FUNCTIONS=========================================================

    def frontier_clusters(self):
        #set clusters with frontiers as active
        frontier_clusters = set(c for c in self.subgraphs
                              if any(self.graph.node[v]['frontiers'] != 0
                                     for v in self.subgraphs[c]))

        for c in self.children_first_iter():
            if any(child_c in frontier_clusters for child_c,_ in self.child_clusters[c]):
               frontier_clusters.add(c)

        return frontier_clusters

    def activate_agents(self):
        self.active_agents = {}
        for c in self.problems:

            agents = set(r for r in self.agent_clusters[c]).union(self.submasters[child[0]] for child in self.child_clusters[c])

            # initial position activated nodes
            self.active_agents[c] = [r for r in self.graph.agents
                                    if (self.graph.agents[r] == self.graph.agents[self.submasters[c]])]

            # connectivity activated nodes
            for t, conn_t in self.problems[c].conn.items():
                for (v1, v2, b) in conn_t:
                    if b == 'master':    #if b is a tuple then b is a master (by the construction of conn)
                        for r in agents:
                            if self.graph.agents[r] == v2:
                                if r not in self.active_agents[c]:
                                    self.active_agents[c].append(r)
                    elif b == self.submasters[c]:
                        for r in agents:
                            if self.graph.agents[r] == v2:
                                if r not in self.active_agents[c]:
                                    self.active_agents[c].append(r)

            # transition activated nodes
            for t, tran_t in self.problems[c].tran.items():
                for (v1, v2, b) in tran_t:
                    if b == 'master':    #if b is a tuple then b is a master (by the construction of tran)
                        for r in agents:
                            if self.graph.agents[r] == v2 and r not in self.active_agents[c]:
                                    self.active_agents[c].append(r)
                    elif b == self.submasters[c]:
                        for r in agents:
                            if self.graph.agents[r] == v2 and r not in self.active_agents[c]:
                                    self.active_agents[c].append(r)

    def merge_solutions(self, order = 'forward'):

        #Find start and end times for each cluster
        fwd_start_time = {self.master_cluster : 0}
        rev_end_time = {self.master_cluster : 0}

        for c in self.parent_first_iter():
            if c not in self.problems:
                continue

            for child, _ in self.child_clusters[c]:
                submaster = self.submasters[child]
                submaster_node = self.problems[c].graph.agents[submaster]

                # first time when c has finished communicating with submaster of child
                t_cut = next( (t for t in range(self.problems[c].T, 0, -1)
                               if self.problems[c].traj[(submaster, t)]
                                  != self.problems[c].traj[(submaster, t - 1)]
                               or any(submaster_node in conn_t[0:2]
                                      for conn_t in self.problems[c].conn[t])),
                              0)

                fwd_start_time[child] = fwd_start_time[c] + t_cut
                rev_end_time[child] = rev_end_time[c] - self.problems[c].T + t_cut

        if order == 'reversed' and len(self.problems)>0:
            min_t = min(rev_end_time[c] - self.problems[c].T for c in self.problems)
            start_time = {c: rev_end_time[c] - self.problems[c].T - min_t for c in self.problems}
        else:
            start_time = fwd_start_time

        #Find maximal evac time
        max_evac_time = 0
        for c in self.evac:
            max_evac_c = max(len(path) for r, path in self.evac[c].items())
            if max_evac_c > max_evac_time:
                max_evac_time = max_evac_c

        # Cluster problem total time
        if len(self.problems)>0:
            self.T = max(max_evac_time,max(start_time[c] + self.problems[c].T for c in self.problems))
        else:
            self.T = max_evac_time

        # Trajectories for cluster problem
        self.traj = {}
        for c in self.parent_first_iter():
            if c in self.problems:
                for (r, t), v in self.problems[c].traj.items():
                    if r not in self.subsinks[c]:
                         self.traj[r, start_time[c] + t] = v

        # Trajectories for evac
        if self.evac != None:
            for c in self.evac:
                for r, path in self.evac[c].items():
                    for i, v in enumerate(path):
                        self.traj[r, i] = v


        # Communication dictionary for cluster problem
        self.conn = {t : set() for t in range(self.T+1)}
        for c in self.parent_first_iter():
            if c in self.problems:
                for t, conn_t in self.problems[c].conn.items():
                    for (v1, v2, b) in conn_t:
                        # save connectivity info with submaster as third element
                        self.conn[start_time[c] + t].add((v1, v2, b))

        # Fill out empty trajectory slots
        for r, t in product(self.graph.agents, range(self.T+1)):
            if t == 0:
                self.traj[(r,t)] = self.graph.agents[r]
            if (r,t) not in self.traj:
                self.traj[(r,t)] = self.traj[(r,t-1)]

    def solve_to_frontier_problem(self, verbose=False, soft = False, dead = False):

        self.find_dead_nodes()
        self.original_graph = self.graph
        self.graph = deepcopy(self.graph)
        dead_nodes = [v for v in self.graph.nodes if self.graph.nodes[v]['dead']]
        self.graph.remove_nodes_from(dead_nodes)

        self.create_subgraphs(verbose=verbose)

        cluster_reward = {}

        self.active_subgraphs = self.frontier_clusters()

        for c in self.children_first_iter():

            if c in self.active_subgraphs:

                #Setup connectivity problem
                cp = ConnectivityProblem()
                cp.always_src = True

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
                    sinks.append(self.submasters[C[0]])

                cp.static_agents = static_agents
                cp.eagents = self.eagents

                g = deepcopy(self.graph)
                g.remove_nodes_from(set(self.graph.nodes) - set(self.subgraphs[c]) - set(additional_nodes))
                g.init_agents(agents)
                cp.graph = g

                cp.reward_dict = betweenness_centrality(g)
                norm = max(cp.reward_dict.values())
                if norm == 0:
                    norm = 1
                cp.reward_dict = {v: 20*val/norm for v, val in cp.reward_dict.items()}

                if soft:
                    #add reward for k = 1 for at subsinks instead of hard flow contraints
                    for c_child, v_child in self.child_clusters[c]:
                        cp.reward_dict[v_child] -= cluster_reward[c_child]
                else:
                    #Source is submaster
                    cp.src = [self.submasters[c]]
                    #Sinks are submaster in active higher ranked subgraphs
                    cp.snk = sinks

                cp.diameter_solve_flow(master = True, connectivity = not soft,
                                       optimal = True, verbose = verbose)
                self.problems[c] = cp


            elif dead:

                #Setup connectivity problem
                cp = ConnectivityProblem()
                cp.always_src = True

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
                    sinks.append(self.submasters[C[0]])

                cp.static_agents = static_agents
                cp.eagents = self.eagents

                g = deepcopy(self.graph)
                g.remove_nodes_from(set(self.graph.nodes) - set(self.subgraphs[c]) - set(additional_nodes))
                g.init_agents(agents)
                cp.graph = g

                cp.reward_dict = {v: self.evac_reward for v in g.nodes}

                if soft:
                    #add reward for k = 1 for at subsinks instead of hard flow contraints
                    for c_child, v_child in self.child_clusters[c]:
                        cp.reward_dict[v_child] -= cluster_reward[c_child]

                else:
                    #Source is submaster
                    cp.src = [self.submasters[c]]
                    #Sinks are submaster in active higher ranked subgraphs
                    cp.snk = sinks

                cp.diameter_solve_flow(master = True, connectivity = not soft,
                                       optimal = True, frontier_reward = False,
                                       verbose = verbose)
                self.problems[c] = cp




            if soft:
                #find activated subgraphs
                activated_subgraphs = set()
                # connectivity activated nodes
                for t, conn_t in self.problems[c].conn.items():
                    for child in self.child_clusters[c]:
                        for (v1, v2, b) in conn_t:
                            if b == 'master' and v2 == child[1] and child[0] in self.problems:    #if b is a tuple then b is a master (by the construction of conn)
                                activated_subgraphs.add(child[0])

                # transition activated nodes
                for t, tran_t in self.problems[c].conn.items():
                    for child in self.child_clusters[c]:
                        for (v1, v2, b) in tran_t:
                            if b == 'master' and v2 == child[1] and child[0] in self.problems:    #if b is a tuple then b is a master (by the construction of conn)
                                activated_subgraphs.add(child[0])

                for child in self.child_clusters[c]:
                    if child[0] not in activated_subgraphs:
                        del self.problems[child[0]]

                cp.reward_dict = {v: self.evac_reward for v in g.nodes}

                initial_centrality_reward = sum(cp.reward_dict[self.graph.agents[r]] for r in self.agent_clusters[c])
                cluster_reward[c] = cp.solution['primal objective'] - initial_centrality_reward

        self.merge_solutions(order = 'forward')

        # find activated agents
        self.activate_agents()

    def solve_to_base_problem(self, verbose=False, dead = True):

        self.find_dead_nodes()
        self.original_graph = self.graph
        self.graph = deepcopy(self.graph)

        # if previous to_frontier_problem specified, use same clusters
        if self.to_frontier_problem != None:
            self.agent_clusters = self.to_frontier_problem.agent_clusters
            self.child_clusters = self.to_frontier_problem.child_clusters
            self.parent_clusters = self.to_frontier_problem.parent_clusters
            self.subgraphs = self.to_frontier_problem.subgraphs
            self.submasters = self.to_frontier_problem.submasters
            self.subsinks = self.to_frontier_problem.subsinks
            self.active_subgraphs = self.to_frontier_problem.active_subgraphs
            dead_nodes = [v for v in self.to_frontier_problem.graph.nodes if self.to_frontier_problem.graph.nodes[v]['dead']]
            self.graph.remove_nodes_from(dead_nodes)
        else:
            self.create_subgraphs(verbose=verbose)
            self.active_subgraphs = self.frontier_clusters()
            dead_nodes = [v for v in self.graph.nodes if self.graph.nodes[v]['dead']]
            self.graph.remove_nodes_from(dead_nodes)

        for c in self.children_first_iter():

            if c in self.active_subgraphs:

                #Setup connectivity problem
                cp = ConnectivityProblem()
                cp.always_src = True

                if self.to_frontier_problem != None:
                    #Masters are active agents in cluster
                    dead_agents = [self.submasters[child[0]] for child in self.child_clusters[c] if child[0] not in self.active_subgraphs]
                    cp.master = [r for r in self.to_frontier_problem.active_agents[c] if r not in dead_agents]
                else:
                    cp.master = [self.submasters[c]]


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
                    if C[0] in self.active_subgraphs:
                        sources.append(self.submasters[C[0]])
                sources = list(set(sources))

                cp.static_agents = static_agents
                cp.eagents = self.eagents

                g = deepcopy(self.graph)
                g.remove_nodes_from(set(self.graph.nodes) - set(self.subgraphs[c]) - set(additional_nodes))
                g.init_agents(agents)
                cp.graph = g

                cp.reward_dict = betweenness_centrality(g)
                norm = max(cp.reward_dict.values())
                if norm == 0:
                    norm = 1
                cp.reward_dict = {v: self.max_centrality_reward*val/norm for v, val in cp.reward_dict.items()}

                #force submasters to go back to initial positions for communication
                if self.to_frontier_problem != None:
                    cp.final_position = {r: v for r, v in self.to_frontier_problem.graph.agents.items() if r == self.submasters[c]}
                else:
                    cp.final_position = {r: v for r,v in self.graph.agents.items() if r == self.submasters[c]}

                #Sources are submaster in active higher ranked subgraphs
                cp.src = sources

                #Submaster is sink
                cp.snk = [self.submasters[c]]

                #force master to be
                cp.additional_constraints = [('constraint_static_master',self.submasters[c])]

                cp.diameter_solve_flow(master = True, connectivity = True,
                                       optimal = True, frontier_reward = False,
                                       verbose = verbose)

                self.problems[c] = cp

            elif dead:

                evac = self.find_evac_path(c)

                self.evac[c] = evac

        self.merge_solutions(order = 'reversed')
