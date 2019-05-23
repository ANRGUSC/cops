import networkx as nx
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy
from itertools import product
from sklearn.cluster import SpectralClustering
from sklearn import metrics

from graph_connectivity.problem import *

class ClusterProblem(object):

    def __init__(self, graph, k, master, static_agents):
        # Problem definition
        self.graph = graph                   #Graph
        self.graph_tran = None               #Transition-only graph
        self.k = k                       #number of clusters
        self.master = master
        self.static_agents = static_agents
        self.T = None
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
        graph_tran = deepcopy(self.graph)
        graph_tran.remove_edges_from(list(self.graph.conn_edges()))

        self.priority_list = []
        for c in self.subgraphs:
            dist, path = nx.multi_source_dijkstra(self.graph_tran, sources = self.subgraphs[c], target=master_node)
            self.priority_list.append((c,dist))
        sorted_priority_list = sorted(self.priority_list, key=lambda x: x[1])
        self.priority_list = [c[0] for c in sorted_priority_list]

    def create_clusters(self):

        #create subgraphs corresponding to agent clusters
        self.subgraphs = {}
        for c in self.agent_clusters:
            self.subgraphs[c] = []
            for r in self.agent_clusters[c]:
                self.subgraphs[c].append(self.graph.agents[r])

        #list of nodes available to be added into a cluster
        self.free_nodes = [n for n in self.graph.nodes if n not in self.graph.agents.values()]

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

        #detect agent clusters
        self.spectral_clustering()

        #dictionary mapping cluster to nodes
        self.create_clusters()

        #dictionary mapping cluster to submaster, subsinks
        self.create_submaster_subsinks_dictionary()

        #dictionary mapping cluster to submaster, subsinks
        self.create_hierarchical_dictionary()

    def frontiers_in_cluster(self, c):
        cluster_frontiers = []
        for v in self.graph.nodes:
            if self.graph.nodes[v]['frontiers'] != 0 and v in self.subgraphs[c] and v!=self.subsinks[c]:
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

    def augment_solutions(self, order = 'ascend'):

        self.trajectories = {}

        if order == 'ascend':
            priority_list = self.priority_list
        else:
            priority_list = reversed(self.priority_list)

        t_start = 0
        for r, v in self.graph.agents.items():
            self.trajectories[(r,0)] = v

        for c in priority_list:
            for r, v, t in product(self.problems[c].graph.agents, self.problems[c].graph.nodes, range(self.problems[c].T + 1)):
                if self.problems[c].solution['x'][self.problems[c].get_z_idx(r, v, t)] > 0.5:
                    self.trajectories[(r,t + t_start)] = v
            t_start+=self.problems[c].T + 1

        for r, v, t in product(self.graph.agents, self.graph.nodes, range(t_start)):
            if (r,t) not in self.trajectories:
                self.trajectories[(r,t)] = self.trajectories[(r,t-1)]

        self.T = t_start - 1

    def solve_to_frontier_problem(self):

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

        self.augment_solutions(order = 'ascend')



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
