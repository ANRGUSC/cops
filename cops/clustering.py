import networkx as nx
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
from networkx.algorithms.centrality import betweenness_centrality
import time
from copy import deepcopy
from itertools import product
from sklearn.cluster import SpectralClustering
from colorama import Fore, Style

from cops.problem import *


@dataclass
class ClusterStructure(object):
    subgraphs: dict = None
    agent_clusters: dict = None
    child_clusters: dict = None
    parent_clusters: dict = None
    submasters: dict = None
    subsinks: dict = None


@dataclass
class ToFrontierData(object):
    initial_pos: dict = None
    clusterstructure: ClusterStructure = None
    active_clusters: list = None
    active_agents: list = None


class ClusterProblem(AbstractConnectivityProblem):
    def __init__(self):
        super(ClusterProblem, self).__init__()

        # CLUSTERING PARAMETERS
        self.k = None  # desired number of clusters
        self.max_problem_size = 4000  # max problem size

        self.max_centrality_reward = 20
        self.evac_reward = 100

    # === HELPER FUNCTIONS======================================================

    def prepare_problem(self):
        super(ClusterProblem, self).prepare_problem()

        # compute and store the following things:
        #  1. graph_tran: graph with only mobility edges
        #  2. length_to_master: distance from each robot to master node
        #  3. node_children_dict: tree based on length_to_master

        # save a graph with transition only
        self.graph_tran = deepcopy(self.graph)
        self.graph_tran.remove_edges_from(list(self.graph.conn_edges()))

        # build a tree based on distance from master
        self.length_to_master = nx.shortest_path_length(
            self.graph_tran,
            source=None,
            target=self.graph.agents[self.master],
            weight=None,
        )

        self.node_children_dict = {v: set() for v in self.graph.nodes}

        for v in self.graph.nodes:
            nbrs = self.graph_tran.neighbors(v)
            for nbr in nbrs:
                if self.length_to_master[nbr] > self.length_to_master[v]:
                    self.node_children_dict[v].add(nbr)

    def expand_dead_nodes(self, initial_dead):
        """
        traverse graph from outside to in, add nodes to dead set
        if all their children are dead
        """

        node_order = reversed(sorted(self.length_to_master.items(), key=lambda v: v[1]))

        # kill nodes with only dead children and no agents in node
        for v, _ in node_order:
            if (
                self.graph_tran.nodes[v]["frontiers"] == 0
                and v not in self.graph_tran.agents.values()
                and self.node_children_dict[v] <= initial_dead
            ):
                initial_dead.add(v)

        return initial_dead

    def find_evac_path(self, c, tofront_data):
        """evacuate a cluster by finding shortest path from each robot in cluster to origin"""

        master_nodes = set(v for r, v in self.graph.agents.items() if r == self.master)
        active_nodes = set(
            v
            for C in tofront_data.active_clusters
            for v in tofront_data.clusterstructure.subgraphs[C]
        ).union(master_nodes)

        evac = {}

        for r, v in self.graph.agents.items():
            if r in tofront_data.clusterstructure.agent_clusters[c]:
                length, path = nx.multi_source_dijkstra(
                    self.graph_tran, sources=active_nodes, target=v, weight="weight"
                )
                path = path[::-1]
                evac[r] = path

        return evac

    # ===SOLVE FUNCTIONS=========================================================

    def frontier_clusters(self, clusterstructure):
        # set clusters with frontiers as active
        frontier_clusters = set(
            c
            for c in clusterstructure.subgraphs
            if any(
                self.graph.node[v]["frontiers"] != 0
                for v in clusterstructure.subgraphs[c]
            )
        )

        for c in children_first_iter(clusterstructure.child_clusters):
            if any(
                child_c in frontier_clusters
                for child_c, _ in clusterstructure.child_clusters[c]
            ):
                frontier_clusters.add(c)

        return frontier_clusters


    def activate_agents(self, problems, clusterstructure):
        """find agents that received masterdata in previous problem"""
        active_agents = {}
        for c in problems:

            agents = set(r for r in clusterstructure.agent_clusters[c]).union(
                clusterstructure.submasters[child[0]]
                for child in clusterstructure.child_clusters[c]
            )

            # initial position activated nodes
            active_agents[c] = [
                r
                for r in self.graph.agents
                if (
                    self.graph.agents[r]
                    == self.graph.agents[clusterstructure.submasters[c]]
                )
            ]

            # connectivity activated nodes
            for t, conn_t in problems[c].conn.items():
                for (v1, v2, b) in conn_t:
                    if (
                        b == "master"
                    ):  # if b is a tuple then b is a master (by the construction of conn)
                        for r in agents:
                            if self.graph.agents[r] == v2:
                                if r not in active_agents[c]:
                                    active_agents[c].append(r)
                    elif b == clusterstructure.submasters[c]:
                        for r in agents:
                            if self.graph.agents[r] == v2:
                                if r not in active_agents[c]:
                                    active_agents[c].append(r)

            # transition activated nodes
            for t, tran_t in problems[c].tran.items():
                for (v1, v2, b) in tran_t:
                    if (
                        b == "master"
                    ):  # if b is a tuple then b is a master (by the construction of tran)
                        for r in agents:
                            if self.graph.agents[r] == v2 and r not in active_agents[c]:
                                active_agents[c].append(r)
                    elif b == clusterstructure.submasters[c]:
                        for r in agents:
                            if self.graph.agents[r] == v2 and r not in active_agents[c]:
                                active_agents[c].append(r)
        return active_agents


    def merge_solutions(self, problems, clusterstructure, evac=None, order="forward"):
        """merge and store solutions"""

        # find master cluster
        master_cluster = set(clusterstructure.child_clusters.keys())
        for children in clusterstructure.child_clusters.values():
            for child in children:
                master_cluster.remove(child[0])
        master_cluster = list(master_cluster)[0]

        # Find start and end times for each cluster
        fwd_start_time = {master_cluster: 0}
        rev_end_time = {master_cluster: 0}

        for c in parent_first_iter(clusterstructure.child_clusters):
            if c not in problems:
                continue

            for child, _ in clusterstructure.child_clusters[c]:
                submaster = clusterstructure.submasters[child]
                submaster_node = problems[c].graph.agents[submaster]

                # first time when c has finished communicating with submaster of child
                t_cut = next(
                    (
                        t
                        for t in range(problems[c].T_sol, 0, -1)
                        if problems[c].traj[(submaster, t)]
                        != problems[c].traj[(submaster, t - 1)]
                        or any(
                            submaster_node in conn_t[0:2]
                            for conn_t in problems[c].conn[t]
                        )
                    ),
                    0,
                )

                fwd_start_time[child] = fwd_start_time[c] + t_cut
                rev_end_time[child] = rev_end_time[c] - problems[c].T_sol + t_cut

        if order == "reversed" and len(problems) > 0:
            min_t = min(rev_end_time[c] - problems[c].T_sol for c in problems)
            start_time = {
                c: rev_end_time[c] - problems[c].T_sol - min_t for c in problems
            }
        else:
            start_time = fwd_start_time

        # Find maximal evac time
        max_evac_time = 0
        if evac is not None:
            for c in evac:
                max_evac_c = max(len(path) for r, path in evac[c].items())
                if max_evac_c > max_evac_time:
                    max_evac_time = max_evac_c

        # Cluster problem total time
        if len(problems) > 0:
            T_sol = max(
                max_evac_time, max(start_time[c] + problems[c].T_sol for c in problems)
            )
        else:
            T_sol = max_evac_time

        # Trajectories for cluster problem
        traj = {}
        for c in parent_first_iter(clusterstructure.child_clusters):
            if c in problems:
                for (r, t), v in problems[c].traj.items():
                    if r not in clusterstructure.subsinks[c]:
                        traj[r, start_time[c] + t] = v

        # Trajectories for evac
        if evac is not None:
            for c in evac:
                for r, path in evac[c].items():
                    for i, v in enumerate(path):
                        traj[r, i] = v

        # Communication dictionary for cluster problem
        conn = {t: set() for t in range(T_sol + 1)}
        for c in parent_first_iter(clusterstructure.child_clusters):
            if c in problems:
                for t, conn_t in problems[c].conn.items():
                    for (v1, v2, b) in conn_t:
                        # save connectivity info with submaster as third element
                        conn[start_time[c] + t].add((v1, v2, b))

        # Fill out empty trajectory slots
        for r, t in product(self.graph.agents, range(T_sol + 1)):
            if t == 0:
                traj[(r, t)] = self.graph.agents[r]
            if (r, t) not in traj:
                traj[(r, t)] = traj[(r, t - 1)]

        self.T_sol = T_sol
        self.traj = traj
        self.conn = conn


    def solve_to_frontier_problem(self, verbose=False, soft=False, dead=False):
        """solve a connectivity problem to get robots to frontiers"""

        self.prepare_problem()

        # identify and delete dead nodes
        initial_dead = set(
            [
                v
                for v in self.graph.nodes
                if len(self.node_children_dict[v]) == 0
                and self.graph.nodes[v]["frontiers"] == 0
                and v not in self.graph_tran.agents.values()
            ]
        )

        dead_nodes = self.expand_dead_nodes(initial_dead)
        print("Removing {} dead nodes".format(len(dead_nodes)))
        self.graph.remove_nodes_from(dead_nodes)

        clusterstructure = cluster(self.k, self.graph, self.graph_tran, self.master, self.static_agents, self.max_problem_size, verbose=verbose)

        problems = {}
        cluster_reward = {}

        active_clusters = self.frontier_clusters(clusterstructure)

        for c in children_first_iter(clusterstructure.child_clusters):

            if c in active_clusters or dead:

                # Setup connectivity problem
                cp = ConnectivityProblem()
                cp.always_src = True

                # Master is submaster of cluster
                cp.master = clusterstructure.submasters[c]

                # Setup agents and static agents in subgraph
                agents = {}
                static_agents = []
                sinks = []
                additional_nodes = []

                for r in clusterstructure.agent_clusters[c]:
                    agents[r] = self.graph.agents[r]
                    if r in self.static_agents:
                        static_agents.append(r)

                for C in clusterstructure.child_clusters[c]:
                    agents[clusterstructure.submasters[C[0]]] = C[1]
                    static_agents.append(clusterstructure.submasters[C[0]])
                    additional_nodes.append(C[1])
                    sinks.append(clusterstructure.submasters[C[0]])

                cp.static_agents = static_agents
                cp.eagents = [r for r in self.eagents if r in agents]
                cp.big_agents = [r for r in self.big_agents if r in agents]

                g = deepcopy(self.graph)
                g.remove_nodes_from(
                    set(self.graph.nodes)
                    - set(clusterstructure.subgraphs[c])
                    - set(additional_nodes)
                )
                g.init_agents(agents)
                cp.graph = g

                cp.reward_dict = betweenness_centrality(g)
                norm = max(cp.reward_dict.values())
                if norm == 0:
                    norm = 1

                if c in active_clusters:
                    cp.reward_dict = {
                        v: 20 * val / norm for v, val in cp.reward_dict.items()
                    }
                else:
                    cp.reward_dict = {v: self.evac_reward for v in g.nodes}

                if soft:
                    # add reward for k = 1 for at subsinks instead of hard flow contraints
                    for c_child, v_child in clusterstructure.child_clusters[c]:
                        cp.reward_dict[v_child] -= cluster_reward[c_child]
                else:
                    # Source is submaster
                    cp.src = [clusterstructure.submasters[c]]
                    # Sinks are submaster in active higher ranked subgraphs
                    cp.snk = sinks

                solution = cp.diameter_solve_flow(
                    master=True,
                    connectivity=not soft,
                    frontier_reward=True,
                    verbose=verbose,
                )

                print("Solved cluster {}".format(c))
                problems[c] = cp

            if soft:
                # find activated subgraphs
                activated_subgraphs = set()
                # connectivity activated nodes
                for t, conn_t in problems[c].conn.items():
                    for child in clusterstructure.child_clusters[c]:
                        for (v1, v2, b) in conn_t:
                            if (
                                b == "master"
                                and v2 == child[1]
                                and child[0] in problems
                            ):  # if b is a tuple then b is a master (by the construction of conn)
                                activated_subgraphs.add(child[0])

                # transition activated nodes
                for t, tran_t in problems[c].tran.items():
                    for child in clusterstructure.child_clusters[c]:
                        for (v1, v2, b) in tran_t:
                            if (
                                b == "master"
                                and v2 == child[1]
                                and child[0] in problems
                            ):  # if b is a tuple then b is a master (by the construction of conn)
                                activated_subgraphs.add(child[0])

                # remove problems that are not reached
                for child in clusterstructure.child_clusters[c]:
                    if child[0] not in activated_subgraphs:
                        del problems[child[0]]

                # set reward to optimal value plus evacuation value
                cluster_reward[c] = (
                    solution["primal objective"] - len(clusterstructure.agent_clusters[c]) * self.evac_reward
                )

        print("Merging {}".format(problems.keys()))
        self.merge_solutions(problems, clusterstructure, order="forward")

        # find activated agents
        active_agents = self.activate_agents(problems, clusterstructure)

        return ToFrontierData(
            initial_pos=self.graph.agents,
            clusterstructure=clusterstructure,
            active_clusters=active_clusters,
            active_agents=active_agents,
        )


    def solve_to_base_problem(self, tofront_data, verbose=False, dead=True):
        """solve a connectivity problem to get information back to the base,
        using data from a to_frontier solution"""

        self.prepare_problem()

        problems = {}
        evac = {}

        for c in children_first_iter(tofront_data.clusterstructure.child_clusters):

            if c in tofront_data.active_clusters:

                # Setup connectivity problem
                cp = ConnectivityProblem()
                cp.always_src = True

                # Masters are active agents in cluster
                dead_agents = [
                    tofront_data.clusterstructure.submasters[child[0]]
                    for child in tofront_data.clusterstructure.child_clusters[c]
                    if child[0] not in tofront_data.active_clusters
                ]
                cp.master = [
                    r for r in tofront_data.active_agents[c] if r not in dead_agents
                ]

                # Setup agents and static agents in subgraph
                agents = {}
                static_agents = []
                sources = []
                additional_nodes = []

                for r in tofront_data.clusterstructure.agent_clusters[c]:
                    agents[r] = self.graph.agents[r]
                    if r in self.static_agents:
                        static_agents.append(r)
                    if self.graph.nodes[self.graph.agents[r]]["frontiers"] != 0:
                        sources.append(r)

                for C in tofront_data.clusterstructure.child_clusters[c]:
                    agents[tofront_data.clusterstructure.submasters[C[0]]] = C[1]
                    static_agents.append(tofront_data.clusterstructure.submasters[C[0]])
                    additional_nodes.append(C[1])
                    if C[0] in tofront_data.active_clusters:
                        sources.append(tofront_data.clusterstructure.submasters[C[0]])
                sources = list(set(sources))

                cp.static_agents = static_agents
                cp.eagents = [r for r in self.eagents if r in agents]
                cp.big_agents = [r for r in self.big_agents if r in agents]

                g = deepcopy(self.graph)
                g.remove_nodes_from(
                    set(self.graph.nodes)
                    - set(tofront_data.clusterstructure.subgraphs[c])
                    - set(additional_nodes)
                )
                g.init_agents(agents)
                cp.graph = g

                cp.reward_dict = betweenness_centrality(g)
                norm = max(cp.reward_dict.values())
                if norm == 0:
                    norm = 1
                cp.reward_dict = {
                    v: self.max_centrality_reward * val / norm
                    for v, val in cp.reward_dict.items()
                }

                # force submaster to go back to initial positions for communication
                c_submaster = tofront_data.clusterstructure.submasters[c]
                cp.final_position = {c_submaster : tofront_data.initial_pos[c_submaster]}
                # Sources are submaster in active higher ranked subgraphs
                cp.src = sources
                # Submaster is sink
                cp.snk = [tofront_data.clusterstructure.submasters[c]]  
                # force master to be static
                cp.extra_constr = [
                    (
                        "constraint_static_master",
                        tofront_data.clusterstructure.submasters[c],
                    )
                ]

                cp.diameter_solve_flow(
                    master=True,
                    connectivity=True,
                    frontier_reward=False,
                    verbose=verbose,
                )

                problems[c] = cp

            elif dead:

                evac[c] = self.find_evac_path(c, tofront_data)

        self.merge_solutions(
            problems, tofront_data.clusterstructure, evac=evac, order="reversed"
        )


def spectral_clustering(graph_tran, k, static_agents):
    """
    cluster agents into k clusters 

    RETURNS
    =======

        agent_clusters  : dict(c: set(r))  clustering of agents
    """

    # Step 1: create dynamic agent graph
    da_graph = nx.Graph()

    dynamic_agents = [r for r in graph_tran.agents if r not in static_agents]

    dynamic_agent_nodes = set(graph_tran.agents[r] for r in dynamic_agents)

    da_agents = [
        (
            tuple(
                [
                    r
                    for r in graph_tran.agents
                    if graph_tran.agents[r] == v and r not in static_agents
                ]
            ),
            v,
        )
        for v in dynamic_agent_nodes
    ]
    for r, v in da_agents:
        da_graph.add_node(r)

    for (r1, v1), (r2, v2) in product(da_agents, da_agents):
        if r1 == r2:
            add_path = False
        else:
            add_path = True
            shortest_path = nx.shortest_path(
                graph_tran, source=v1, target=v2, weight="weight"
            )
            for v in shortest_path:
                if v in dynamic_agent_nodes and v != v1 and v != v2:
                    add_path = False
        if add_path:
            w = nx.shortest_path_length(
                graph_tran, source=v1, target=v2, weight="weight"
            )
            da_graph.add_edge(r1, r2, weight=w)

    # add small weight to every edge to prevent divide by zero. (w += 0.1 -> agents in same node has 10 similarity)
    for edge in da_graph.edges:
        da_graph.edges[edge]["weight"] += 0.1

    # inverting edge weights as spectral_clustering use them as similarity measure
    for edge in da_graph.edges:
        da_graph.edges[edge]["weight"] = 1 / da_graph.edges[edge]["weight"]

    # Step 2: cluster it

    # Cluster (uses weights as similarity measure)
    sc = SpectralClustering(k, affinity="precomputed")
    sc.fit(nx.to_numpy_matrix(da_graph))

    # construct dynamic agent clusters
    agent_clusters = {}
    c_group = []

    for c in range(k):
        for i, r_list in enumerate(da_graph):
            if sc.labels_[i] == c:
                c_group += r_list
        agent_clusters["cluster" + str(c)] = c_group
        c_group = []

    # add static agents to nearest cluster
    for (r, v) in graph_tran.agents.items():
        added = False
        if r in static_agents:
            start_node = nx.multi_source_dijkstra(
                graph_tran, sources=dynamic_agent_nodes, target=v
            )[1][0]
            for c in agent_clusters:
                for rc in agent_clusters[c]:
                    if start_node == graph_tran.agents[rc] and not added:
                        agent_clusters[c].append(r)
                        added = True

    return agent_clusters


def inflate_clusters(graph, agent_clusters, master):
    """
    inflate a clustering of agents to return a clustering of nodes
    
    INPUTS
    ======
        graph
        graph_tran
        agent_clusters
        master

    RETURNS
    =======
        clusters  : dict(c: set(v))
        child_clusters  : dict(c0 : (c1,v1))
        parent_clusters : dict(c1 : (c0,v0))

    """

    graph_tran = deepcopy(graph)
    graph_tran.remove_edges_from(list(graph.conn_edges()))

    # node clusters with agent positions
    clusters = {
        c: set(graph.agents[r] for r in r_list) for c, r_list in agent_clusters.items()
    }

    child_clusters = {c: [] for c in agent_clusters.keys()}
    parent_clusters = {c: [] for c in agent_clusters.keys()}

    # start with master cluster active
    active_clusters = [c for c, r_list in agent_clusters.items() if master in r_list]

    while True:
        # check if active cluster can activate other clusters directly
        found_new = True
        while found_new:
            found_new = False
            for c0, c1 in product(active_clusters, clusters.keys()):
                if c0 == c1 or c1 in active_clusters:
                    continue
                for v0, v1 in product(clusters[c0], clusters[c1]):
                    if graph.has_conn_edge(v0, v1):
                        active_clusters.append(c1)
                        child_clusters[c0].append((c1, v1))
                        parent_clusters[c1].append((c0, v0))
                        found_new = True

        # all active nodes
        active_nodes = set.union(*[clusters[c] for c in active_clusters])
        free_nodes = set(graph.nodes) - set.union(*clusters.values())

        # make sure clusters are connected via transitions, if not split
        for c, v_set in clusters.items():
            c_subg = graph_tran.subgraph(v_set | free_nodes).to_undirected()

            comps = nx.connected_components(c_subg)
            comp_dict = {
                i: [r for r in agent_clusters[c] if graph.agents[r] in comp]
                for i, comp in enumerate(comps)
            }

            comp_dict = {i: l for i, l in comp_dict.items() if len(l) > 0}

            if len(comp_dict) > 1:
                # split and restart
                del agent_clusters[c]
                for i, r_list in comp_dict.items():
                    agent_clusters["{}_{}".format(c, i + 1)] = r_list
                return inflate_clusters(graph, agent_clusters, master)  # recursive call

        # find closest neighbor and activate it
        neighbors = graph.post_tran(active_nodes) - active_nodes

        if len(neighbors) == 0:
            break  # nothing more to do

        new_cluster, new_node, min_dist, min_path = -1, -1, 99999, None
        for c in active_clusters:
            c_neighbors = graph.post_tran(clusters[c])

            agent_positions = [graph.agents[r] for r in agent_clusters[c]]

            for n in neighbors & c_neighbors:

                dist, path = nx.multi_source_dijkstra(
                    graph_tran, sources=agent_positions, target=n
                )
                if dist < min_dist:
                    min_dist, new_node, new_cluster, min_path = dist, n, c, path

        clusters[new_cluster].add(new_node)

    return clusters, child_clusters, parent_clusters


def parent_first_iter(child_clusters):
    """iterate over clusters s.t. parents come before children"""

    # find master cluster: the one that is not a child
    active = set(child_clusters.keys())
    for children in child_clusters.values():
        for child in children:
            active.remove(child[0])
    active = list(active)

    while len(active) > 0:
        c = active.pop(0)
        if c in child_clusters:
            for child in child_clusters[c]:
                active.append(child[0])
        yield c


def children_first_iter(child_clusters):
    """iterate over clusters s.t. children come before parents"""
    for c in reversed(list(parent_first_iter(child_clusters))):
        yield c


def problem_size(graph, subgraphs, agent_clusters, child_clusters, verbose=False):
    """heuristic to estimate the size of a cluster problem
    
    RETURNS
    =======
        cluster_size  : dict(c : int)
    """

    cluster_size = {}
    for c in subgraphs:

        # number of robots
        R = len(agent_clusters[c]) + len(child_clusters[c])
        # number of nodes
        V = len(subgraphs[c]) + len(child_clusters[c])
        # number of occupied nodes
        Rp = len(
            set(v for r, v in graph.agents.items() if r in agent_clusters[c])
        )
        # number of transition edges
        Et = graph.number_of_tran_edges(subgraphs[c])
        # number of connectivity edges
        Ec = graph.number_of_conn_edges(subgraphs[c])
        # graph diameter
        D = nx.diameter(nx.subgraph(graph, subgraphs[c]))

        T = int(max(D / 2, D - int(Rp / 2)))

        size = R * Et * T

        if verbose:
            print(
                "{} size={} [R={}, V={}, Et={}, Ec={}, D={}, Rp={}]".format(
                    c, size, R, V, Et, Ec, D, Rp
                )
            )

        cluster_size[c] = size

    return cluster_size


def cluster(num_clusters, graph, graph_tran, master, static_agents, max_problem_size, verbose=False):
    """main clustering loop"""

    small_problem_size = False

    if num_clusters == None:
        num_clusters = 1
        agent_clusters = {"cluster0": [r for r in graph.agents]}
        subgraphs, child_clusters, parent_clusters = inflate_clusters(
            graph, agent_clusters, master
        )
        small_problem_size = (
            max(
                problem_size(
                    graph, subgraphs, agent_clusters, child_clusters
                ).values()
            )
            < max_problem_size
        )

        # Strategy 1: cluster
        while not small_problem_size and num_clusters + 1 < len(
            set(
                graph.agents[r]
                for r in graph.agents
                if r not in static_agents
            )
        ):
            num_clusters += 1

            print("Strategy 1: clustering with k={}".format(num_clusters))
            # detect agent clusters
            agent_clusters = spectral_clustering(
                graph_tran, num_clusters, static_agents
            )
            # dictionary mapping cluster to nodes
            subgraphs, child_clusters, parent_clusters = inflate_clusters(
                graph, agent_clusters, master
            )

            small_problem_size = (
                max(
                    problem_size(
                        graph, subgraphs, agent_clusters, child_clusters, verbose=True
                    ).values()
                )
                < max_problem_size
            )

        # Strategy 2: kill parts of graph
        while not small_problem_size:
            print("Strategy 2: Kill frontiers")

            # TODO: THIS IS UNTESTED, DOES NOT WORK WHEN THERE ARE NO FRONTIERS
            print("There are {} frontiers".format(len(v
                        for v in graph.nodes
                        if graph.nodes[v]["frontiers"] != 0
                        and not graph.nodes[v]["dead"])))

            # find frontier furthest away from master for large problems
            cluster_size = problem_size(graph, subgraphs, agent_clusters, child_clusters)

            for c, val in cluster_size.items():
                if val >= max_problem_size:

                    # revive
                    for v in graph.nodes:
                        graph.nodes[v]["dead"] = False

                    # find not occupied frontiers in large subgraphs
                    frontiers = [
                        v
                        for v in subgraphs[c]
                        if graph.nodes[v]["frontiers"] != 0
                        and not graph.nodes[v]["dead"]
                    ]
                    for r in agent_clusters[c]:
                        if graph.agents[r] in frontiers:
                            frontiers.remove(graph.agents[r])

                    if len(parent_clusters[c]) > 0:
                        master_node = parent_clusters[c][0][1]
                    else:
                        master_node = graph.agents[master]

                    max_length = None
                    max_frontier = None
                    for f in frontiers:
                        length, path = nx.single_source_dijkstra(
                            graph_tran, source=master_node, target=f
                        )
                        if max_length == None:
                            max_length = length
                            max_frontier = f
                        elif length > max_length:
                            max_length = length
                            max_frontier = f

                    # TODO: should this be there??
                    # graph.nodes[f]["dead"] = True

                    # kill max_frontier
                    graph.nodes[max_frontier]["dead"] = True

                    # kill nodes with only dead children and no agents in node
                    dead_nodes = self.expand_dead_nodes(set([f, max_frontier]))

                    print(
                        "eliminated frontier and removed {} nodes!".format(
                            len(dead_nodes)
                        )
                    )
                    for v in dead_nodes:
                        graph.nodes[v]["dead"] = True

            small_problem_size = (
                max(
                    problem_size(
                        graph, subgraphs, agent_clusters, child_clusters
                    ).values()
                )
                < max_problem_size
            )

    else:
        # detect agent clusters
        agent_clusters = spectral_clustering(graph_tran, num_clusters, static_agents)

        # dictionary mapping cluster to nodes
        subgraphs, child_clusters, parent_clusters = inflate_clusters(
            graph, agent_clusters, master
        )

    # create dictionaries mapping cluster to submaster, subsinks
    master_cluster = next(
        c for c in agent_clusters if master in agent_clusters[c]
    )
    submasters = {master_cluster: master}
    subsinks = {c: [] for c in subgraphs}

    for c in parent_first_iter(child_clusters):
        for child, r in product(child_clusters[c], graph.agents):
            if graph.agents[r] == child[1]:
                if child[0] not in submasters:
                    submasters[child[0]] = r
                if r not in subsinks[c]:
                    subsinks[c].append(r)

    return ClusterStructure(
        subgraphs=subgraphs,
        agent_clusters=agent_clusters,
        child_clusters=child_clusters,
        parent_clusters=parent_clusters,
        submasters=submasters,
        subsinks=subsinks,
    )
