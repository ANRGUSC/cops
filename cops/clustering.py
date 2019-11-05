from dataclasses import dataclass

import networkx as nx
from networkx.algorithms.centrality import betweenness_centrality
from copy import deepcopy
from itertools import product
from sklearn.cluster import SpectralClustering

from cops.graph import Graph
from cops.problem import AbstractConnectivityProblem, ConnectivityProblem


@dataclass
class ClusterStructure(object):
    subgraphs: dict = None
    agent_clusters: dict = None
    child_clusters: dict = None
    parent_clusters: dict = None
    submasters: dict = None
    subsinks: dict = None
    dead_nodes: set = None

    def subproblem(self, c, clusterproblem):
        # return basic connectivityproblem
        #  (graph, agents, eagents, big_agents, reward_dict)
        # induced by cluster c

        agents = {r: clusterproblem.graph.agents[r] for r in self.agent_clusters[c]}

        static_agents = [r for r in agents.keys() if r in clusterproblem.static_agents]

        # add childcluster stuff
        addn_nodes = set()
        for C in self.child_clusters[c]:
            agents[self.submasters[C[0]]] = C[1]
            static_agents.append(self.submasters[C[0]])
            addn_nodes.add(C[1])

        G = deepcopy(clusterproblem.graph)
        del_nodes = set(clusterproblem.graph.nodes) - self.subgraphs[c] - addn_nodes
        G.remove_nodes_from(del_nodes)
        G.init_agents(agents)

        # basic rewards based on centrality
        reward_dict = betweenness_centrality(nx.DiGraph(G))
        norm = max(reward_dict.values())
        if norm == 0:
            norm = 1
        reward_dict = {
            v: clusterproblem.max_centrality_reward * val / norm
            for v, val in reward_dict.items()
        }

        # initialize subproblem
        return ConnectivityProblem(
            graph=G,
            static_agents=static_agents,
            eagents=[r for r in agents if r in clusterproblem.eagents],
            big_agents=[r for r in agents if r in clusterproblem.big_agents],
            reward_dict=reward_dict,
        )


@dataclass
class ToFrontierData(object):
    initial_pos: dict = None
    cs: ClusterStructure = None
    frontier_clusters: list = None
    active_agents: list = None


class ClusterProblem(AbstractConnectivityProblem):
    def __init__(self, **kwargs):
        super(ClusterProblem, self).__init__()

        # CLUSTERING PARAMETERS
        self.num_clusters = None  # desired number of clusters
        self.max_problem_size = 4000  # max problem size

        self.max_centrality_reward = 20
        self.evac_reward = 100

    # === HELPER FUNCTIONS======================================================

    def prepare_problem(self, remove_dead=True):

        super(ClusterProblem, self).prepare_problem()

        #  1. graph_tran: graph with only mobility edges
        #  2. node_children_dict: tree based on length_to_master
        #  3. length_to_master: distance from each robot to master node

        # save a graph with transition only
        self.graph_tran = deepcopy(self.graph)
        self.graph_tran.remove_edges_from(list(self.graph.conn_edges()))

        if self.master is None:
            return

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

    def expand_from_leaves(self, initial_set):
        """
        traverse graph from outside to in, add nodes to set
        if all their children are in set
        """

        node_order = reversed(sorted(self.length_to_master.items(), key=lambda v: v[1]))

        # kill nodes with only dead children and no agents in node
        for v, _ in node_order:
            if (
                (
                    "frontiers" not in self.graph.nodes[v]
                    or self.graph_tran.nodes[v]["frontiers"] == 0
                )
                and v not in self.graph_tran.agents.values()
                and self.node_children_dict[v] <= initial_set
            ):
                initial_set.add(v)

        return initial_set

    def find_evac_path(self, c, tofront_data):
        """evacuate a cluster by finding shortest path from each robot in cluster to origin"""

        master_nodes = set(v for r, v in self.graph.agents.items() if r == self.master)
        active_nodes = set(
            v
            for C in tofront_data.frontier_clusters
            for v in tofront_data.cs.subgraphs[C]
        ).union(master_nodes)

        evac = {}

        for r, v in self.graph.agents.items():
            if r in tofront_data.cs.agent_clusters[c]:
                _, path = nx.multi_source_dijkstra(
                    self.graph_tran, sources=active_nodes, target=v, weight="weight"
                )
                path = path[::-1]
                evac[r] = path

        return evac

    # ===SOLVE FUNCTIONS=========================================================

    def frontier_clusters(self, cs):
        # find all clusters that have frontiers, or that are on the way
        # to a frontier
        frontier_clusters = set(
            c
            for c in cs.subgraphs
            if any(self.graph.nodes[v]["frontiers"] != 0 for v in cs.subgraphs[c])
        )

        for c in children_first_iter(cs.child_clusters):
            if any(child_c in frontier_clusters for child_c, _ in cs.child_clusters[c]):
                frontier_clusters.add(c)

        return frontier_clusters

    def activate_agents(self, problems, cs):
        """find agents that received masterdata in previous problem"""
        active_agents = {}
        for c in cs.subgraphs:

            agents = set(r for r in cs.agent_clusters[c]).union(
                cs.submasters[child[0]] for child in cs.child_clusters[c]
            )

            # initial position activated nodes
            active_agents[c] = [
                r
                for r in self.graph.agents
                if (self.graph.agents[r] == self.graph.agents[cs.submasters[c]])
            ]

            # connectivity activated nodes
            for _, conn_t in problems[c].conn.items():
                for (_, v2, b) in conn_t:
                    if (
                        b == "master"
                    ):  # if b is a tuple then b is a master (by the construction of conn)
                        for r in agents:
                            if self.graph.agents[r] == v2:
                                if r not in active_agents[c]:
                                    active_agents[c].append(r)
                    elif b == cs.submasters[c]:
                        for r in agents:
                            if self.graph.agents[r] == v2:
                                if r not in active_agents[c]:
                                    active_agents[c].append(r)

            # transition activated nodes
            for _, tran_t in problems[c].tran.items():
                for (_, v2, b) in tran_t:
                    if (
                        b == "master"
                    ):  # if b is a tuple then b is a master (by the construction of tran)
                        for r in agents:
                            if self.graph.agents[r] == v2 and r not in active_agents[c]:
                                active_agents[c].append(r)
                    elif b == cs.submasters[c]:
                        for r in agents:
                            if self.graph.agents[r] == v2 and r not in active_agents[c]:
                                active_agents[c].append(r)
        return active_agents

    def merge_solutions(self, problems, cs, evac=None, order="forward"):
        """merge and store solutions"""

        # find master cluster
        master_cluster = set(cs.child_clusters.keys())
        for children in cs.child_clusters.values():
            for child in children:
                master_cluster.remove(child[0])
        master_cluster = list(master_cluster)[0]

        # Find start and end times for each cluster
        fwd_start_time = {master_cluster: 0}
        rev_end_time = {master_cluster: 0}

        for c in parent_first_iter(cs.child_clusters):
            if c not in problems:
                continue

            for child, _ in cs.child_clusters[c]:
                submaster = cs.submasters[child]
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
        for c in parent_first_iter(cs.child_clusters):
            if c in problems:
                for (r, t), v in problems[c].traj.items():
                    if r not in cs.subsinks[c]:
                        traj[r, start_time[c] + t] = v

        # Trajectories for evac
        if evac is not None:
            for c in evac:
                for r, path in evac[c].items():
                    for i, v in enumerate(path):
                        traj[r, i] = v

        # Communication dictionary for cluster problem
        conn = {t: set() for t in range(T_sol + 1)}
        for c in parent_first_iter(cs.child_clusters):
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

        self.prepare_problem(remove_dead=True)

        cs = clustering(self, verbose=verbose)
        self.subgraphs = cs.subgraphs
        for n in self.graph.nodes:
            if not any(n in v_list for j, v_list in self.subgraphs.items()):
                self.graph.nodes[n]['dead'] = True

        frontier_clusters = self.frontier_clusters(cs)

        problems = {}
        cluster_reward = {}

        for c in children_first_iter(cs.child_clusters):

            if c in frontier_clusters or dead:

                cp = cs.subproblem(c, self)

                cp.always_src = True
                cp.master = cs.submasters[c]

                if c not in frontier_clusters:  # reward to activate agents
                    cp.reward_dict = {v: self.evac_reward for v in cp.graph.nodes}

                if soft:  # reward to activate subclusters
                    for c_child, v_child in cs.child_clusters[c]:
                        cp.reward_dict[v_child] -= cluster_reward[c_child]
                else:  # hard constraint to activate subclusters
                    cp.src = [cs.submasters[c]]
                    cp.snk = [cs.submasters[child[0]] for child in cs.child_clusters[c]]

                solution = cp.diameter_solve_flow(
                    master=True,
                    connectivity=not soft,
                    frontier_reward=True,
                    verbose=verbose,
                )

                print("Solved cluster {}".format(c))
                problems[c] = cp

                # set soft reward to optimal value plus evacuation value
                cluster_reward[c] = solution["primal objective"]
                cluster_reward[c] -= len(cs.agent_clusters[c]) * self.evac_reward

        # find agents that received masterdata
        active_agents = self.activate_agents(problems, cs)

        # find activated clusters
        active_clusters = set(c for c in cs.subgraphs if len(active_agents[c]) != 0)

        # delete cluster solutions that were not activated
        for c in cs.subgraphs:
            if c not in active_clusters:
                del problems[c]

        # store solution
        self.merge_solutions(problems, cs, order="forward")

        return ToFrontierData(
            initial_pos=self.graph.agents,
            cs=cs,
            frontier_clusters=frontier_clusters,
            active_agents=active_agents,
        )

    def solve_to_base_problem(self, tofront_data, verbose=False, dead=True):
        """solve a connectivity problem to get information back to the base,
        using data from a to_frontier solution"""

        self.prepare_problem(remove_dead=False)
        self.subgraphs = tofront_data.cs.subgraphs
        for n in self.graph.nodes:
            if not any(n in v_list for j, v_list in self.subgraphs.items()):
                self.graph.nodes[n]['dead'] = True

        problems = {}
        evac = {}

        for c in children_first_iter(tofront_data.cs.child_clusters):

            if c in tofront_data.frontier_clusters:

                cp = tofront_data.cs.subproblem(c, self)
                cp.master = tofront_data.active_agents[c]

                # send data from frontiers explorers ...
                cp.src = []
                for r in tofront_data.cs.agent_clusters[c]:
                    if self.graph.nodes[self.graph.agents[r]]["frontiers"] != 0:
                        cp.src.append(r)
                # ... and from activated subclusters
                for C in tofront_data.cs.child_clusters[c]:
                    if C[0] in tofront_data.frontier_clusters:
                        cp.src.append(tofront_data.cs.submasters[C[0]])

                # to submaster
                cp.snk = [tofront_data.cs.submasters[c]]

                # submaster must return to his position
                c_submaster = tofront_data.cs.submasters[c]
                cp.final_position = {c_submaster: tofront_data.initial_pos[c_submaster]}

                # force master to be static
                cp.extra_constr = [
                    ("constraint_static_master", tofront_data.cs.submasters[c])
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

        self.merge_solutions(problems, tofront_data.cs, evac=evac, order="reversed")


def agent_clustering(cp, num_clusters):
    """
    cluster agents into k clusters

    INPUTS
    ======

        cp.graph_tran  : transition graph
        cp.static_agents  : static agent list


    RETURNS
    =======

        agent_clusters  : dict(c: set(r))  clustering of agents
    """

    # Step 1: create dynamic agent graph
    da_graph = nx.Graph()

    dynamic_agents = [r for r in cp.graph_tran.agents if r not in cp.static_agents]

    dynamic_agent_nodes = set(cp.graph_tran.agents[r] for r in dynamic_agents)

    da_agents = [
        (tuple([r for r in dynamic_agents if cp.graph_tran.agents[r] == v]), v)
        for v in dynamic_agent_nodes
    ]

    for r, v in da_agents:
        da_graph.add_node(r)

    for (r1, v1), (r2, v2) in product(da_agents, da_agents):
        if r1 == r2:
            add_path = False
        else:
            add_path = True
            sh_path = nx.shortest_path(
                cp.graph_tran, source=v1, target=v2, weight="weight"
            )
            for v in sh_path:
                if v in dynamic_agent_nodes and v != v1 and v != v2:
                    add_path = False
        if add_path:
            w = nx.shortest_path_length(
                cp.graph_tran, source=v1, target=v2, weight="weight"
            )
            da_graph.add_edge(r1, r2, weight=w)

    # add small weight to every edge to prevent divide by zero. (w += 0.1 -> agents in same node has 10 similarity)
    for edge in da_graph.edges:
        da_graph.edges[edge]["weight"] += 0.1

    # inverting edge weights as agent_clustering use them as similarity measure
    for edge in da_graph.edges:
        da_graph.edges[edge]["weight"] = 1 / da_graph.edges[edge]["weight"]

    # Step 2: cluster it

    # Cluster (uses weights as similarity measure)
    sc = SpectralClustering(num_clusters, affinity="precomputed")
    sc.fit(nx.to_numpy_matrix(da_graph))

    # construct dynamic agent clusters
    agent_clusters = {}
    c_group = []

    for c in range(num_clusters):
        for i, r_list in enumerate(da_graph):
            if sc.labels_[i] == c:
                c_group += r_list
        agent_clusters["cluster" + str(c)] = c_group
        c_group = []

    # add static agents to nearest cluster
    for (r, v) in cp.graph_tran.agents.items():
        added = False
        if r in cp.static_agents:
            start_node = nx.multi_source_dijkstra(
                cp.graph_tran, sources=dynamic_agent_nodes, target=v
            )[1][0]
            for c in agent_clusters:
                for rc in agent_clusters[c]:
                    if start_node == cp.graph_tran.agents[rc] and not added:
                        agent_clusters[c].append(r)
                        added = True

    return agent_clusters


def inflate_agent_clusters(cp, cs):
    """
    inflate a clustering of agents to return a clustering of nodes

    REQUIRES
    ========
        cp
        cs.agent_clusters
        cs.dead_nodes      (optional)

    WRITES TO
    =========
        cs.subgraphs  : dict(c: set(v))
        cs.child_clusters  : dict(c0 : (c1,v1))
        cs.parent_clusters : dict(c1 : (c0,v0))

    """

    # node clusters with agent positions
    clusters = {
        c: set(cp.graph.agents[r] for r in r_list)
        for c, r_list in cs.agent_clusters.items()
    }

    child_clusters = {c: set() for c in cs.agent_clusters.keys()}
    parent_clusters = {}

    # start with master cluster active
    active_clusters = [
        c for c, r_list in cs.agent_clusters.items() if cp.master in r_list
    ]

    while True:
        # check if active cluster can activate other clusters directly
        found_new = True
        while found_new:
            found_new = False
            for c0, c1 in product(active_clusters, clusters.keys()):
                if c0 == c1 or c1 in active_clusters:
                    continue
                for v0, v1 in product(clusters[c0], clusters[c1]):
                    if cp.graph.has_conn_edge(v0, v1):
                        active_clusters.append(c1)
                        child_clusters[c0].add((c1, v1))
                        parent_clusters[c1] = (c0, v0)
                        found_new = True

        # all active nodes
        active_nodes = set.union(*[clusters[c] for c in active_clusters])
        free_nodes = set(cp.graph.nodes) - set.union(*clusters.values())
        if cs.dead_nodes is not None:
            free_nodes -= cs.dead_nodes

        # make sure clusters are connected via transitions, if not split
        for c, v_set in clusters.items():
            c_subg = cp.graph_tran.subgraph(v_set | free_nodes).to_undirected()

            comps = nx.connected_components(c_subg)
            comp_dict = {
                i: [r for r in cs.agent_clusters[c] if cp.graph.agents[r] in comp]
                for i, comp in enumerate(comps)
            }

            comp_dict = {i: l for i, l in comp_dict.items() if len(l) > 0}

            if len(comp_dict) > 1:
                # split and restart
                del cs.agent_clusters[c]
                for i, r_list in comp_dict.items():
                    cs.agent_clusters["{}_{}".format(c, i + 1)] = r_list
                return inflate_agent_clusters(cp, cs)  # recursive call

        # find closest neighbor and activate it
        neighbors = cp.graph.post_tran(active_nodes) - active_nodes
        if cs.dead_nodes is not None:
            neighbors -= cs.dead_nodes

        if len(neighbors) == 0:
            break  # nothing more to do

        new_cluster, new_node, min_dist = -1, -1, 99999
        for c in active_clusters:
            c_neighbors = cp.graph.post_tran(clusters[c])

            agent_positions = [cp.graph.agents[r] for r in cs.agent_clusters[c]]

            for n in neighbors & c_neighbors:

                dist, _ = nx.multi_source_dijkstra(
                    cp.graph_tran, sources=agent_positions, target=n
                )
                if dist < min_dist:
                    min_dist, new_node, new_cluster = dist, n, c

        clusters[new_cluster].add(new_node)

    cs.subgraphs = clusters
    cs.child_clusters = child_clusters
    cs.parent_clusters = parent_clusters

    return cs


def clustering(cp, verbose=False):
    """main clustering loop"""

    done = False

    initial_dead = set()
    for v in cp.graph.nodes:
        if len(cp.node_children_dict[v]) != 0:
            continue  # it has children
        if cp.graph.is_local_frontier(v):
            continue  # it's a frontier
        if v in cp.graph.agents.values():
            continue  # an agent is standing there
        initial_dead.add(v)

    cs = ClusterStructure(dead_nodes=cp.expand_from_leaves(initial_dead))

    max_num_cluster = len(set(cp.graph.agents.values())) - 1
    max_num_cluster -= len(set(cp.graph.agents[r] for r in cp.static_agents))

    if cp.num_clusters == None:
        num_clusters = 1
        cs.agent_clusters = {"cluster0": [r for r in cp.graph.agents]}
        cs = inflate_agent_clusters(cp, cs)
        done = max(problem_size(cp.graph, cs).values()) < cp.max_problem_size

        # Strategy 1: cluster
        while not done and num_clusters < max_num_cluster:
            num_clusters += 1

            print("Strategy 1: clustering with k={}".format(num_clusters))
            cs.agent_clusters = agent_clustering(cp, num_clusters)
            cs = inflate_agent_clusters(cp, cs)

            done = (
                max(problem_size(cp.graph, cs, verbose=True).values())
                < cp.max_problem_size
            )

        # Strategy 2: kill parts of graph
        while not done:
            print("Strategy 2: Kill frontiers")

            new_dead_nodes = kill_largest_frontiers(cp, cs)
            if len(new_dead_nodes) > 0:
                cs.dead_nodes |= new_dead_nodes
                cs = inflate_agent_clusters(cp, cs)
                done = max(problem_size(cp.graph, cs).values()) < cp.max_problem_size
            else:
                print("Strategy 2: didn't find any frontier to delete, returning...")
                break
    else:
        # create exactly cp.num_clusters clusters
        cs.agent_clusters = agent_clustering(cp, num_clusters=cp.num_clusters)
        cs = inflate_agent_clusters(cp, cs)

    # create dictionaries mapping cluster to submaster, subsinks
    master_cluster = next(
        c for c in cs.agent_clusters if cp.master in cs.agent_clusters[c]
    )
    cs.submasters = {master_cluster: cp.master}
    cs.subsinks = {c: [] for c in cs.subgraphs}

    for c in parent_first_iter(cs.child_clusters):
        for child, r in product(cs.child_clusters[c], cp.graph.agents):
            if cp.graph.agents[r] == child[1]:
                if child[0] not in cs.submasters:
                    cs.submasters[child[0]] = r
                if r not in cs.subsinks[c]:
                    cs.subsinks[c].append(r)

    return cs


def problem_size(graph, cs, verbose=False):
    """heuristic to estimate the size of a cluster problem

    RETURNS
    =======
        cluster_size  : dict(c : int)
    """

    cluster_size = {}
    for c in cs.subgraphs:

        # number of robots
        R = len(cs.agent_clusters[c]) + len(cs.child_clusters[c])
        # number of nodes
        V = len(cs.subgraphs[c]) + len(cs.child_clusters[c])
        # number of occupied nodes
        Rp = len(set(v for r, v in graph.agents.items() if r in cs.agent_clusters[c]))
        # number of transition edges
        Et = graph.number_of_tran_edges(cs.subgraphs[c])
        # number of connectivity edges
        Ec = graph.number_of_conn_edges(cs.subgraphs[c])
        # graph diameter
        D = nx.diameter(nx.subgraph(graph, cs.subgraphs[c]))

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


def kill_largest_frontiers(cp, cs):
    """ find clusters that violate size limit, and remove
    a frontier from them """

    # find frontier furthest away from master for large problems
    cluster_size = problem_size(cp.graph, cs)

    initial_dead = set()

    for c, val in cluster_size.items():
        if val < cp.max_problem_size:
            continue

        # find unoccupied frontiers in large subgraphs
        frontiers = [v for v in cs.subgraphs[c] if cp.graph.nodes[v]["frontiers"] != 0]
        for r in cs.agent_clusters[c]:
            if cp.graph.agents[r] in frontiers:
                frontiers.remove(cp.graph.agents[r])

        if c in cs.parent_clusters:
            master_node = cs.parent_clusters[c][1]
        else:
            master_node = cp.graph.agents[cp.master]

        max_length = None
        max_frontier = None
        for f in frontiers:
            length, _ = nx.single_source_dijkstra(
                cp.graph_tran, source=master_node, target=f
            )
            if max_length == None:
                max_length = length
                max_frontier = f
            elif length > max_length:
                max_length = length
                max_frontier = f

        if max_frontier is not None:
            initial_dead.add(max_frontier)

    # kill nodes with only dead children and no agents in node
    return cp.expand_from_leaves(initial_dead)


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
