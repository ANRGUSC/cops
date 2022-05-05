import time
import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import networkx as nx
import scipy.sparse as sp

from itertools import chain, combinations, product
from networkx.drawing.nx_agraph import to_agraph

class AgentProblem(object):
    def __init__(self,r):
        # Problem definition
        self.graph = None  # full graph
        self.static_agents = None
        self.graph_list = []
        self.eagents = None
        self.r = r

        # Solution
        self.T_sol = 0
        self.traj = {}
        self.conn = {}

    ##PROPERTIES##
    @property
    def num_v(self):
        return self.graph.number_of_nodes()

    ##INDEX HELPER FUNCTIONS##

    def prepare_problem(self):

        if self.graph is None:
            raise Exception("Can not solve problem without 'graph'")

        # create subgraph
        self.known_graph = deepcopy(self.graph)
        unknown = [v for v in self.graph.nodes if not self.graph.nodes[v]["known"][self.r]]
        self.known_graph.remove_nodes_from(unknown)

        if self.static_agents is None:  # no static agents
            self.static_agents = []

        if (
            self.eagents is None
        ):  # if no exploration agents specified, all agents can explore
            self.eagents = [r for r in self.graph.agents]

        # Create dictionaries for (i,j)->k mapping for edges
        self.dict_tran = {(i, j): k for k, (i, j) in enumerate(self.graph.tran_edges())}
        self.dict_conn = {(i, j): k for k, (i, j) in enumerate(self.graph.conn_edges())}

        if not set(self.graph.agents.values()) <= set(self.graph.nodes()):
            raise Exception("Invalid initial positions")

        # add initial graph to graph_list
        self.graph_list.append(deepcopy(self.graph))

    ##SOLVER FUNCTIONS##

    def solve_explore(self):
        print("solve_explore agent", self.r)
        self.prepare_problem()

        # solve
        T_sol = 0
        AGENT = self.r
        AGENT_v = self.graph.agents[AGENT]

        # get to the closest frontier
        shortest_path, path_length = None, np.inf
        frontiers = self.graph.get_frontiers(AGENT)
        if not frontiers:
            return
        # print("AGENT POSITION:",AGENT_v)
        # print("FRONTIERS:",frontiers)
        for f in frontiers:
            this_l = nx.shortest_path_length(self.known_graph, AGENT_v, f)
            if this_l < path_length:
                shortest_path = nx.shortest_path(self.known_graph, AGENT_v, f)
                path_length = this_l
        shortest_path = shortest_path[1:]
        # print("SHORTEST PATH TO FRONTIER:",shortest_path)

        if not shortest_path:
            next_frontiers = [(a,b) for (a,b) in self.graph.tran_out_edges(AGENT_v) if not self.graph.nodes[b]["known"][AGENT]]
        else:
            next_frontiers = [(a,b) for (a,b) in self.graph.tran_out_edges(shortest_path[-1]) if not self.graph.nodes[b]["known"][AGENT]]
        if next_frontiers:
            next_v = random.choice(next_frontiers)[1]
        shortest_path.append(next_v)

        for t in range(len(shortest_path)):
            self.conn[t] = set()
            v = shortest_path[t]
            self.traj[(0,t)] = self.graph.agents[0]
            self.traj[(AGENT,t)] = v
            self.graph.nodes[v]["known"][AGENT] = True
            agent_locs = {0:self.graph.agents[0], AGENT:v}
            for (r,rv) in agent_locs.items():
                for (nbr_u, nbr_v) in self.graph.conn_out_edges(rv):
                    for (other_agent, other_agent_v) in agent_locs.items(): # TODO
                        if nbr_v == other_agent_v:
                            self.conn[t].add((rv,nbr_v,r))
                            print("Sharing known",rv,"->",nbr_v)
                            for ver in self.graph:
                                if self.graph.nodes[ver]["known"][AGENT]:
                                    print("setting",ver,"known about by",other_agent)
                                    self.graph.nodes[ver]["known"][other_agent] = True
            self.graph_list.append(deepcopy(self.graph))
            T_sol += 1

        self.T_sol = T_sol
        # print("TRAJ:",self.traj)
        # print("CONN:",self.conn)

        frontiers = self.graph.get_frontiers(r)
        self.graph.agents[AGENT] = self.traj[(AGENT,self.T_sol-1)]
        # print("AGENT POSITION:",self.graph.agents[AGENT])
        # print("FRONTIERS:",frontiers)

        # print("*****************")

    def solve_return(self):
        print("solve_return")
        self.prepare_problem()

        # solve
        T_sol = 0
        AGENT = 1 # TODO
        HOME = 0
        AGENT_v = self.graph.agents[AGENT]

        if AGENT_v == HOME:
            return

        # get to the closest frontier
        # print("AGENT POSITION:",AGENT_v)
        shortest_path = nx.shortest_path(self.known_graph, AGENT_v, HOME)
        shortest_path = shortest_path[1:]
        # print("SHORTEST PATH TO BASE:",shortest_path)

        v = AGENT_v
        for t in range(len(shortest_path)):
            self.conn[t] = set()
            v = shortest_path[t]
            self.traj[(0,t)] = self.graph.agents[0]
            self.traj[(AGENT,t)] = v
            agent_locs = {0:self.graph.agents[0], AGENT:v}
            for (r,rv) in agent_locs.items():
                for (nbr_u, nbr_v) in self.graph.conn_out_edges(rv):
                    if nbr_v in agent_locs.values():
                        self.conn[t].add((rv,nbr_v,r))
                        print("Sharing known",rv,"->",nbr_v)
                        for ver in self.graph:
                            if self.graph.nodes[ver]["known"][AGENT]:
                                print("setting",ver,"known about by",other_agent)
                                self.graph.nodes[ver]["known"][other_agent] = True
            self.graph_list.append(deepcopy(self.graph))
            T_sol += 1

        self.T_sol = T_sol
        # print("TRAJ:",self.traj)
        # print("CONN:",self.conn)

        frontiers = self.graph.get_frontiers()
        self.graph.agents[AGENT] = self.traj[(AGENT,self.T_sol-1)]
        # print("AGENT POSITION:",self.graph.agents[AGENT])
        # print("FRONTIERS (unchanged):",frontiers)

        # print("*****************")
