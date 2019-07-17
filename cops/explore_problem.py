import time
import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import networkx as nx
import scipy.sparse as sp

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from itertools import chain, combinations, product
from networkx.drawing.nx_agraph import to_agraph

@dataclass
class Variable(object):
    start: int
    size: int
    binary: bool = False

class ExplorationProblem(object):

    def __init__(self):
        # Problem definition
        self.graph = None                    #full graph
        self.T = None                        #Time horizon
        self.static_agents = None
        self.frontier_robot_dict = None
        self.graph_list = []
        self.eagents = None

        #variables
        self.z = None
        self.fbar = None
        self.k = None

        # Heuristic solution
        self.traj = {}
        self.conn = {}

    ##PROPERTIES##
    @property
    def num_r(self):
        return len(self.frontier_robot_dict)

    @property
    def num_v(self):
        return self.graph.number_of_nodes()

    @property
    def num_vars(self):
        return sum(var.size for var in self.vars.values())

    ##INDEX HELPER FUNCTIONS##

    def prepare_problem(self):

        if self.graph is None:
            raise Exception("Can not solve problem without 'graph'")

        if self.T is None:
            raise Exception("Can not solve problem without horizon 'T'")

        if self.static_agents is None: # no static agents
            self.static_agents = []

        if self.eagents is None: # if no exploration agents specified, all agents can explore
            self.eagents = [r for r in self.graph.agents]

        # Create dictionaries for (i,j)->k mapping for edges
        self.dict_tran = {(i,j): k for k, (i,j) in enumerate(self.graph.tran_edges())}
        self.dict_conn = {(i,j): k for k, (i,j) in enumerate(self.graph.conn_edges())}

        if not set(self.graph.agents.values()) <= set(self.graph.nodes()):
            raise Exception("Invalid initial positions")

        #add initial graph to graph_list
        self.graph_list.append(deepcopy(self.graph))

        #create frontier robot dictionary
        frontier_robots = []
        for r in self.graph.agents:
            if self.graph.is_frontier(self.graph.agents[r]):
                frontier_robots.append(r)
        self.frontier_robot_dict = {i: r for i, r in enumerate(frontier_robots)}

    def get_z_idx(self, r, v, t):
        return np.ravel_multi_index((t,v,r), (self.T+1, self.num_v, self.num_r))

    def get_fbar_idx(self, b, i, j, t):
        k = self.dict_conn[(i, j)]
        idx = np.ravel_multi_index((t,b,k), (self.T+1, self.num_r, len(self.dict_conn)))
        return idx

    def get_k_idx(self, r, v):
        idx = np.ravel_multi_index((r,v), (self.num_r, self.num_v))
        return idx

    ##HELPER FUNCTIONS

    def get_agent_position(self, fr, t):
        for v in self.graph.nodes:
            if self.z[self.get_z_idx(fr, v, t)] == 1:
                return v
        return None

    def share_data(self, t):
        for fr in self.frontier_robot_dict:
            fr_v = self.get_agent_position(fr, t)
            for nbr_v, nbr_r in product(self.graph.conn_out_edges(fr_v), self.frontier_robot_dict):
                if self.get_agent_position(nbr_r, t) == nbr_v[1]:
                    self.fbar[self.get_fbar_idx(fr,fr_v,nbr_v[1],t)] = 1
                    for v in self.graph.nodes:
                        if self.graph.nodes[v]['known']:
                            self.k[self.get_k_idx(fr,v)] = 1

    def choose_fork(self, v_alt, t):
        #Takes all possible next nodes, returns a random node with least number of robots in it
        num_r_in_v_alt = np.zeros(len(v_alt))
        for v, fr in product(v_alt, self.frontier_robot_dict):
            v_idx = v_alt.index(v)
            fr_v = self.get_agent_position(fr, t)
            if v == fr_v:
                num_r_in_v_alt[v_idx] += 1
        min_num_r = min(num_r_in_v_alt)
        min_v_alt = [v_alt[index] for index, element in enumerate(num_r_in_v_alt)
                          if min_num_r == element]
        return random.choice(min_v_alt)

    def set_return_path(self):
        for t in range(int(self.T/2) + 1, self.T + 1):
            for fr in self.frontier_robot_dict:
                v_corr = self.get_agent_position(fr, self.T - t)
                self.z[self.get_z_idx(fr, v_corr, t)] = 1
                self.graph_list.append(deepcopy(self.graph))

    ##SOLVER FUNCTIONS##

    def solve(self):

        self.prepare_problem()

        zvar = Variable(size=(self.T+1) * self.num_v * self.num_r,
                        start=0,
                        binary=True)

        fbarvar = Variable(size=(self.T+1) * self.num_r * len(self.dict_conn),
                           start=zvar.start + zvar.size,
                           binary=True)

        kvar = Variable(size=self.num_v * self.num_r,
                        start=zvar.size + fbarvar.size,
                        binary=True)

        self.vars = {'z': zvar, 'fbar': fbarvar, 'k': kvar}

        self.z = np.zeros(zvar.size)
        self.fbar = np.zeros(fbarvar.size)
        self.k = np.zeros(kvar.size)

        #Set start values
        for fr, v in product(self.frontier_robot_dict, self.graph.nodes):
            r = self.frontier_robot_dict[fr]
            if self.graph.nodes[v]['known']:
                self.k[self.get_k_idx(fr,v)] = 1
            if v == self.graph.agents[r]:
                self.z[self.get_z_idx(fr,v,0)] = 1

        self._solve()

        #generate trajectories
        self.traj = {}
        for t in range(self.T+1):
            for fr, r in self.frontier_robot_dict.items():
                v = self.get_agent_position(fr,t)
                self.traj[(r,t)] = v
            for r in self.graph.agents:
                if r not in self.frontier_robot_dict.values():
                    self.traj[(r,t)] = self.graph.agents[r]

    def _solve(self):
        for t in range(1,int(self.T/2)+1):
            for fr in self.frontier_robot_dict:
                if self.frontier_robot_dict[fr] not in self.static_agents and self.frontier_robot_dict[fr] in self.eagents:
                    v_next_alt = []
                    v_prev = self.get_agent_position(fr, t-1)
                    for v1, v2 in self.graph.tran_out_edges(v_prev):
                        if self.k[self.get_k_idx(fr, v2)] == 0:
                            v_next_alt.append(v2)
                    if len(v_next_alt) == 0:
                        v_next_alt.append(v_prev)
                    v_next = self.choose_fork(v_next_alt, t)
                    self.z[self.get_z_idx(fr,v_next,t)] = 1
                    self.k[self.get_k_idx(fr,v_next)] = 1
                    self.graph.nodes[v_next]['known'] = True
                else:
                    v_prev = self.get_agent_position(fr, t-1)
                    self.z[self.get_z_idx(fr,v_prev,t)] = 1
            self.share_data(t)
            self.graph_list.append(deepcopy(self.graph))
        self.set_return_path()

    def animate_solution(self, ANIM_STEP=30, filename='animation.mp4', labels=False):

        # Initiate plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.axis('off')

        #Setup position dictionary for node positions
        dict_pos = {n: (self.graph.nodes[n]['x'], self.graph.nodes[n]['y']) for n in self.graph}

        # Build dictionary robot,time -> position
        traj_x = {(r,t): np.array([self.graph.nodes[v]['x'], self.graph.nodes[v]['y']])
                  for (r,t), v in self.traj.items()}



        #Plot unknown IRM

        #Node color list
        node_color = np.full(len(self.graph.nodes), 'grey')
        coll_node = nx.draw_networkx_nodes(self.graph, dict_pos, ax=ax, nodelist = self.graph.nodes,
                               node_color='white', edgecolors=node_color, linewidths=1.0)
        coll_tedge = nx.draw_networkx_edges(self.graph, dict_pos, ax=ax, edgelist=list(self.graph.tran_edges()),
                               connectionstyle='arc', edge_color='grey')

        if labels:
            nx.draw_networkx_labels(self.graph, dict_pos, font_color='grey')

        # connectivity edges
        coll_cedge = nx.draw_networkx_edges(self.graph, dict_pos, ax=ax, edgelist=list(self.graph.conn_edges()),
                                            edge_color='grey')
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

            graph = self.graph_list[min(self.T, t)]
            #Update node 'known' color
            for n in graph.nodes:
                if graph.nodes[n]['known']:
                    node_color[n] = 'k'
            coll_node = nx.draw_networkx_nodes(self.graph, dict_pos, ax=ax, nodelist = self.graph.nodes,
                               node_color='white', edgecolors=node_color, linewidths=1.0)

            for n in graph.nodes:
                if graph.nodes[n]['known']:
                    nx.draw_networkx_labels(self.graph, dict_pos, labels={n:n}, font_color='black')

            #Update transition-edge 'known' color
            for i, (v1, v2) in enumerate(graph.tran_edges()):
                if graph.nodes[v1]['known'] == True and graph.nodes[v2]['known'] == True:
                    coll_tedge[i].set_color('black')

            #Update connectivity-edge 'known' color
            for i, (v1, v2) in enumerate(graph.conn_edges()):
                if graph.nodes[v1]['known'] == True and graph.nodes[v2]['known'] == True:
                    coll_cedge[i].set_color('black')

            # Update connectivity edge colors if there is flow information
            if 'fbar' in self.vars:
                for i, (v1, v2) in enumerate(graph.conn_edges()):
                    if graph.nodes[v1]['known'] == True and graph.nodes[v2]['known'] == True:
                        coll_cedge[i].set_color('black')
                    col_list = [colors[b_r] for b, b_r in enumerate(self.frontier_robot_dict)
                                if self.fbar[self.get_fbar_idx(b, v1, v2, min(self.T, t))] > 0.5]
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
