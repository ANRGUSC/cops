import time
from dataclasses import dataclass

import numpy as np
import networkx as nx
import scipy.sparse as sp

from graph_connectivity.optimization_wrappers import solve_ilp, Constraint
from graph_connectivity.graph import Graph

from graph_connectivity.constr_dyn import *
from graph_connectivity.constr_flow import *
from graph_connectivity.constr_powerset import *

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from itertools import chain, combinations, product
from networkx.drawing.nx_agraph import to_agraph

@dataclass
class Variable(object):
    start: int
    size: int
    binary: bool = False

class ConnectivityProblem(object):

    def __init__(self):
        # Problem definition
        self.graph = None                    #Graph
        self.T = None                        #Time horizon
        self.static_agents = None
        self.src = None
        self.snk = None
        self.master = None

        # ILP setup
        self.dict_tran = None
        self.dict_conn = None
        self.vars = None
        self.constraint = Constraint()

        # ILP solution
        self.solution = None
        self.trajectories = None

    ##PROPERTIES##

    @property
    def num_src(self):
        return len(self.src)

    @property
    def num_snk(self):
        return len(self.snk)

    @property
    def min_src_snk(self):
        return self.src if len(self.src) <= len(self.snk) else self.snk

    @property
    def num_min_src_snk(self):
        return min(self.num_src, self.num_snk)

    @property
    def num_r(self):
        return len(self.graph.agents)

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

        if self.src is None: # all-to-sinks connectivity if sources undefined
            self.src = self.graph.agents.keys()

        if self.snk is None: # sources-to-all connectivity if sinks undefined
            self.snk = self.graph.agents.keys()

        # Create dictionaries for (i,j)->k mapping for edges
        self.dict_tran = {(i,j): k for k, (i,j) in enumerate(self.graph.tran_edges())}
        self.dict_conn = {(i,j): k for k, (i,j) in enumerate(self.graph.conn_edges())}

        if not set(self.src) <= set(self.graph.agents.keys()):
            raise Exception("Invalid sources")

        if not set(self.snk) <= set(self.graph.agents.keys()):
            raise Exception("Invalid sinks")

        if not set(self.graph.agents.values()) <= set(self.graph.nodes()):
            raise Exception("Invalid initial positions")

    def get_z_idx(self, r, v, t):
        return self.vars['z'].start + np.ravel_multi_index((t,v,r), (self.T+1, self.num_v, self.num_r))

    def get_e_idx(self, i, j, t):
        k = self.dict_tran[(i, j)]
        idx = np.ravel_multi_index((t,k), (self.T, len(self.dict_tran)))
        return self.vars['e'].start + idx

    def get_y_idx(self, v):
        return self.vars['y'].start + v

    def get_yb_idx(self, b, v, t):
        idx = np.ravel_multi_index((t,b,v), (self.T+1, self.num_src, self.num_v))
        return self.vars['yb'].start + idx

    def get_f_idx(self, b, i, j, t):
        k = self.dict_tran[(i, j)]
        idx = np.ravel_multi_index((t,b,k), (self.T, self.num_min_src_snk, len(self.dict_tran)))
        return self.vars['f'].start + idx

    def get_fbar_idx(self, b, i, j, t):
        k = self.dict_conn[(i, j)]
        idx = np.ravel_multi_index((t,b,k), (self.T+1, self.num_min_src_snk, len(self.dict_conn)))
        return self.vars['fbar'].start + idx

    def get_x_idx(self, b, i, j, t):
        k = self.dict_tran[(i, j)]
        idx = np.ravel_multi_index((t,b,k), (self.T, self.num_src, len(self.dict_tran)))
        return self.vars['x'].start + idx

    def get_xbar_idx(self, b, i, j, t):
        k = self.dict_conn[(i, j)]
        idx = np.ravel_multi_index((t,b,k), (self.T+1, self.num_src, len(self.dict_conn)))
        return self.vars['xbar'].start + idx

    def get_m_idx(self, i, j, t):
        k = self.dict_tran[(i, j)]
        idx = np.ravel_multi_index((t,k), (self.T, len(self.dict_tran)))
        return self.vars['m'].start + idx

    def get_mbar_idx(self, i, j, t):
        k = self.dict_conn[(i, j)]
        idx = np.ravel_multi_index((t,k), (self.T+1, len(self.dict_conn)))
        return self.vars['mbar'].start + idx



    ##SOLVER FUNCTIONS##

    def solve_powerset(self, solver=None, output=False, integer=True):

        if self.snk is not None:
            print("WARNING: sinks not implemented for solve_powerset, defaulting to all sinks")
            self.snk = None

        self.prepare_problem()

        zvar = Variable(size=(self.T+1) * self.num_r * self.num_v,
                        start=0,
                        binary=True)
        evar = Variable(size=self.T * len(self.dict_tran),
                        start=zvar.start + zvar.size,
                        binary=False)
        ybvar = Variable(size=(self.T+1) * self.num_src * self.num_v,
                        start=evar.start + evar.size,
                        binary=True)
        xvar = Variable(size=self.T * self.num_src * len(self.dict_tran),
                        start=ybvar.start + ybvar.size,
                        binary=True)
        xbarvar = Variable(size=(self.T+1) * self.num_src * len(self.dict_conn),
                           start=xvar.start + xvar.size,
                           binary=True)

        self.vars = {'z': zvar, 'e': evar, 'yb': ybvar, 'x': xvar, 'xbar': xbarvar}
        t0 = time.time()

        # Initial constraints on z
        self.constraint &= generate_initial_constraints(self)
        # Dynamic constraints on z, e
        self.constraint &= generate_dynamic_constraints(self)
        # Bridge z, e to x, xbar, yb
        self.constraint &= generate_bridge_constraints(self)
        # Connectivity constraints on x, xbar, yb
        self.constraint &= generate_connectivity_constraint_all(self)

        print("Constraints setup time {:.2f}s".format(time.time() - t0))

        self._solve(solver=None, output=False, integer=True)

    def solve_adaptive(self, solver=None, output=False, integer=True):

        self.prepare_problem()

        zvar = Variable(size=(self.T+1) * self.num_r * self.num_v,
                        start=0,
                        binary=True)
        evar = Variable(size=self.T * len(self.dict_tran),
                        start=zvar.start + zvar.size,
                        binary=False)
        ybvar = Variable(size=(self.T+1) * self.num_src * self.num_v,
                        start=evar.start + evar.size,
                        binary=True)
        xvar = Variable(size=self.T * self.num_src * len(self.dict_tran),
                        start=ybvar.start + ybvar.size,
                        binary=True)
        xbarvar = Variable(size=(self.T+1) * self.num_src * len(self.dict_conn),
                           start=xvar.start + xvar.size,
                           binary=True)

        self.vars = {'z': zvar, 'e': evar, 'yb': ybvar, 'x': xvar, 'xbar': xbarvar}
        t0 = time.time()

        # Initial constraints on z
        self.constraint &= generate_initial_constraints(self)
        # Dynamic constraints on z, e
        self.constraint &= generate_dynamic_constraints(self)
        # Bridge z, e to x, xbar, yb
        self.constraint &= generate_bridge_constraints(self)

        print("Constraints setup time {:.2f}s".format(time.time() - t0))

        valid_solution = False
        while not valid_solution:
            self._solve(solver, output, integer)
            if self.solution['status'] == 'infeasible':
                break
            valid_solution, add_S = self.test_solution()
            self.constraint &= generate_connectivity_constraint(self, range(self.num_src), add_S)

    def solve_flow(self, solver=None, output=False, integer=True):

        self.prepare_problem()

        zvar = Variable(size=(self.T+1) * self.num_r * self.num_v,
                        start=0,
                        binary=True)
        evar = Variable(size=self.T * len(self.dict_tran),
                        start=zvar.start + zvar.size,
                        binary=False)
        yvar = Variable(size=self.num_v,
                        start=evar.start + evar.size,
                        binary=True)
        fvar = Variable(size=self.T * self.num_min_src_snk * len(self.dict_tran),
                        start=yvar.start + yvar.size,
                        binary=False)
        fbarvar = Variable(size=(self.T+1) * self.num_min_src_snk * len(self.dict_conn),
                           start=fvar.start + fvar.size,
                           binary=False)
        mvar = Variable(size=self.T * len(self.dict_tran),
                        start=fbarvar.start + fbarvar.size,
                        binary=False)
        mbarvar = Variable(size=(self.T+1) * len(self.dict_conn),
                           start=mvar.start + mvar.size,
                           binary=False)

        self.vars = {'z': zvar, 'e': evar, 'y': yvar, 'f': fvar, 'fbar': fbarvar, 'm': mvar, 'mbar': mbarvar}
        t0 = time.time()

        # Initial Constraints on z
        self.constraint &= generate_initial_constraints(self)
        # Dynamic Constraints on z, e
        self.constraint &= generate_dynamic_constraints(self)
        # Constraints on y for optimization
        self.constraint &= generate_optim_constraints(self)
        # Flow constraints on z, e, f, fbar
        self.constraint &= generate_flow_constraints(self)

        print("Constraints setup time {:.2f}s".format(time.time() - t0))

        self._solve(solver=None, output=False, integer=True)

    def _solve(self, solver=None, output=False, integer=True):
        obj = np.zeros(self.num_vars)

        J_int = sum([list(range(var.start, var.start + var.size))
                    for var in self.vars.values() if not var.binary], [])
        J_bin = sum([list(range(var.start, var.start + var.size))
                    for var in self.vars.values() if var.binary], [])

        print("Number of variables: {} ({} bin, {} int)".format(self.num_vars, len(J_bin), len(J_int)))
        print("Number of constraints: {}".format(self.constraint.A_eq.shape[0]+self.constraint.A_iq.shape[0]))

        # Solve it
        t0 = time.time()
        self.solution = solve_ilp(obj, self.constraint, J_int, J_bin, solver, output)
        print("Solver time {:.2f}s".format(time.time() - t0))

        self.trajectories = {}

        if self.solution['status'] == 'infeasible':
            print("Problem infeasible")
        else:
            for r, v, t in product(self.graph.agents, self.graph.nodes, range(self.T+1)):
                if self.solution['x'][self.get_z_idx(r, v, t)] > 0.5:
                    self.trajectories[(r,t)] = v

    ##GRAPH HELPER FUNCTIONS##

    def get_time_augmented_id(self, n, t):
        return np.ravel_multi_index((t,n), (self.T+1, self.num_v))

    def get_time_augmented_n_t(self, id):
        t, n = np.unravel_index(id, (self.T+1, self.num_v))
        return n,t

    def powerset_exclude_agent(self, b):
        time_augmented_nodes = []
        for t in range(self.T+1):
            for v in self.graph.nodes:
                time_augmented_nodes.append(self.get_time_augmented_id(v,t))
        s = list(time_augmented_nodes)
        s.remove(self.get_time_augmented_id(self.graph.agents[b],0))
        return(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))

    def test_solution(self):
        solution = self.solution['x']
        add_S = []
        for r in self.snk:
            S = []
            #Find end position of agent r and set as V
            for v in self.graph.nodes:
                z_idx = self.get_z_idx(r, v, self.T)
                if solution[z_idx] == 1:
                    S.append((v, self.T))

            # Backwards reachability
            idx = 0
            while idx < len(S):
                pre_S_transition = self.graph.get_pre_S_transition([S[idx]])
                pre_S_connectivity = self.graph.get_pre_S_connectivity([S[idx]])
                for v0, v1, t in pre_S_transition:
                    occupied = False
                    for robot in self.graph.agents:
                        z_idx = self.get_z_idx(robot, v0, t)
                        if solution[z_idx] == 1:
                            occupied = True
                    if occupied == True and (v0, t) not in S:
                        S.append((v0, t))
                for v0, v1, t in pre_S_connectivity:
                    occupied = False
                    for robot in self.graph.agents:
                        z_idx = self.get_z_idx(robot, v0, t)
                        if solution[z_idx] == 1:
                            occupied = True
                    if occupied == True and (v0, t) not in S:
                        S.append((v0, t))
                idx +=1

            valid = True
            for b in self.src:
                if (self.graph.agents[b],0) not in S:
                    valid = False
            if valid == False:
                for b in self.src:
                    if (self.graph.agents[b],0) in S:
                        S.remove((self.graph.agents[b],0))
                add_S.append(S)

        if len(add_S) != 0:
            return False, add_S
        else:
            return True, add_S

    ##VISUALIZATION FUNCTIONS##

    def plot_solution(self):
        if self.solution is None:
            print("No solution found: call solve() to obtain solution")
        else:
            g = self.graph
            #create time-augmented graph G
            G = Graph()
            #space between timesteps
            space = 2

            #loop of graph and add nodes to time-augmented graph
            for t in range(self.T+1):
                for n in g.nodes:
                    id = self.get_time_augmented_id(n, t)
                    G.add_node(id)
                    G.nodes[id]['x'] = g.nodes[n]['x'] + space*t
                    G.nodes[id]['y'] = g.nodes[n]['y']
                    G.nodes[id]['agents'] = []

            #loop of graph and add edges to time-augmented graph
            i = 0
            for n in g.nodes:
                for t in range(self.T+1):
                    for edge in g.out_edges(n, data = True):
                        if edge[2]['type'] == 'transition' and t < self.T:
                            out_id = self.get_time_augmented_id(n, t)
                            in_id = self.get_time_augmented_id(edge[1], t+1)
                            G.add_edge(out_id , in_id, type='transition')
                        elif edge[2]['type'] == 'connectivity':
                            out_id = self.get_time_augmented_id(n, t)
                            in_id = self.get_time_augmented_id(edge[1], t)
                            G.add_edge(out_id , in_id, type='connectivity')


            #set 'pos' attribute for accurate position in plot (use ! in end for accurate positioning)
            for n in G:
                G.nodes[n]['pos'] =  ",".join(map(str,[G.nodes[n]['x'],G.nodes[n]['y']])) + '!'

            #Set general edge-attributes
            G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}

            #Set individual attributes based on solution
            for (r,t), v in self.trajectories.items():
                G.nodes[self.get_time_augmented_id(v, t)]['agents'].append(r)

            for v in G.nodes:
                G.nodes[v]['number_of_agents'] = len(G.nodes[v]['agents'])
                G.nodes[v]['label'] = " ".join(map(str,(G.nodes[v]['agents'])))

            for n in G:
                if G.nodes[n]['number_of_agents']!=0:
                    G.nodes[n]['color'] = 'black'
                    G.nodes[n]['fillcolor'] = 'red'
                    G.nodes[n]['style'] = 'filled'
                else:
                    G.nodes[n]['color'] = 'black'
                    G.nodes[n]['fillcolor'] = 'white'
                    G.nodes[n]['style'] = 'filled'
                for nbr in G[n]:
                    for edge in G[n][nbr]:
                        if G[n][nbr][edge]['type']=='connectivity':
                            if len(G.nodes[n]['agents']) != 0 and len(G.nodes[nbr]['agents'])!= 0:
                                G[n][nbr][edge]['color']='black'
                                G[n][nbr][edge]['style']='dashed'
                            else:
                                G[n][nbr][edge]['color']='grey'
                                G[n][nbr][edge]['style']='dashed'
                        else:
                            G[n][nbr][edge]['color']='grey'
                            G[n][nbr][edge]['style']='solid'
            for n in G:
                for nbr in G[n]:
                    for edge in G[n][nbr]:
                        if G[n][nbr][edge]['type']=='transition':
                            for a in G.nodes[n]['agents']:
                                if a in G.nodes[nbr]['agents']:
                                    G[n][nbr][edge]['color']='black'
                                    G[n][nbr][edge]['style']='solid'


            #Plot/save graph
            A = to_agraph(G)
            A.layout()
            A.draw('solution.png')

    def animate_solution(self, ANIM_STEP=30, filename='animation.mp4'):

        # Initiate plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.axis('off')

        #Setup position dictionary for node positions
        dict_pos = {n: (self.graph.nodes[n]['x'], self.graph.nodes[n]['y']) for n in self.graph}

        # Build dictionary robot,time -> position
        traj_x = {(r,t): np.array([self.graph.nodes[v]['x'], self.graph.nodes[v]['y']])
                  for (r,t), v in self.trajectories.items()}

        # FIXED STUFF
        nx.draw_networkx_nodes(self.graph, dict_pos, ax=ax,
                               node_color='white', edgecolors='black', linewidths=1.0)
        nx.draw_networkx_edges(self.graph, dict_pos, ax=ax, edgelist=list(self.graph.tran_edges()),
                               connectionstyle='arc', edge_color='black')

        # VARIABLE STUFF
        # connectivity edges
        coll_cedge = nx.draw_networkx_edges(self.graph, dict_pos, ax=ax, edgelist=list(self.graph.conn_edges()),
                                            edge_color='black')
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
            if 'fbar' in self.vars:
                for i, (v1, v2) in enumerate(self.graph.conn_edges()):
                    coll_cedge[i].set_color('black')
                    col_list = [colors[b_r] for b, b_r in enumerate(self.min_src_snk)
                                if self.solution['x'][self.get_fbar_idx(b, v1, v2, min(self.T, t))] > 0.5]
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
