import time
from dataclasses import dataclass

import numpy as np
import networkx as nx
import scipy.sparse as sp

from colorama import Fore, Style

from graph_connectivity.optimization_wrappers import solve_ilp, Constraint
from graph_connectivity.graph import Graph

from graph_connectivity.constr_dyn import *
from graph_connectivity.constr_flow import *
from graph_connectivity.constr_powerset import *


from itertools import chain, combinations, product

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
        self.final_position = None
        self.src = None
        self.snk = None
        self.master = None
        self.std_frontier_reward = 100
        self.reward_dict = None

        # ILP setup
        self.dict_tran = None
        self.dict_conn = None
        self.dict_node = None
        self.dict_agent = None
        self.vars = None
        self.constraint = None
        self.obj = []

        # ILP solution
        self.solution = None
        self.traj = None

        self.conn = None   # { t : [(v00, v01, b0), (v10, v11, b1)] }
        self.tran = None   # { t : [(v00, v01, b0), (v10, v11, b1)] }

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

        if type(self.master) is not list:
            self.master = [self.master]

        #reset contraints
        self.constraint = Constraint()

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
        # Create dictionary for v -> k mapping for nodes
        self.dict_node = {v: k for k, v in enumerate(self.graph.nodes)}
        # Create agent dictionary
        self.dict_agent = {r: k for k, r in enumerate(self.graph.agents)}


        if not set(self.src) <= set(self.graph.agents.keys()):
            print((self.src,self.graph.agents.keys()))
            raise Exception("Invalid sources")

        if not set(self.snk) <= set(self.graph.agents.keys()):
            print((self.snk,self.graph.agents.keys()))
            raise Exception("Invalid sinks")

        if not set(self.graph.agents.values()) <= set(self.graph.nodes()):
            raise Exception("Invalid initial positions")

    def get_z_idx(self, r, v, t):
        R = self.dict_agent[r]
        k = self.dict_node[v]
        return self.vars['z'].start + np.ravel_multi_index((t,k,R), (self.T+1, self.num_v, self.num_r))

    def get_e_idx(self, i, j, t):
        k = self.dict_tran[(i, j)]
        idx = np.ravel_multi_index((t,k), (self.T, len(self.dict_tran)))
        return self.vars['e'].start + idx

    def get_y_idx(self, v):
        k = self.dict_node[v]
        return self.vars['y'].start + k

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

    def get_xf_idx(self, r, i, j, t):
        R = self.dict_agent[r]
        k = self.dict_tran[(i, j)]
        idx = np.ravel_multi_index((t,R,k), (self.T, len(self.graph.agents), len(self.dict_tran)))
        return self.vars['xf'].start + idx

    def get_m_idx(self, i, j, t):
        k = self.dict_tran[(i, j)]
        idx = np.ravel_multi_index((t,k), (self.T, len(self.dict_tran)))
        return self.vars['m'].start + idx

    def get_mbar_idx(self, i, j, t):
        k = self.dict_conn[(i, j)]
        idx = np.ravel_multi_index((t,k), (self.T+1, len(self.dict_conn)))
        return self.vars['mbar'].start + idx

    ##OBJECTIVE FUNCTION##

    def generate_powerset_objective(self, optimal):

        obj = np.zeros(self.num_vars)

        if optimal:

            # add user-defined rewards
            if self.reward_dict != None:
                for v, r in self.reward_dict.items():
                    obj[self.get_y_idx(v)] = -r

            # add frontier rewards
            for v in self.graph.nodes:
                if self.graph.nodes[v]['frontiers'] != 0:
                    obj[self.get_y_idx(v)] -= self.std_frontier_reward

            # add transition weights
            for e, t in product(self.graph.edges(data=True), range(self.T)):
                if e[2]['type'] == 'transition':
                    obj[self.get_e_idx(e[0], e[1], t)] = (1.01**t) * e[2]['weight']

        return obj

    def generate_flow_objective(self, optimal, frontier_reward = True):

        obj = np.zeros(self.num_vars)

        if optimal:

            # add user-defined rewards
            if self.reward_dict != None:
                for v, r in self.reward_dict.items():
                    obj[self.get_y_idx(v)] -= r

            # add frontier rewards
            if frontier_reward:
                for v in self.graph.nodes:
                    if self.graph.nodes[v]['frontiers'] != 0:
                        obj[self.get_y_idx(v)] -= self.std_frontier_reward

            # add transition weights
            for e, t, r in product(self.graph.edges(data=True), range(self.T), self.graph.agents):
                if e[2]['type'] == 'transition':
                    obj[self.get_xf_idx(r, e[0], e[1], t)] = (1.01**t) * e[2]['weight']

            #add connectivity weights
            if 'fbar' in self.vars:
                for b, e, t in product(range(len(self.min_src_snk)), self.graph.edges(data=True), range(self.T+1)):
                    if e[2]['type'] == 'connectivity':
                        obj[self.get_fbar_idx(b, e[0], e[1], t)] = (1.01**t) * e[2]['weight']

            #add master connectivity weights (prevents unnecessary passing of masterplan)
            if 'mbar' in self.vars:
                for e, t in product(self.graph.edges(data=True), range(self.T+1)):
                    if e[2]['type'] == 'connectivity':
                        obj[self.get_mbar_idx(e[0], e[1], t)] = (1.01**t) * e[2]['weight']

        return obj

    ##SOLVER FUNCTIONS##

    def cut_solution(self):
        t = self.T
        cut = True
        while cut and t>0:
            for r, v in product(self.graph.agents, self.graph.nodes):
                if abs(self.solution['x'][self.get_z_idx(r, v, t)] - self.solution['x'][self.get_z_idx(r, v, t-1)]) > 0.5:
                    cut = False
            if cut:
                t -= 1
        self.T = t

    def solve_powerset(self, optimal = False, solver=None, output=False, integer=True):

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
        self.constraint &= generate_powerset_dynamic_constraints(self)
        # Bridge z, e to x, xbar, yb
        self.constraint &= generate_powerset_bridge_constraints(self)
        # Connectivity constraints on x, xbar, yb
        self.constraint &= generate_connectivity_constraint_all(self)
        #Objective
        self.obj = self.generate_powerset_objective(optimal)

        print("Constraints setup time {:.2f}s".format(time.time() - t0))

        self._solve(solver=None, output=False, integer=True)

    def solve_adaptive(self, optimal = False, solver=None, output=False, integer=True):

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
        self.constraint &= generate_powerset_dynamic_constraints(self)
        # Bridge z, e to x, xbar, yb
        self.constraint &= generate_powerset_bridge_constraints(self)
        #Objective
        self.obj = self.generate_powerset_objective(optimal)

        print("Constraints setup time {:.2f}s".format(time.time() - t0))

        valid_solution = False
        while not valid_solution:
            self._solve(solver, output, integer)
            if self.solution['status'] == 'infeasible':
                break
            valid_solution, add_S = self.test_solution()
            self.constraint &= generate_connectivity_constraint(self, range(self.num_src), add_S)

    def solve_flow(self, master = False, connectivity = True, optimal = False,
                   solver=None, output=False, integer=True,
                   frontier_reward = True, verbose=False):

        self.prepare_problem()

        zvar = Variable(size=(self.T+1) * self.num_r * self.num_v,
                        start=0,
                        binary=True)
        xfvar = Variable(size=self.T * len(self.graph.agents) * len(self.dict_tran),
                        start=zvar.start + zvar.size,
                        binary=True)
        yvar = Variable(size=self.num_v,
                        start=xfvar.start + xfvar.size,
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

        self.vars = {'z': zvar, 'xf': xfvar, 'y': yvar, 'f': fvar, 'fbar': fbarvar, 'm': mvar, 'mbar': mbarvar}
        t0 = time.time()

        # Initial Constraints on z
        self.constraint &= generate_initial_constraints(self)
        # Dynamic Constraints on z, e
        self.constraint &= generate_flow_dynamic_constraints(self)
        # Bridge z, f, fbar to x, xbar, yb
        self.constraint &= generate_flow_bridge_constraints(self)
        # Flow connectivity constraints on z, e, f, fbar
        if connectivity:
            self.constraint &= generate_flow_connectivity_constraints(self)
        # Flow master constraints on z, e, m, mbar
        if master:
            self.constraint &= generate_flow_master_constraints(self)
        # Flow objective
        self.obj = self.generate_flow_objective(optimal, frontier_reward)

        if verbose:
            print("Constraints setup time {:.2f}s".format(time.time() - t0))

        self._solve(solver=None, output=False, integer=True, verbose=verbose)

    def diameter_solve_flow(self, master = False, connectivity = True,
                            optimal = False, solver=None, output=False,
                            integer=True, frontier_reward = True,
                            verbose=False):

        D = nx.diameter(self.graph)
        Rp = len(set(v for r, v in self.graph.agents.items()))
        V = self.graph.number_of_nodes()

        T = int(max(D/2, D - int(Rp/2)))

        feasible_solution = False

        if verbose:
            print(Fore.GREEN + "Solving flow [R={}, V={}, Et={}, Ec={}, static={}]"
                  .format(len(self.graph.agents),
                          self.graph.number_of_nodes(),
                          self.graph.number_of_tran_edges(),
                          self.graph.number_of_conn_edges(),
                          self.static_agents)
                  + Style.RESET_ALL )

        while not feasible_solution:
            self.T = T

            if verbose:
                print("Trying" + Style.BRIGHT + " T={}".format(self.T)
                      + Style.RESET_ALL)

            #Solve
            self.solve_flow(master, connectivity, optimal, solver,
                            output, integer, frontier_reward,
                            verbose=verbose)

            if self.solution['status'] is not 'infeasible':
                feasible_solution = True

            T += 1

    def linear_search_solve_flow(self, master = False, connectivity = True, optimal = False, solver=None, output=False, integer=True, frontier_reward = True):

        T = 0
        feasible_solution = False
        while not feasible_solution:

            self.T = T

            #Solve
            self.solve_flow(master, connectivity, optimal, solver, output, integer, frontier_reward)

            if self.solution['status'] is not 'infeasible':
                feasible_solution = True

            T += 1

    def _solve(self, solver=None, output=False, integer=True, verbose=False):

        obj = self.obj

        J_int = sum([list(range(var.start, var.start + var.size))
                    for var in self.vars.values() if not var.binary], [])
        J_bin = sum([list(range(var.start, var.start + var.size))
                    for var in self.vars.values() if var.binary], [])

        if verbose:
            print("NumConst: {} ({} bin, {} int), NumVar: {}"
                  .format(self.num_vars, len(J_bin), len(J_int),
                         self.constraint.A_eq.shape[0]+self.constraint.A_iq.shape[0])
                  )

        # Solve it
        t0 = time.time()
        self.solution = solve_ilp(obj, self.constraint, J_int, J_bin, solver, output)

        if verbose:
            print("Solver time {:.2f}s".format(time.time() - t0))

        if self.solution['status'] == 'infeasible':
            if verbose:
                print("Problem infeasible")

            self.traj = {}
            self.conn = {}
            self.tran = {}
        else:
            #cut static part of solution
            self.cut_solution()

            # save info
            self.traj = {}
            self.conn = {t : set() for t in range(self.T+1)}
            self.tran = {t : set() for t in range(self.T)}

            for r, v, t in product(self.graph.agents, self.graph.nodes, range(self.T+1)):
                if self.solution['x'][self.get_z_idx(r, v, t)] > 0.5:
                    self.traj[(r,t)] = v

            if 'fbar' in self.vars:
                for t, b, (v1, v2) in product(range(self.T+1), range(self.num_min_src_snk),
                                              self.graph.conn_edges()):
                    if self.solution['x'][self.get_fbar_idx(b, v1, v2, t)] > 0.5:
                        b_r = self.src[b] if len(self.src) <= len(self.snk) else self.snk[b]
                        self.conn[t].add((v1, v2, b_r))

            if 'mbar' in self.vars:
                for t, (v1, v2) in product(range(self.T+1), self.graph.conn_edges()):
                    if self.solution['x'][self.get_mbar_idx(v1, v2, t)] > 0.5:
                        self.conn[t].add((v1, v2, tuple(self.master)))


            if 'f' in self.vars:
                for t, b, (v1, v2) in product(range(self.T), range(self.num_min_src_snk),
                                              self.graph.tran_edges()):
                    if self.solution['x'][self.get_f_idx(b, v1, v2, t)] > 0.5:
                        b_r = self.src[b] if len(self.src) <= len(self.snk) else self.snk[b]
                        self.tran[t].add((v1, v2, b_r))

            if 'm' in self.vars:
                for t, (v1, v2) in product(range(self.T), self.graph.tran_edges()):
                    if self.solution['x'][self.get_m_idx(v1, v2, t)] > 0.5:
                        self.tran[t].add((v1, v2, tuple(self.master)))


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
                pre_Svt_transition = self.graph.pre_tran_vt([S[idx]])
                pre_Svt_connectivity = self.graph.pre_conn_vt([S[idx]])
                for v0, v1, t in pre_Svt_transition:
                    occupied = False
                    for robot in self.graph.agents:
                        z_idx = self.get_z_idx(robot, v0, t)
                        if solution[z_idx] == 1:
                            occupied = True
                    if occupied == True and (v0, t) not in S:
                        S.append((v0, t))
                for v0, v1, t in pre_Svt_connectivity:
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
