import time
from dataclasses import dataclass
from copy import deepcopy
from itertools import chain, combinations, product

import numpy as np
import networkx as nx

from colorama import Fore, Style

from cops.optimization_wrappers import solve_ilp, Constraint
from cops.graph import Graph

from cops.constr_dyn import (
    generate_powerset_dynamic_constraints,
    generate_flow_dynamic_constraints,
    generate_initial_constraints,
)
from cops.constr_flow import (
    generate_flow_bridge_constraints,
    generate_flow_connectivity_constraints,
    generate_flow_master_constraints,
)
from cops.constr_powerset import (
    generate_powerset_bridge_constraints,
    generate_connectivity_constraint,
    generate_connectivity_constraint_all,
)
from cops.constr_cluster import constraint_static_master


@dataclass
class Variable(object):
    start: int
    size: int
    binary: bool = False


class AbstractConnectivityProblem(object):
    def __init__(self, **kwargs):

        # BASIC PROBLEM DEFINITON
        self.graph = None  # mobility-communication graph with initial conditions

        self.static_agents = None  # set(r) of agents that don't move
        self.big_agents = None  # set(r) of agents that can't pass each other
        self.final_position = None  # dict(r: v) of constraints on final position

        # MASTER CONSTRAINTS
        self.master = None  # list(r) of master agents

        # OBJECTIVE FUNCTION
        self.eagents = None  # set(r) of agents that can explore frontiers
        self.frontier_reward = 100  # reward to end at frontier node
        self.frontier_reward_decay = 0.4  # decay factor for additional robots at node
        self.reward_dict = None  # dict(v: n) user-defined additional rewards

        if "graph" in kwargs:
            self.graph = kwargs["graph"]
        if "static_agents" in kwargs:
            self.static_agents = kwargs["static_agents"]
        if "big_agents" in kwargs:
            self.big_agents = kwargs["big_agents"]
        if "final_position" in kwargs:
            self.final_position = kwargs["final_position"]
        if "master" in kwargs:
            self.master = kwargs["master"]
        if "eagents" in kwargs:
            self.eagents = kwargs["eagents"]
        if "frontier_reward" in kwargs:
            self.frontier_reward = kwargs["frontier_reward"]
        if "frontier_reward_decay" in kwargs:
            self.frontier_reward_decay = kwargs["frontier_reward_decay"]
        if "reward_dict" in kwargs:
            self.reward_dict = kwargs["reward_dict"]

        # STORED SOLUTION
        self.T_sol = None  # length of solution
        self.traj = None  # dict(r,t: v) of robot positions
        self.conn = None  # dict(t: set(v1,v2,b)) of flow over communication edges
        self.tran = None  # dict(t: set(v1,v2,b)) of flow over transition edges

    def prepare_problem(self):

        if self.graph is None:
            raise Exception("Can not solve problem without 'graph'")

        if self.static_agents is None:
            self.static_agents = []

        if self.big_agents is None:
            self.big_agents = []

        if self.eagents is None:
            self.eagents = [r for r in self.graph.agents]


class ConnectivityProblem(AbstractConnectivityProblem):
    def __init__(self, **kwargs):
        super(ConnectivityProblem, self).__init__(**kwargs)

        # BASIC PROBLEM DEFINITON
        self.T = None  # time horizon

        # CONNECTIVITY
        self.src = None  # set(r) of source agents
        self.snk = None  # set(r) of sink agents
        self.always_src = False  # if true, always use source->sink type constraint

        self.reward_demand = 0.4  # fraction of total reward demanded
        self.max_reward_demand_iter = 5 # max number of iterations to find a better solutions
        self.extra_constr = None  # additional constraints

        if "src" in kwargs:
            self.src = kwargs["src"]
        if "snk" in kwargs:
            self.snk = kwargs["snk"]
        if "always_src" in kwargs:
            self.always_src = kwargs["always_src"]
        if "reward_demand" in kwargs:
            self.reward_demand = kwargs["reward_demand"]
        if "extra_constr" in kwargs:
            self.extra_constr = kwargs["extra_constr"]

        ##########################
        #### INTERNAL MEMORY #####
        ##########################

        # ILP SETUP
        self.vars = None
        self.dict_tran = None
        self.dict_conn = None
        self.dict_node = None
        self.dict_agent = None

    ##PROPERTIES##
    @property
    def num_src(self):
        return len(self.src)

    @property
    def num_snk(self):
        return len(self.snk)

    @property
    def min_src_snk(self):
        if self.always_src:
            return self.src
        return self.src if len(self.src) <= len(self.snk) else self.snk

    @property
    def num_min_src_snk(self):
        if self.always_src:
            return self.num_src
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

        super(ConnectivityProblem, self).prepare_problem()

        # reset contraints
        constraint = Constraint()

        if self.T is None:
            raise Exception("Can not solve problem without horizon 'T'")

        if self.src is None:  # all-to-sinks connectivity if sources undefined
            self.src = list(self.graph.agents.keys())

        if self.snk is None:  # sources-to-all connectivity if sinks undefined
            self.snk = list(self.graph.agents.keys())

        if not set(self.src) <= set(self.graph.agents.keys()):
            print((self.src, self.graph.agents.keys()))
            raise Exception("Invalid sources")

        if not set(self.snk) <= set(self.graph.agents.keys()):
            print((self.snk, self.graph.agents.keys()))
            raise Exception("Invalid sinks")

        if not set(self.graph.agents.values()) <= set(self.graph.nodes()):
            raise Exception("Invalid initial positions")

        # redefine master as a list to enable multiple masters
        if type(self.master) is not list:
            self.master = [self.master]

        # Create dictionaries for (i,j)->k mapping for edges
        self.dict_tran = {(i, j): k for k, (i, j) in enumerate(self.graph.tran_edges())}
        self.dict_conn = {(i, j): k for k, (i, j) in enumerate(self.graph.conn_edges())}
        # Create dictionary for v -> k mapping for nodes
        self.dict_node = {v: k for k, v in enumerate(self.graph.nodes)}
        # Create agent dictionary
        self.dict_agent = {r: k for k, r in enumerate(self.graph.agents)}

    def get_z_idx(self, r, v, t):
        R = self.dict_agent[r]
        k = self.dict_node[v]
        return self.vars["z"].start + np.ravel_multi_index(
            (t, k, R), (self.T + 1, self.num_v, self.num_r)
        )

    def get_e_idx(self, i, j, t):
        k = self.dict_tran[(i, j)]
        idx = np.ravel_multi_index((t, k), (self.T, len(self.dict_tran)))
        return self.vars["e"].start + idx

    def get_y_idx(self, v, k):
        V = self.dict_node[v]
        K = k - 1
        idx = np.ravel_multi_index((V, K), (len(self.dict_node), self.num_r))
        return self.vars["y"].start + idx

    def get_yb_idx(self, b, v, t):
        idx = np.ravel_multi_index((t, b, v), (self.T + 1, self.num_src, self.num_v))
        return self.vars["yb"].start + idx

    def get_f_idx(self, b, i, j, t):
        k = self.dict_tran[(i, j)]
        idx = np.ravel_multi_index(
            (t, b, k), (self.T, self.num_min_src_snk, len(self.dict_tran))
        )
        return self.vars["f"].start + idx

    def get_fbar_idx(self, b, i, j, t):
        k = self.dict_conn[(i, j)]
        idx = np.ravel_multi_index(
            (t, b, k), (self.T + 1, self.num_min_src_snk, len(self.dict_conn))
        )
        return self.vars["fbar"].start + idx

    def get_x_idx(self, b, i, j, t):
        k = self.dict_tran[(i, j)]
        idx = np.ravel_multi_index(
            (t, b, k), (self.T, self.num_src, len(self.dict_tran))
        )
        return self.vars["x"].start + idx

    def get_xbar_idx(self, b, i, j, t):
        k = self.dict_conn[(i, j)]
        idx = np.ravel_multi_index(
            (t, b, k), (self.T + 1, self.num_src, len(self.dict_conn))
        )
        return self.vars["xbar"].start + idx

    def get_xf_idx(self, r, i, j, t):
        R = self.dict_agent[r]
        k = self.dict_tran[(i, j)]
        idx = np.ravel_multi_index(
            (t, R, k), (self.T, len(self.graph.agents), len(self.dict_tran))
        )
        return self.vars["xf"].start + idx

    def get_m_idx(self, i, j, t):
        k = self.dict_tran[(i, j)]
        idx = np.ravel_multi_index((t, k), (self.T, len(self.dict_tran)))
        return self.vars["m"].start + idx

    def get_mbar_idx(self, i, j, t):
        k = self.dict_conn[(i, j)]
        idx = np.ravel_multi_index((t, k), (self.T + 1, len(self.dict_conn)))
        return self.vars["mbar"].start + idx

    ##OBJECTIVE FUNCTION##

    def generate_powerset_objective(self, add_frontier_rewards):

        obj = np.zeros(self.num_vars)

        # add user-defined rewards
        if self.reward_dict != None:
            for t, r in product(range(self.T + 1), self.graph.agents):
                for v, R in self.reward_dict.items():
                    obj[self.get_z_idx(r, v, t)] = -(0.9 ** t) * R

        # add frontier rewards
        if add_frontier_rewards:
            for t, r, v in product(
                range(self.T + 1), self.graph.agents, self.graph.nodes
            ):
                if (
                    "frontiers" in self.graph.nodes[v]
                    and self.graph.nodes[v]["frontiers"] != 0
                ):
                    obj[self.get_z_idx(r, v, t)] -= (0.9 ** t) * self.frontier_reward

        # add transition weights
        for e, t in product(self.graph.edges(data=True), range(self.T)):
            if e[2]["type"] == "transition":
                obj[self.get_e_idx(e[0], e[1], t)] = (1.01 ** t) * e[2]["weight"]

        return obj

    def generate_flow_objective(self, add_frontier_rewards):

        obj = np.zeros(self.num_vars)

        # add user-defined rewards
        if self.reward_dict != None:
            for v, r in self.reward_dict.items():
                obj[self.get_y_idx(v, 1)] -= r

        # add frontier rewards
        if add_frontier_rewards:
            max_weight = 0
            for v in self.graph.nodes:
                if (
                    "frontiers" in self.graph.nodes[v]
                    and self.graph.nodes[v]["frontiers"] > max_weight
                ):
                    max_weight = self.graph.nodes[v]["frontiers"]
            for v in self.graph.nodes:
                if (
                    "frontiers" in self.graph.nodes[v]
                    and self.graph.nodes[v]["frontiers"] > 0
                ):
                    for k in range(1, self.num_r + 1):
                        obj[self.get_y_idx(v, k)] -= (
                            self.frontier_reward_decay ** (k - 1)
                        ) * self.frontier_reward * (
                        self.graph.nodes[v]["frontiers"] / max_weight)

        # add transition weights
        for e, t, r in product(
            self.graph.edges(data=True), range(self.T), self.graph.agents
        ):
            if e[2]["type"] == "transition":
                obj[self.get_xf_idx(r, e[0], e[1], t)] = (1.01 ** t) * e[2]["weight"]

        # add regular communication weights
        if "fbar" in self.vars:
            for b, e, t in product(
                range(len(self.min_src_snk)),
                self.graph.edges(data=True),
                range(self.T + 1),
            ):
                if e[2]["type"] == "connectivity":
                    obj[self.get_fbar_idx(b, e[0], e[1], t)] = (1.01 ** t) * e[2][
                        "weight"
                    ]

        # add master communication weights
        if "mbar" in self.vars:
            for e, t in product(self.graph.edges(data=True), range(self.T + 1)):
                if e[2]["type"] == "connectivity":
                    obj[self.get_mbar_idx(e[0], e[1], t)] = (1.01 ** t) * e[2]["weight"]

        return obj

    ##SOLVER FUNCTIONS##

    def cut_solution(self, solution):
        t = self.T
        cut = True
        while cut and t > 0:
            for r, v in product(self.graph.agents, self.graph.nodes):
                if (
                    abs(
                        solution["x"][self.get_z_idx(r, v, t)]
                        - solution["x"][self.get_z_idx(r, v, t - 1)]
                    )
                    > 0.5
                ):
                    cut = False
            if cut:
                t -= 1
        self.T_sol = t

        return solution

    def solve_powerset(self, **kwargs):

        if self.snk is not None:
            print(
                "WARNING: sinks not implemented for solve_powerset, defaulting to all sinks"
            )
            self.snk = None

        self.prepare_problem()

        zvar = Variable(
            size=(self.T + 1) * self.num_r * self.num_v, start=0, binary=True
        )
        evar = Variable(
            size=self.T * len(self.dict_tran),
            start=zvar.start + zvar.size,
            binary=False,
        )
        ybvar = Variable(
            size=(self.T + 1) * self.num_src * self.num_v,
            start=evar.start + evar.size,
            binary=True,
        )
        xvar = Variable(
            size=self.T * self.num_src * len(self.dict_tran),
            start=ybvar.start + ybvar.size,
            binary=True,
        )
        xbarvar = Variable(
            size=(self.T + 1) * self.num_src * len(self.dict_conn),
            start=xvar.start + xvar.size,
            binary=True,
        )

        self.vars = {"z": zvar, "e": evar, "yb": ybvar, "x": xvar, "xbar": xbarvar}
        t0 = time.time()

        # Initial constraints on z
        constraint = generate_initial_constraints(self)
        # Dynamic constraints on z, e
        constraint &= generate_powerset_dynamic_constraints(self)
        # Bridge z, e to x, xbar, yb
        constraint &= generate_powerset_bridge_constraints(self)
        # Connectivity constraints on x, xbar, yb
        constraint &= generate_connectivity_constraint_all(self)
        # Objective
        obj = self.generate_powerset_objective(add_frontier_rewards=True)

        print("Constraints setup time {:.2f}s".format(time.time() - t0))

        self._solve(obj, constraint, **kwargs)

    def solve_adaptive(self, **kwargs):

        self.prepare_problem()

        zvar = Variable(
            size=(self.T + 1) * self.num_r * self.num_v, start=0, binary=True
        )
        evar = Variable(
            size=self.T * len(self.dict_tran),
            start=zvar.start + zvar.size,
            binary=False,
        )
        ybvar = Variable(
            size=(self.T + 1) * self.num_src * self.num_v,
            start=evar.start + evar.size,
            binary=True,
        )
        xvar = Variable(
            size=self.T * self.num_src * len(self.dict_tran),
            start=ybvar.start + ybvar.size,
            binary=True,
        )
        xbarvar = Variable(
            size=(self.T + 1) * self.num_src * len(self.dict_conn),
            start=xvar.start + xvar.size,
            binary=True,
        )

        self.vars = {"z": zvar, "e": evar, "yb": ybvar, "x": xvar, "xbar": xbarvar}
        t0 = time.time()

        # Initial constraints on z
        constraint = generate_initial_constraints(self)
        # Dynamic constraints on z, e
        constraint &= generate_powerset_dynamic_constraints(self)
        # Bridge z, e to x, xbar, yb
        constraint &= generate_powerset_bridge_constraints(self)
        # Objective
        obj = self.generate_powerset_objective(add_frontier_rewards=True)

        print("Constraints setup time {:.2f}s".format(time.time() - t0))

        valid_solution = False
        while not valid_solution:
            solution = self._solve(obj, constraint, cut=False, **kwargs)
            if solution["status"] == "infeasible":
                break
            valid_solution, add_S = self.test_solution(solution)
            constraint &= generate_connectivity_constraint(
                self, range(self.num_src), add_S
            )
        # cut static part of solution
        solution = self.cut_solution(solution)

        # save info
        self.traj = {}
        self.conn = {t: set() for t in range(self.T + 1)}
        self.tran = {t: set() for t in range(self.T)}

        for r, v, t in product(self.graph.agents, self.graph.nodes, range(self.T + 1)):
            if solution["x"][self.get_z_idx(r, v, t)] > 0.5:
                self.traj[(r, t)] = v

    def solve_flow(
        self, master=False, connectivity=True, frontier_reward=True, **kwargs
    ):

        self.prepare_problem()

        zvar = Variable(
            size=(self.T + 1) * self.num_r * self.num_v, start=0, binary=True
        )
        xfvar = Variable(
            size=self.T * len(self.graph.agents) * len(self.dict_tran),
            start=zvar.start + zvar.size,
            binary=True,
        )
        yvar = Variable(
            size=self.num_v * self.num_r, start=xfvar.start + xfvar.size, binary=True
        )
        fvar = Variable(
            size=self.T * self.num_min_src_snk * len(self.dict_tran),
            start=yvar.start + yvar.size,
            binary=False,
        )
        fbarvar = Variable(
            size=(self.T + 1) * self.num_min_src_snk * len(self.dict_conn),
            start=fvar.start + fvar.size,
            binary=False,
        )
        mvar = Variable(
            size=self.T * len(self.dict_tran),
            start=fbarvar.start + fbarvar.size,
            binary=False,
        )
        mbarvar = Variable(
            size=(self.T + 1) * len(self.dict_conn),
            start=mvar.start + mvar.size,
            binary=False,
        )

        self.vars = {
            "z": zvar,
            "xf": xfvar,
            "y": yvar,
            "f": fvar,
            "fbar": fbarvar,
            "m": mvar,
            "mbar": mbarvar,
        }
        t0 = time.time()

        # Initial Constraints on z
        constraint = generate_initial_constraints(self)
        # Dynamic Constraints on z, e
        constraint &= generate_flow_dynamic_constraints(self)
        # Bridge z, f, fbar to x, xbar, yb
        constraint &= generate_flow_bridge_constraints(self)
        # Flow master constraints on z, e, m, mbar
        if master:
            constraint &= generate_flow_master_constraints(self)
        # Flow connectivity constraints on z, e, f, fbar
        if connectivity:
            constraint &= generate_flow_connectivity_constraints(self)
        # Flow objective
        obj = self.generate_flow_objective(frontier_reward)

        # User specified as additional constraints
        if self.extra_constr != None:
            for func in self.extra_constr:
                try:
                    constraint &= eval(func[0])(self, func[1])
                except:
                    print("Couldn't find constraint function", func)

        if "verbose" in kwargs and kwargs["verbose"]:
            print("Constraints setup time {:.2f}s".format(time.time() - t0))

        return self._solve(obj, constraint, **kwargs)

    def diameter_solve_flow(self, **kwargs):

        num_frontiers = len(
            [v for v in self.graph.nodes if self.graph.nodes[v]["frontiers"] != 0]
        )

        D = nx.diameter(self.graph)
        Rp = len(set(v for r, v in self.graph.agents.items()))

        T = int(max(D / 2, D - int(Rp / 2)))

        feasible_solution = False
        small_optimal_value = True

        if "verbose" in kwargs and kwargs["verbose"]:
            print(
                Fore.GREEN
                + "Solving flow [R={}, V={}, Et={}, Ec={}, static={}]".format(
                    len(self.graph.agents),
                    self.graph.number_of_nodes(),
                    self.graph.number_of_tran_edges(),
                    self.graph.number_of_conn_edges(),
                    self.static_agents,
                )
                + Style.RESET_ALL
            )

        iter = 0
        while not feasible_solution or small_optimal_value:
            iter += 1
            self.T = T

            if "verbose" in kwargs and kwargs["verbose"]:
                print(
                    "Trying" + Style.BRIGHT + " T={}".format(self.T) + Style.RESET_ALL
                )

            # Solve
            solution = self.solve_flow(**kwargs)

            if solution["status"] is not "infeasible":
                feasible_solution = True

            if (
                num_frontiers > 0
                and ("frontier_reward" in kwargs and kwargs["frontier_reward"])
                and feasible_solution and iter < self.max_reward_demand_iter
            ):
                if (
                    solution["primal objective"]
                    < -self.reward_demand * self.frontier_reward
                ):
                    small_optimal_value = False
                else:
                    print("small optimal value")
            else:
                small_optimal_value = False

            T += 1
        return solution

    def linear_search_solve_flow(self, **kwargs):

        T = 0
        feasible_solution = False
        while not feasible_solution:

            self.T = T

            # Solve
            solution = self.solve_flow(**kwargs)

            if solution["status"] is not "infeasible":
                feasible_solution = True

            T += 1

    def _solve(self, obj, constraint, cut=True, solver=None, verbose=False):

        J_int = sum(
            [
                list(range(var.start, var.start + var.size))
                for var in self.vars.values()
                if not var.binary
            ],
            [],
        )
        J_bin = sum(
            [
                list(range(var.start, var.start + var.size))
                for var in self.vars.values()
                if var.binary
            ],
            [],
        )

        if verbose:
            print(
                "NumConst: {} ({} bin, {} int), NumVar: {}".format(
                    self.num_vars,
                    len(J_bin),
                    len(J_int),
                    constraint.A_eq.shape[0] + constraint.A_iq.shape[0],
                )
            )

        # Solve it
        t0 = time.time()
        solution = solve_ilp(obj, constraint, J_int, J_bin, solver)

        if verbose:
            print("Solver time {:.2f}s".format(time.time() - t0))

        if solution["status"] == "infeasible":
            if verbose:
                print("Problem infeasible")
            self.T_sol = 0
            self.traj = {}
            self.conn = {}
            self.tran = {}
        else:
            self.T_sol = self.T
            if cut:
                solution = self.cut_solution(solution)

            # save info
            self.traj = {}
            self.conn = {t: set() for t in range(self.T_sol + 1)}
            self.tran = {t: set() for t in range(self.T_sol)}

            for r, v, t in product(
                self.graph.agents, self.graph.nodes, range(self.T_sol + 1)
            ):
                if solution["x"][self.get_z_idx(r, v, t)] > 0.5:
                    self.traj[(r, t)] = v

            if "fbar" in self.vars:
                for t, b, (v1, v2) in product(
                    range(self.T_sol + 1),
                    range(self.num_min_src_snk),
                    self.graph.conn_edges(),
                ):
                    if solution["x"][self.get_fbar_idx(b, v1, v2, t)] > 0.5:
                        b_r = (
                            self.src[b]
                            if self.always_src or len(self.src) <= len(self.snk)
                            else self.snk[b]
                        )
                        self.conn[t].add((v1, v2, b_r))

            if "mbar" in self.vars:
                for t, (v1, v2) in product(
                    range(self.T_sol + 1), self.graph.conn_edges()
                ):
                    if solution["x"][self.get_mbar_idx(v1, v2, t)] > 0.5:
                        self.conn[t].add((v1, v2, "master"))

            if "f" in self.vars:
                for t, b, (v1, v2) in product(
                    range(self.T_sol),
                    range(self.num_min_src_snk),
                    self.graph.tran_edges(),
                ):
                    if solution["x"][self.get_f_idx(b, v1, v2, t)] > 0.5:
                        b_r = (
                            self.src[b]
                            if self.always_src or len(self.src) <= len(self.snk)
                            else self.snk[b]
                        )
                        self.tran[t].add((v1, v2, b_r))

            if "m" in self.vars:
                for t, (v1, v2) in product(range(self.T_sol), self.graph.tran_edges()):
                    if solution["x"][self.get_m_idx(v1, v2, t)] > 0.5:
                        self.tran[t].add((v1, v2, "master"))
        return solution

    ##GRAPH HELPER FUNCTIONS##

    def get_time_augmented_id(self, n, t):
        return np.ravel_multi_index((t, n), (self.T + 1, self.num_v))

    def get_time_augmented_n_t(self, id):
        midx = np.unravel_index(id, (self.T + 1, self.num_v))
        return midx[1], midx[0]

    def powerset_exclude_agent(self, b):
        time_augmented_nodes = []
        for t in range(self.T + 1):
            for v in self.graph.nodes:
                time_augmented_nodes.append(self.get_time_augmented_id(v, t))
        s = list(time_augmented_nodes)
        s.remove(self.get_time_augmented_id(self.graph.agents[b], 0))
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

    def test_solution(self, solution):
        add_S = []
        for r in self.snk:
            S = []
            # Find end position of agent r and set as V
            for v in self.graph.nodes:
                z_idx = self.get_z_idx(r, v, self.T)
                if solution["x"][z_idx] == 1:
                    S.append((v, self.T))

            # Backwards reachability
            idx = 0
            while idx < len(S):
                pre_Svt_transition = self.graph.pre_tran_vt([S[idx]])
                pre_Svt_connectivity = self.graph.pre_conn_vt([S[idx]])
                for v0, _, t in pre_Svt_transition:
                    occupied = False
                    for robot in self.graph.agents:
                        z_idx = self.get_z_idx(robot, v0, t)
                        if solution["x"][z_idx] == 1:
                            occupied = True
                    if occupied == True and (v0, t) not in S:
                        S.append((v0, t))
                for v0, _, t in pre_Svt_connectivity:
                    occupied = False
                    for robot in self.graph.agents:
                        z_idx = self.get_z_idx(robot, v0, t)
                        if solution["x"][z_idx] == 1:
                            occupied = True
                    if occupied == True and (v0, t) not in S:
                        S.append((v0, t))
                idx += 1

            valid = True
            for b in self.src:
                if (self.graph.agents[b], 0) not in S:
                    valid = False
            if valid == False:
                add_S.append(S)

        if len(add_S) != 0:
            return False, add_S
        else:
            return True, add_S
