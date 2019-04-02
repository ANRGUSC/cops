import numpy as np
import networkx as nx
import scipy.sparse as sp
from collections import deque

from optimization_wrappers import *

from networkx.drawing.nx_pydot import write_dot
from networkx.drawing.nx_agraph import to_agraph


#MultiDiGraph-------------------------------------------------------------------


class Graph(nx.MultiDiGraph):

    def __init__(self):
        super(Graph, self).__init__()
        self.num_nodes = None #Number of nodes
        self.agents = None

    def plot_graph(self):
        #Set general edge-attributes
        self.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}

        #Set individual attributes
        for n in self:
            if self.nodes[n]['number_of_agents']!=0:
                self.nodes[n]['color'] = 'black'
                self.nodes[n]['fillcolor'] = 'red'
                self.nodes[n]['style'] = 'filled'
            else:
                self.nodes[n]['color'] = 'black'
                self.nodes[n]['fillcolor'] = 'white'
                self.nodes[n]['style'] = 'filled'
            for nbr in self[n]:
                for edge in self[n][nbr]:
                    if self[n][nbr][edge]['type']=='connectivity':
                        self[n][nbr][edge]['color']='grey'
                        self[n][nbr][edge]['style']='dashed'
                    else:
                        self[n][nbr][edge]['color']='black'
                        self[n][nbr][edge]['style']='solid'

        #Plot/save graph
        A = to_agraph(self)
        A.layout()
        A.draw('graph.png')

    def add_transition_path(self, transition_list):
        self.add_path(transition_list, type='transition', self_loop = False)
        self.add_path(transition_list[::-1], type='transition', self_loop = False)
        for n in self:
            self.add_edge( n,n , type='transition', self_loop = True)


    def add_connectivity_path(self, connectivity_list):
        self.add_path(connectivity_list, type='connectivity', self_loop = False)
        self.add_path(connectivity_list[::-1] , type='connectivity', self_loop = False)
        for n in self:
            self.add_edge( n,n, type='connectivity', self_loop = True)

    def set_node_positions(self, position_dictionary):
        self.add_nodes_from(position_dictionary.keys())
        for n, p in position_dictionary.items():
            self.node[n]['pos'] = p

    def init_agents(self, agent_position_dictionary):

        for n in self:
            self.node[n]['number_of_agents']=0
            #Change node label to number of agents in node
            self.node[n]['label']=0

        for agent in agent_position_dictionary:
            self.node[agent_position_dictionary[agent]]['number_of_agents']+=1
            #Change node label to number of agents in node
            self.node[agent_position_dictionary[agent]]['label']=self.node[agent_position_dictionary[agent]]['number_of_agents']

        self.agents = agent_position_dictionary
        self.num_nodes = self.number_of_nodes()

    def transition_adjacency_matrix(self):
        num_nodes = self.number_of_nodes()
        adj = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
        for n in self:
            for edge in self.in_edges(n, data = True):
                if edge[2]['type'] == 'transition':
                    adj[edge[0]][edge[1]] = 1
        return adj

    def connectivity_adjacency_matrix(self):
        num_nodes = self.number_of_nodes()
        adj = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
        for n in self:
            for edge in self.in_edges(n, data = True):
                if edge[2]['type'] == 'connectivity':
                    adj[edge[0]][edge[1]] = 1
        return adj


#Dynamic constraints------------------------------------------------------------

class DynamicConstraints(object):
    def __init__(self, problem):
        self.A_eq = None
        self.b_eq = None
        self.problem = problem
        self.generate_dynamic_contraints(self.problem)

        # variables: z^b_rvt, e_ijt, y^b_vt, x^b_ijt, xbar^b_ijt

    def generate_dynamic_contraints(self, problem):
        """Generate equalities (35),(36)"""

        if problem.num_vars == None: problem.add_num_var()

        # Obtain adjacency matrix for transition/connectivity edges separately
        transition_adj = problem.graph.transition_adjacency_matrix()
        connectivity_adj = problem.graph.connectivity_adjacency_matrix()

        #number of dynamic constraints
        num_constraints = (problem.T-1)*len(problem.graph.nodes)

        # Constructing A_eq and b_eq for equality (35) as sp.coo matrix
        A_eq_row  = np.array([])
        A_eq_col  = np.array([])
        A_eq_data = np.array([])


        constraint_idx = 0
        for t in range(problem.T-1):
            for v in problem.graph.nodes:
                #left side of (35)
                for r in problem.graph.agents:
                    A_eq_row = np.append(A_eq_row, constraint_idx)
                    A_eq_col = np.append(A_eq_col, problem.get_z_idx(r, v, t+1))
                    A_eq_data = np.append(A_eq_data, 1)
                #right side of (35)
                for edge in problem.graph.in_edges(v, data = True):
                    if edge[2]['type'] == 'transition':
                        A_eq_row = np.append(A_eq_row, constraint_idx)
                        A_eq_col = np.append(A_eq_col, problem.get_e_idx(edge[0], edge[1], t))
                        A_eq_data = np.append(A_eq_data, -1)
                constraint_idx += 1
        A_eq_35 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(num_constraints, problem.num_vars))#.toarray()
        b_eq_35 = sp.coo_matrix(np.zeros(num_constraints), shape=(num_constraints,))#.toarray()


        # Constructing A_eq and b_eq for equality (36) as sp.coo matrix
        A_eq_row  = np.array([])
        A_eq_col  = np.array([])
        A_eq_data = np.array([])

        constraint_idx = 0
        for t in range(problem.T-1):
            for v in problem.graph.nodes:
                #left side of (35)
                for r in problem.graph.agents:
                    A_eq_row = np.append(A_eq_row, constraint_idx)
                    A_eq_col = np.append(A_eq_col, problem.get_z_idx(r, v, t))
                    A_eq_data = np.append(A_eq_data, 1)
                #right side of (35)
                for edge in problem.graph.out_edges(v, data = True):
                    if edge[2]['type'] == 'transition':
                        A_eq_row = np.append(A_eq_row, constraint_idx)
                        A_eq_col = np.append(A_eq_col, problem.get_e_idx(edge[0], edge[1], t))
                        A_eq_data = np.append(A_eq_data, -1)
                constraint_idx += 1
        A_eq_36 = sp.coo_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(num_constraints, problem.num_vars))#.toarray()
        b_eq_36 = sp.coo_matrix(np.zeros(num_constraints), shape=(num_constraints,))#.toarray()


        self.A_eq = sp.bmat([[A_eq_35], [A_eq_36]])
        self.b_eq = sp.bmat([[b_eq_35], [b_eq_36]])



#Occupancy constraints----------------------------------------------------------

class OccupancyConstraints(object):
    def __init__(self):
        self.A_eq = None
        self.b_eq = None
        self.A_iq = None
        self.b_iq = None


#Connectivity constraints-------------------------------------------------------

class ConnectivityConstraint(object):
    def __init__(self):
        self.A_eq = None
        self.b_eq = None
        self.A_iq = None
        self.b_iq = None


#Connectivity problem-----=-----------------------------------------------------

class ConnectivityProblem(object):
    """For multiple classes"""
    def __init__(self):
        self.graph = None                    #Graph
        self.equality_constraints = None       #equality contraints
        self.inequality_constraints = None     #inequality contraints
        self.T = None                        #Time horizon
        self.b = None                        #bases (agent positions)
        self.num_b = None
        self.num_r = None
        self.num_v = None
        self.num_i = None
        self.num_j = None
        self.num_z = None
        self.num_e = None
        self.num_y = None
        self.num_x = None
        self.num_xbar = None
        self.num_vars = None

        # variables: z^b_rvt, e_ijt, y^b_vt, x^b_ijt, xbar^b_ijt

    def add_num_var(self):

        self.num_b = len(self.b)
        self.num_r = len(self.graph.agents)
        self.num_v = self.graph.num_nodes
        self.num_i = self.num_v
        self.num_j = self.num_v

        self.num_z = self.T * self.num_b * self.num_r * self.num_v
        self.num_e = self.T * self.num_i * self.num_j
        self.num_y = self.T * self.num_b * self.num_v
        self.num_x = self.T * self.num_b * self.num_i * self.num_j
        self.num_xbar = self.T * self.num_b * self.num_i * self.num_j

        self.num_vars = self.num_z + self.num_e + self.num_y + self.num_x + self.num_xbar
        print("Number of variables: {}".format(self.num_vars))


    def get_z_idx(self, r, v, t):

        start = 0
        z_t = (self.num_r * self.num_v) * t
        z_v = (self.num_r) * v
        z_r = r
        idx = start + z_t + z_v + z_r
        return(idx)

    def get_e_idx(self, i, j, t):

        start = self.num_z
        e_t = (self.num_i * self.num_j) * t
        e_i = (self.num_j) * i
        e_j = j
        idx = start + e_t + e_i + e_j
        return(idx)


    def get_y_idx(self, b, v, t):

        start = self.num_z + self.num_e
        y_t = (self.num_b * self.num_v) * t
        y_b = (self.num_v) * b
        y_v = v
        idx = start + y_t + y_b + y_v
        return(idx)

    def get_x_idx(self, b, i, j, t):

        start = self.num_z + self.num_e + self.num_y
        x_t = (self.num_b * self.num_i * self.num_j) * t
        x_b = (self.num_i * self.num_j) * b
        x_i = (self.num_j) * i
        x_j = j
        idx = start + x_t + x_b +x_i + x_j
        return(idx)


    def get_xbar_idx(self, b, i, j, t):

        start = self.num_z + self.num_e + self.num_y + self.num_x
        xbar_t = (self.num_b * self.num_i * self.num_j) * t
        xbar_b = (self.num_i * self.num_j) * b
        xbar_i = (self.num_j) * i
        xbar_j = j
        idx = start + xbar_t + xbar_b +xbar_i + xbar_j
        return(idx)


    def add_constraints(self, constraints):
        for constraint in constraints:
            if type(constraint) is DynamicConstraints:
                if self.equality_constraints is not None:
                    self.equality_constraints[0] = sp.bmat([[self.equality_constraints[0]], [constraint[0]]])
                    self.equality_constraints[1] = sp.bmat([[self.equality_constraints[1]], [constraint[1]]])

                else:
                    self.equality_constraints = constraint.A_eq
                    self.equality_constraints = constraint.b_eq




    def check_well_defined(self):
        # Check input data
        if self.T is None:
            raise Exception("No time horizon 'T' specified")

        if self.graph is None:
            raise Exception("No graph 'G' specified")



    def solve(self, solver=None, output=False, integer=True):

        # variables: z^b_rvt, e_ijt, y^b_vt, x^b_ijt, xbar^b_ijt

        #Equality Constraints
        try:
            A_eq = self.equality_constraints[0]
            b_eq = self.equality_constraints[1]
        except:
            A_eq = sp.coo_matrix(([], ([],[])), shape=(0, self.num_vars))
            b_eq = []

        #Inequality Constraints
        try:
            A_iq = self.inequality_constraints[0]
            b_iq = self.inequality_constraints[1]
        except:
            A_iq = sp.coo_matrix(([], ([],[])), shape=(0, self.num_vars))
            b_iq = []

        print(A_iq.shape)

        obj = np.zeros(self.num_vars)

        # Solve it
        if integer:
            sol = solve_ilp(obj, A_iq, b_iq, A_eq, b_eq,
                            None, solver, output);
        else:
            sol = solve_ilp(obj, A_iq, b_iq, A_eq, b_eq,
                            [], solver, output);

        return sol['x']

    def test_solution(self):
        for g in range(len(self.graphs)):
            # Check dynamics
            np.testing.assert_almost_equal(
                self.x[g][:, 1:],
                self.graphs[g].system_matrix().dot(self.u[g])
            )

            # Check control constraints
            np.testing.assert_almost_equal(
                self.x[g][:, :-1],
                _id_stacked(self.graphs[g].K(), self.graphs[g].M())
                    .dot(self.u[g])
            )

            # Check prefix-suffix connection
            assgn_sum_g = np.zeros(self.graphs[g].K())
            for C, a in zip(self.cycle_sets[g], self.assignments[g]):
                for (Ci, ai) in zip(C, a):
                    assgn_sum_g[self.graphs[g].order_fcn(Ci[0])] += ai

            np.testing.assert_almost_equal(
                self.x[g][:, -1],
                assgn_sum_g
            )

        # Check counting constraints
        for cc in self.constraints:
            for t in range(1000):  # enough to do T + LCM(cycle_lengths)
                assert(self.mode_count(cc.X, t) <= cc.R)

    def mode_count(self, X, t):
        """When a solution has been found, return the `X`-count
        at time `t`"""
        X_count = 0
        for g in range(len(self.graphs)):
            G = self.graphs[g]
            if t < self.T:
                u_t_g = self.u[g][:, t]
                X_count += sum(u_t_g[G.order_fcn(v) +
                                     G.index_of_mode(m) * G.K()]
                               for (v, m) in X[g])
            else:
                ass_rot = [deque(a) for a in self.assignments[g]]
                for a in ass_rot:
                    a.rotate(t - self.T)
                X_count += sum(np.inner(_cycle_row(C, X[g]), a)
                               for C, a in zip(self.cycle_sets[g],
                                               ass_rot))
        return X_count

    def get_aggregate_input(self, t):
        """When an integer solution has been found, return aggregate inputs `u`
            at time t"""
        if self.u is None:
            raise Exception("No solution available")

        agg_u = []

        for g in range(len(self.graphs)):
            G = self.graphs[g]

            if t < self.T:
                # We are in prefix; states are available
                u_g = self.u[g][:, t]

            else:
                u_g = np.zeros(G.K() * G.M())
                # Rotate assignments
                ass_rot = [deque(a) for a in self.assignments[g]]
                for a in ass_rot:
                    a.rotate(t - self.T)

                for assgn, c in zip(ass_rot,
                                    self.cycle_sets[g]):
                    for ai, ci in zip(assgn, c):
                        u_g[G.order_fcn(ci[0]) +
                            G.index_of_mode(ci[1]) * G.K()] += ai
            agg_u.append(u_g)

        return agg_u


    def get_input(self, xi_list, t):
        """When an integer solution has been found, given individual states
        `xi_list` at time t, return individual inputs `sigma`"""
        if self.u is None:
            raise Exception("No solution available")

        actions = []

        for g in range(len(self.graphs)):
            G = self.graphs[g]
            N_g = np.sum(self.inits[g])

            actions_g = [None] * N_g

            if t < self.T:
                # We are in prefix; states are available
                x_g = self.x[g][:, t]
                u_g = self.u[g][:, t]

            else:
                # We are in suffix, must build x_g, u_g from cycle assignments
                u_g = np.zeros(G.K() * G.M())
                x_g = np.zeros(len(self.graphs[g]))

                # Rotate assignments
                ass_rot = [deque(a) for a in self.assignments[g]]
                for a in ass_rot:
                    a.rotate(t - self.T)

                for assgn, c in zip(ass_rot,
                                    self.cycle_sets[g]):
                    x_g += self.graphs[g].index_matrix(c) \
                               .dot(assgn).flatten()
                    for ai, ci in zip(assgn, c):
                        u_g[G.order_fcn(ci[0]) +
                            G.index_of_mode(ci[1]) * G.K()] += ai

            # Assert that xi_list agrees with aggregate solution
            xi_sum = np.zeros(len(self.graphs[g]))
            for xi in xi_list[g]:
                xi_sum[self.graphs[g].order_fcn(xi)] += 1
            print(xi_sum)
            try:
                np.testing.assert_almost_equal(xi_sum,
                                               x_g)
            except:
                raise Exception("States don't agree with aggregate" +
                                " at time " + str(t))

            for n in range(N_g):
                k = G.order_fcn(xi_list[g][n])
                u_state = [u_g[k + G.K() * m] for m in range(G.M())]
                m = next(i for i in range(len(u_state))
                         if u_state[i] >= 1)
                actions_g[n] = G.mode(m)
                u_g[k + G.K() * m] -= 1
            actions.append(actions_g)

        return actions
























    def solve_prefix_suffix(self, solver=None, output=False, integer=True):

        # self.check_well_defined()

        # Variables for each g in self.graphs:
        #  v_g := u_g[0] ... u_g[T-1] x_g[0] ... x_g[T-1]
        #         a_g[0] ... a_g[C1-1] b[0] ... b[LJ-1]
        # These are stacked horizontally as
        #  v_0 v_1 ... v_G-1

        L = len(self.constraints)
        T = self.T

        # Variable counts for each class g
        N_u = T * self.graph.K() * self.graph.M()   # input vars
        N_x = T * self.graph.K()   # state vars

        N_tot = sum(N_u) + sum(N_x)

        # Add dynamic constraints
        A_eq_list = []
        b_eq_list = []

        A_eq1_u, A_eq1_x, b_eq1 = \
            generate_prefix_dyn_cstr(self.graph, T)
        A_eq2_x, A_eq2_a, b_eq2 = \
            generate_prefix_suffix_cstr(self.graph, T)
        A_eq1_b = sp.coo_matrix((A_eq1_u.shape[0], N_b_list[g]))

        A_eq_list.append(
            sp.bmat([[A_eq1_u, A_eq1_x, None, A_eq1_b],
                     [None, A_eq2_x, A_eq2_a, None]])
        )
        b_eq_list.append(
            np.hstack([b_eq1, b_eq2])
        )

        A_eq = sp.block_diag(A_eq_list)
        b_eq = np.hstack(b_eq_list)

        # Add counting constraints
        A_iq_list = []
        b_iq_list = []
        for l in range(len(self.constraints)):
            cc = self.constraints[l]

            # Count over classes: Should be stacked horizontally
            A_iq1_list = []

            # Bounded by slack vars: Should be block diagonalized
            A_iq2_list = []
            b_iq2_list = []

            # Count over bound vars for each class: Should be stacked
            # horizontally
            A_iq3_list = []

            for g in range(len(self.graphs)):
                # Prefix counting
                A_iq1_u, b_iq1 = \
                    generate_prefix_counting_cstr(self.graphs[g], T,
                                                  cc.X[g], cc.R)
                A_iq1_list.append(
                    sp.bmat([[A_iq1_u, sp.coo_matrix((T, N_x_list[g] +
                                                      N_a_list[g] +
                                                      N_b_list[g]))]])
                )

                # Suffix counting
                A_iq2_a, A_iq2_b, b_iq2, A_iq3_a, A_iq3_b, b_iq3 = \
                    generate_suffix_counting_cstr(self.cycle_sets[g],
                                                  cc.X[g], cc.R)

                b_head2 = sp.coo_matrix((N_a_list[g], l * J_list[g]))
                b_tail2 = sp.coo_matrix((N_a_list[g], (L - 1 - l) * J_list[g]))
                A_iq2_b = sp.bmat([[b_head2, A_iq2_b, b_tail2]])
                b_head3 = sp.coo_matrix((1, l * J_list[g]))
                b_tail3 = sp.coo_matrix((1, (L - 1 - l) * J_list[g]))
                A_iq3_b = sp.bmat([[b_head3, A_iq3_b, b_tail3]])

                A_iq2_u = sp.coo_matrix((N_a_list[g], N_u_list[g]))
                A_iq2_x = sp.coo_matrix((N_a_list[g], N_x_list[g]))

                A_iq2_list.append(
                    sp.bmat([[A_iq2_u, A_iq2_x, A_iq2_a, A_iq2_b]])
                )
                b_iq2_list.append(b_iq2)

                A_iq3_list.append(
                    sp.bmat([[sp.coo_matrix((1, N_u_list[g] + N_x_list[g])),
                              A_iq3_a, A_iq3_b]])
                )

            # Stack horizontally
            A_iq_list.append(sp.bmat([A_iq1_list]))
            b_iq_list.append(b_iq1)

            # Stack by block
            A_iq_list.append(sp.block_diag(A_iq2_list))
            b_iq_list.append(np.hstack(b_iq2_list))

            # Stack horizontally
            A_iq_list.append(sp.bmat([A_iq3_list]))
            b_iq_list.append(b_iq3)

        # Stack everything vertically
        if len(A_iq_list) > 0:
            A_iq = sp.bmat([[A] for A in A_iq_list])
            b_iq = np.hstack(b_iq_list)
        else:
            A_iq = sp.coo_matrix((0, N_tot))
            b_iq = np.zeros(0)

        # Solve it
        if integer:
            sol = solve_mip(np.zeros(N_tot), A_iq, b_iq, A_eq, b_eq,
                            range(N_tot), solver, output);
        else:
            sol = solve_mip(np.zeros(N_tot), A_iq, b_iq, A_eq, b_eq,
                            [], solver, output);

        # Extract solution (if valid)
        if sol['status'] == 2:
            self.u = []
            self.x = []
            self.assignments = []

            idx0 = 0
            for g in range(len(self.graphs)):
                self.u.append(
                    np.array(sol['x'][idx0:idx0 + N_u_list[g]], dtype=int)
                      .reshape(T, self.graphs[g].K() * self.graphs[g].M())
                      .transpose()
                )
                self.x.append(
                    np.hstack([
                        np.array(self.inits[g]).reshape(len(self.inits[g]), 1),
                        np.array(sol['x'][idx0 + N_u_list[g]:
                                          idx0 + N_u_list[g] + N_x_list[g]])
                          .reshape(T, self.graphs[g].K()).transpose()
                    ])
                )

                cycle_lengths = [len(C) for C in self.cycle_sets[g]]
                self.assignments.append(
                    [sol['x'][idx0 + N_u_list[g] + N_x_list[g] + an:
                              idx0 + N_u_list[g] + N_x_list[g] + an + dn]
                     for an, dn in zip(np.cumsum([0] +
                                       cycle_lengths[:-1]),
                                       cycle_lengths)]
                )
                idx0 += N_u_list[g] + N_x_list[g] + N_a_list[g] + N_b_list[g]
        return sol['status']


    def test_solution(self):
        for g in range(len(self.graphs)):
            # Check dynamics
            np.testing.assert_almost_equal(
                self.x[g][:, 1:],
                self.graphs[g].system_matrix().dot(self.u[g])
            )

            # Check control constraints
            np.testing.assert_almost_equal(
                self.x[g][:, :-1],
                _id_stacked(self.graphs[g].K(), self.graphs[g].M())
                    .dot(self.u[g])
            )

            # Check prefix-suffix connection
            assgn_sum_g = np.zeros(self.graphs[g].K())
            for C, a in zip(self.cycle_sets[g], self.assignments[g]):
                for (Ci, ai) in zip(C, a):
                    assgn_sum_g[self.graphs[g].order_fcn(Ci[0])] += ai

            np.testing.assert_almost_equal(
                self.x[g][:, -1],
                assgn_sum_g
            )

        # Check counting constraints
        for cc in self.constraints:
            for t in range(1000):  # enough to do T + LCM(cycle_lengths)
                assert(self.mode_count(cc.X, t) <= cc.R)

    def mode_count(self, X, t):
        """When a solution has been found, return the `X`-count
        at time `t`"""
        X_count = 0
        for g in range(len(self.graphs)):
            G = self.graphs[g]
            if t < self.T:
                u_t_g = self.u[g][:, t]
                X_count += sum(u_t_g[G.order_fcn(v) +
                                     G.index_of_mode(m) * G.K()]
                               for (v, m) in X[g])
            else:
                ass_rot = [deque(a) for a in self.assignments[g]]
                for a in ass_rot:
                    a.rotate(t - self.T)
                X_count += sum(np.inner(_cycle_row(C, X[g]), a)
                               for C, a in zip(self.cycle_sets[g],
                                               ass_rot))
        return X_count

    def get_aggregate_input(self, t):
        """When an integer solution has been found, return aggregate inputs `u`
            at time t"""
        if self.u is None:
            raise Exception("No solution available")

        agg_u = []

        for g in range(len(self.graphs)):
            G = self.graphs[g]

            if t < self.T:
                # We are in prefix; states are available
                u_g = self.u[g][:, t]

            else:
                u_g = np.zeros(G.K() * G.M())
                # Rotate assignments
                ass_rot = [deque(a) for a in self.assignments[g]]
                for a in ass_rot:
                    a.rotate(t - self.T)

                for assgn, c in zip(ass_rot,
                                    self.cycle_sets[g]):
                    for ai, ci in zip(assgn, c):
                        u_g[G.order_fcn(ci[0]) +
                            G.index_of_mode(ci[1]) * G.K()] += ai
            agg_u.append(u_g)

        return agg_u


    def get_input(self, xi_list, t):
        """When an integer solution has been found, given individual states
        `xi_list` at time t, return individual inputs `sigma`"""
        if self.u is None:
            raise Exception("No solution available")

        actions = []

        for g in range(len(self.graphs)):
            G = self.graphs[g]
            N_g = np.sum(self.inits[g])

            actions_g = [None] * N_g

            if t < self.T:
                # We are in prefix; states are available
                x_g = self.x[g][:, t]
                u_g = self.u[g][:, t]

            else:
                # We are in suffix, must build x_g, u_g from cycle assignments
                u_g = np.zeros(G.K() * G.M())
                x_g = np.zeros(len(self.graphs[g]))

                # Rotate assignments
                ass_rot = [deque(a) for a in self.assignments[g]]
                for a in ass_rot:
                    a.rotate(t - self.T)

                for assgn, c in zip(ass_rot,
                                    self.cycle_sets[g]):
                    x_g += self.graphs[g].index_matrix(c) \
                               .dot(assgn).flatten()
                    for ai, ci in zip(assgn, c):
                        u_g[G.order_fcn(ci[0]) +
                            G.index_of_mode(ci[1]) * G.K()] += ai

            # Assert that xi_list agrees with aggregate solution
            xi_sum = np.zeros(len(self.graphs[g]))
            for xi in xi_list[g]:
                xi_sum[self.graphs[g].order_fcn(xi)] += 1
            print(xi_sum)
            try:
                np.testing.assert_almost_equal(xi_sum,
                                               x_g)
            except:
                raise Exception("States don't agree with aggregate" +
                                " at time " + str(t))

            for n in range(N_g):
                k = G.order_fcn(xi_list[g][n])
                u_state = [u_g[k + G.K() * m] for m in range(G.M())]
                m = next(i for i in range(len(u_state))
                         if u_state[i] >= 1)
                actions_g[n] = G.mode(m)
                u_g[k + G.K() * m] -= 1
            actions.append(actions_g)

        return actions


def solve_prefix(cp, solver=None, output=False):

    # cp.check_well_defined()

    # Make sure that we have a valid suffix
    for g in range(len(cp.graphs)):
        assert(len(cp.cycle_sets[g]))
        assert(len(cp.assignments[g]) == len(cp.cycle_sets[g]))
        for i in range(len(cp.cycle_sets[g])):
            assert(len(cp.assignments[g][i]) == len(cp.cycle_sets[g][i]))

    # Variables for each g in cp.graphs:
    #  v_g := u_g[0] ... u_g[T-1] x_g[0] ... x_g[T-1]
    # These are stacked horizontally as
    #  v_0 v_1 ... v_G-1

    L = len(cp.constraints)
    T = cp.T

    # Variable counts for each class g
    N_u_list = [T * G.K() * G.M() for G in cp.graphs]   # input vars
    N_x_list = [T * G.K() for G in cp.graphs]   # state vars

    N_tot = sum(N_u_list) + sum(N_x_list)

    # Add dynamics constraints, should be block diagonalized
    A_eq_list = []
    b_eq_list = []

    for g in range(len(cp.graphs)):
        A_eq1_u, A_eq1_x, b_eq1 = \
            generate_prefix_dyn_cstr(cp.graphs[g], T, cp.inits[g])
        A_eq2_x, A_eq2_a, b_eq2 = \
            generate_prefix_suffix_cstr(cp.graphs[g], T,
                                        cp.cycle_sets[g])

        A_eq_list.append(
            sp.bmat([[A_eq1_u, A_eq1_x],
                     [None, A_eq2_x]])
        )

        b_eq_list.append(
            np.hstack([b_eq1, b_eq2 - A_eq2_a.dot(np.hstack(cp.assignments[g]))])
        )

    A_eq = sp.block_diag(A_eq_list)
    b_eq = np.hstack(b_eq_list)

    # Add counting constraints
    A_iq_list = []
    b_iq_list = []
    for l in range(len(cp.constraints)):
        cc = cp.constraints[l]

        # Count over classes: Should be stacked horizontally
        A_iq1_list = []

        for g in range(len(cp.graphs)):
            # Prefix counting
            A_iq1_u, b_iq1 = \
                generate_prefix_counting_cstr(cp.graphs[g], T,
                                              cc.X[g], cc.R)
            A_iq1_list.append(
                sp.bmat([[A_iq1_u, sp.coo_matrix((T, N_x_list[g]))]])
            )

        # Stack horizontally
        A_iq_list.append(sp.bmat([A_iq1_list]))
        b_iq_list.append(b_iq1)

    # Stack everything vertically
    if len(A_iq_list) > 0:
        A_iq = sp.bmat([[A] for A in A_iq_list])
        b_iq = np.hstack(b_iq_list)
    else:
        A_iq = sp.coo_matrix((0, N_tot))
        b_iq = np.zeros(0)

    # Solve it
    sol = solve_mip(np.zeros(N_tot), A_iq, b_iq, A_eq, b_eq,
                    range(N_tot), solver, output);

    # Extract solution (if valid)
    if sol['status'] == 2:
        cp.u = []
        cp.x = []

        idx0 = 0
        for g in range(len(cp.graphs)):
            cp.u.append(
                np.array(sol['x'][idx0:idx0 + N_u_list[g]], dtype=int)
                  .reshape(T, cp.graphs[g].K() * cp.graphs[g].M())
                  .transpose()
            )
            cp.x.append(
                np.hstack([
                    np.array(cp.inits[g]).reshape(len(cp.inits[g]), 1),
                    np.array(sol['x'][idx0 + N_u_list[g]:
                                      idx0 + N_u_list[g] + N_x_list[g]])
                      .reshape(T, cp.graphs[g].K()).transpose()
                ])
            )
            idx0 += N_u_list[g] + N_x_list[g]
    return sol['status']


####################################
#  Constraint-generating functions #
####################################


def generate_dyn_cstr(G, T, init):
    """Generate equalities (35),(36)"""
    K = G.K()
    M = G.M()

    # variables: u[0], ..., u[T-1], x[1], ..., x[T]

    # Obtain system matrix
    B = G.system_matrix()

    # (47c)
    # T*K equalities
    A_eq1_u = sp.block_diag((B,) * T)
    A_eq1_x = sp.block_diag((sp.identity(K),) * T)
    b_eq1 = np.zeros(T * K)

    # (47e)
    # T*K equalities
    A_eq2_u = sp.block_diag((_id_stacked(K, M),) * T)
    A_eq2_x = sp.bmat([[sp.coo_matrix((K, K * (T - 1))),
                        sp.coo_matrix((K, K))],
                       [sp.block_diag((sp.identity(K),) * (T - 1)),
                        sp.coo_matrix((K * (T - 1), K))]
                       ])
    b_eq2 = np.hstack([init, np.zeros((T - 1) * K)])

    # Forbid non-existent modes
    # T * len(ban_idx) equalities
    ban_idx = [G.order_fcn(v) + m * K
               for v in G.nodes_iter()
               for m in range(M)
               if G.mode(m) not in G.node_modes(v)]
    A_eq3_u_part = sp.coo_matrix(
        (np.ones(len(ban_idx)), (range(len(ban_idx)), ban_idx)),
        shape=(len(ban_idx), K * M)
    )
    A_eq3_u = sp.block_diag((A_eq3_u_part,) * T)
    A_eq3_x = sp.coo_matrix((T * len(ban_idx), T * K))
    b_eq3 = np.zeros(T * len(ban_idx))

    # Stack everything
    A_eq_u = sp.bmat([[A_eq1_u],
                     [A_eq2_u],
                     [A_eq3_u]])
    A_eq_x = sp.bmat([[-A_eq1_x],
                      [-A_eq2_x],
                      [A_eq3_x]])
    b_eq = np.hstack([b_eq1, b_eq2, b_eq3])

    return A_eq_u, A_eq_x, b_eq



def generate_prefix_dyn_cstr(G, T, init):
    """Generate equalities (47c), (47e) for prefix dynamics"""
    K = G.K()
    M = G.M()

    # variables: u[0], ..., u[T-1], x[1], ..., x[T]

    # Obtain system matrix
    B = G.system_matrix()

    # (47c)
    # T*K equalities
    A_eq1_u = sp.block_diag((B,) * T)
    A_eq1_x = sp.block_diag((sp.identity(K),) * T)
    b_eq1 = np.zeros(T * K)

    # (47e)
    # T*K equalities
    A_eq2_u = sp.block_diag((_id_stacked(K, M),) * T)
    A_eq2_x = sp.bmat([[sp.coo_matrix((K, K * (T - 1))),
                        sp.coo_matrix((K, K))],
                       [sp.block_diag((sp.identity(K),) * (T - 1)),
                        sp.coo_matrix((K * (T - 1), K))]
                       ])
    b_eq2 = np.hstack([init, np.zeros((T - 1) * K)])

    # Forbid non-existent modes
    # T * len(ban_idx) equalities
    ban_idx = [G.order_fcn(v) + m * K
               for v in G.nodes_iter()
               for m in range(M)
               if G.mode(m) not in G.node_modes(v)]
    A_eq3_u_part = sp.coo_matrix(
        (np.ones(len(ban_idx)), (range(len(ban_idx)), ban_idx)),
        shape=(len(ban_idx), K * M)
    )
    A_eq3_u = sp.block_diag((A_eq3_u_part,) * T)
    A_eq3_x = sp.coo_matrix((T * len(ban_idx), T * K))
    b_eq3 = np.zeros(T * len(ban_idx))

    # Stack everything
    A_eq_u = sp.bmat([[A_eq1_u],
                     [A_eq2_u],
                     [A_eq3_u]])
    A_eq_x = sp.bmat([[-A_eq1_x],
                      [-A_eq2_x],
                      [A_eq3_x]])
    b_eq = np.hstack([b_eq1, b_eq2, b_eq3])

    return A_eq_u, A_eq_x, b_eq


def generate_prefix_counting_cstr(G, T, X, R):
    """Generate inequalities (47a) for prefix counting constraints"""
    K = G.K()
    M = G.M()

    # variables: u[0], ..., u[T-1]

    col_idx = [G.order_fcn(v) + G.index_of_mode(m) * K for (v, m) in X]
    if len(col_idx) == 0:
        # No matches
        return sp.coo_matrix((T, T * K * M)), R * np.ones(T)

    val = np.ones(len(col_idx))
    row_idx = np.zeros(len(col_idx))
    A_pref_cc = sp.coo_matrix(
        (val, (row_idx, col_idx)), shape=(1, K * M)
    )

    A_iq_u = sp.block_diag((A_pref_cc,) * T)
    b_iq = R * np.ones(T)

    return A_iq_u, b_iq


def generate_prefix_suffix_cstr(G, T, cycle_set):
    """Generate K equalities (47d) that connect prefix and suffix"""
    K = G.K()

    # Variables x[1] ... x[T] a[0] ... a[C-1]

    Psi_mats = [G.index_matrix(C) for C in cycle_set]

    A_eq_x = sp.bmat(
        [[sp.coo_matrix((K, K * (T - 1))), -sp.identity(K)]]
    )

    A_eq_a = sp.bmat([Psi_mats])
    b_eq = np.zeros(K)

    return A_eq_x, A_eq_a, b_eq


def generate_suffix_counting_cstr(cycle_set, X, R):
    """Generate inequalities (47b) for suffix counting"""

    # Variables a[0] ... a[C-1] + slack vars b[c][l]

    J = len(cycle_set)
    N_cycle_tot = sum(len(C) for C in cycle_set)

    # First set: A_iq1_a a + A_iq1_b b \leq b_iq1
    # guarantee that count in each cycle is less than
    # slack var
    A_iq1_a = sp.block_diag(tuple([_cycle_matrix(C, X)
                            for C in cycle_set]))

    A_iq1_b = sp.block_diag(tuple([-np.ones([len(C), 1])
                            for C in cycle_set]))
    b_iq1 = np.zeros(N_cycle_tot)

    # Second set: A_iq2_b b \leq b_iq2
    # guarantees that sum of slack vars
    # less than R
    A_iq2_a = sp.coo_matrix((1, N_cycle_tot))
    A_iq2_b = sp.coo_matrix((np.ones(J), (np.zeros(J), range(J))),
                            shape=(1, J))

    b_iq2 = np.array([R])

    return A_iq1_a, A_iq1_b, b_iq1, A_iq2_a, A_iq2_b, b_iq2

####################
# Helper functions #
####################


def _id_stacked(K, M):
    """Return the (K x MK) sparse matrix [I I ... I]"""
    return sp.coo_matrix((np.ones(K * M),
                         ([k
                           for k in range(K)
                           for m in range(M)],
                          [k + m * K
                           for k in range(K)
                           for m in range(M)]
                          )
                          ),
                         shape=(K, K * M))


def _cycle_row(C, X):
    """Compute vector v s.t. <v,alpha> is
       the number of subsystems in `X`"""
    return [1 if C[i] in X else 0 for i in range(len(C))]


def _cycle_matrix(C, X):
    """Compute matrix A_C s.t. A_C alpha is
       all rotated numbers of subsystems in `X`"""
    idx = deque([i for i in range(len(C)) if C[i] in X])
    vals = np.ones(len(idx) * len(C))
    row_idx = []
    col_idx = []
    for i in range(len(C)):
        row_idx += (i,) * len(idx)
        col_idx += [(j - i) % len(C) for j in idx]
    return sp.coo_matrix(
        (vals, (row_idx, col_idx)), shape=(len(C), len(C))
    )
