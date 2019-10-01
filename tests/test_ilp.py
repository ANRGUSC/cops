from cops.optimization_wrappers import *
import numpy as np
import scipy.sparse as sp

from cops.optimization_wrappers import Constraint, solve_ilp


def test_int_bin():
    A_iq = sp.coo_matrix(np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]))
    b_iq = np.array([2.5, 2.5, 2.5, 2.5])

    constr = Constraint(A_iq=A_iq, b_iq=b_iq)

    c = np.array([-1, -1])

    sol_int = solve_ilp(c, constr, [0, 1], [], solver="gurobi")
    np.testing.assert_equal(sol_int["x"], np.array([2, 2]))

    sol_bin = solve_ilp(c, constr, [], [0, 1], solver="gurobi")
    np.testing.assert_equal(sol_bin["x"], np.array([1, 1]))

    sol_mix = solve_ilp(c, constr, [0], [1], solver="gurobi")
    np.testing.assert_equal(sol_mix["x"], np.array([2, 1]))

    sol_int = solve_ilp(c, constr, [0, 1], [], solver="mosek")
    np.testing.assert_equal(sol_int["x"], np.array([2, 2]))

    sol_bin = solve_ilp(c, constr, [], [0, 1], solver="mosek")
    np.testing.assert_equal(sol_bin["x"], np.array([1, 1]))

    sol_mix = solve_ilp(c, constr, [0], [1], solver="mosek")
    np.testing.assert_equal(sol_mix["x"], np.array([2, 1]))
