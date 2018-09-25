import math

import numpy as np
from numpy.linalg import norm as norm

from solve_phase_max import solvePhaseMax


def solvePhaseLamp(A=None, At=None, b0=None, x0=None, opts=None):
    T = 10
    eps = 0.001
    # Initial Iterate. Computed using one of the many initializer methods provided with Phasepack.
    xk_prev = x0
    # Start the iteration process
    for i in range(1, T):
        if opts.verbose:
            print('PhaseLamp iteration %d\n', i)
        # Running Phasemax in a loop. This command runs phasemax for the
        # current iteration.
        # def solvePhaseMax(A=None, At=None, b0=None, x0=None, opts=None):
        xk_next, outs = solvePhaseMax(A, At, b0, x0, opts)
        # current and next iterate is minimal.
        tol = norm(xk_next - xk_prev) / \
            max(norm(xk_next), norm(xk_prev) + 1e-10)
        if (tol < eps):
            break
        # Update the iterate
        xk_prev = xk_next

    if opts.verbose:
        print('Total iterations of PhaseLamp run: '+str(i))

    sol = xk_next

    return sol, outs
