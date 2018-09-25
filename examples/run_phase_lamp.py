import struct
import sys
sys.path.append(u'../util')
sys.path.append(u'../solvers')

import numpy as np
from numpy.linalg import norm

from build_test_problem import buildTestProblem
from plot_error_convergence import plotErrorConvergence
from plot_recovered_VS_original import plotRecoveredVSOriginal
from solve_phase_retrieval import solvePhaseRetrieval



n = 256          # Dimension of unknown vector
m = 7 * n        # Number of measurements
isComplex = True  # If the signal and measurements are complex

# Build a random test problem
print('Building test problem...\n')
[A, xt, b0] = buildTestProblem(m, n, isComplex)

# Options
opts = struct
opts.initMethod = 'optimal'
opts.algorithm = 'PhaseLamp'
opts.isComplex = isComplex
opts.maxIters = 10000
opts.tol = 1e-6
opts.verbose = 1

# Try to recover x
print('Running algorithm...\n')
[x, outs, opts] = solvePhaseRetrieval(A, A.T, b0, n, opts)

# Determine the optimal phase rotation so that the recovered solution
#  matches the True solution as well as possible.
alpha = (np.dot(x.T, xt)) / (np.dot(x.T, x))

x = alpha * x

# Determine the relative reconstruction error.  If the True signal was recovered, the error should be very small - on the order of the numerical accuracy of the solver.
reconError = norm(xt-x)/norm(xt)
# print('relative recon error = {0}\n'.format(reconError))
# Plot a graph of error(definition depends on if opts.xt is provided) versus the number of iterations.
plotErrorConvergence(outs, opts)
# Plot a graph of the recovered signal x against the True signal xt.
plotRecoveredVSOriginal(x, xt)
