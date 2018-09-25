import sys
sys.path.append(u'../util')
sys.path.append(u'../initializers')
sys.path.append(u'../solvers')
import struct
import warnings
from inspect import isfunction

import numpy as np
from numpy import dot

from init_optimal_spectral import initOptimalSpectral
from manage_options import manageOptions
from solve_phase_lamp import solvePhaseLamp
from solve_phase_max import solvePhaseMax


def solvePhaseRetrieval(A=None, At=None, b0=None, n=None, opts=None, *args, **kwargs):
    # If opts is not provided, create it
    if not opts:
        opts = struct

    # If A is a matrix, infer n and At from A
    if (not np.isnan(A[0][0])) & (A[0][0] != None):
        n = A.shape[2-1]

    # Apply default options and validate user-defined options
    opts = manageOptions(opts)
    # Check that inputs are of valid datatypes and sizes
    validateInput(A, At, b0, n, opts)
    # Check that At is the adjoint/transpose of A
    checkAdjoint(A, At, b0)
    x0 = initX(A, At, b0, n, opts)

    # Truncate imaginary components of x0 if working with real values
    if not (opts.isComplex):
        x0 = x0.real()
    else:
        if opts.isNonNegativeOnly:
            warnings.warn(
                'opts.isNonNegativeOnly will not be used when the signal is complex.')

    sol, outs = solveX(A, At, b0, x0, opts)

    return sol, outs, opts


# Helper functions
# Initialize x0 using the specified initialization method
def initX(A=None, At=None, b0=None, n=None, opts=None):
    if (opts.initMethod).lower() in ['optimal', 'optimalspectral']:
        x0 = initOptimalSpectral(A, At, b0, n, True, opts.verbose)
    else:
        print("error", 'Unknown initialization method "%s"', opts.initMethod)

    return x0


# Estimate x0 using the specified algorithm
def solveX(A=None, At=None, b0=None, x0=None, opts=None, *args, **kwargs):
    # print((opts.algorithm).lower())
    if 'phasemax' == (opts.algorithm).lower():
        sol, outs = solvePhaseMax(A, At, b0, x0, opts)
    elif 'phaselamp' == (opts.algorithm).lower():
        # def solvePhaseLamp(A=None, At=None, b0=None, x0=None, opts=None):
        sol, outs = solvePhaseLamp(A, At, b0, x0, opts)
    else:
        print("error", 'Unknown algorithm {0}'.format(opts.algorithm))

    return sol, outs


# Check validity of input
def validateInput(A=None, At=None, b0=None, n=None, opts=None, *args, **kwargs):
    # if ~isnumeric(A) & (isempty(At)|isempty(n))
    '''
    if (not np.isnan(A[0][0])) & ((At == None) | (n == None)):
        # error('If A is a function handle, then At and n must be provided')
        print('If A is a function handle, then At and n must be provided')
    '''
    assert(n > 0, 'n must be positive')
    assert(abs(b0) == b0, 'b must be real-valued and non-negative')

    # if ~isnumeric(A) & isnumeric(At)
    '''
    if ( np.isnan(A[0][0])) & (not np.isnan(A[0][0])):
        # error('If A is a function handle, then At must also be a function handle');
        print('If A is a function handle, then At must also be a function handle')
    '''

    if opts.customx0 != None:
        assert(np.shape(opts.customx0) == (n, 1),
               'customx0 must be a column vector of length n')

    return


# Check that A and At are indeed ajoints of one another
def checkAdjoint(A=None, At=None, b=None, *args, **kwargs):
    y = np.random.randn(np.shape(b)[0])
    Aty = dot(A.T, y)
    x = np.random.randn(np.shape(Aty)[0])
    Ax = dot(A, x)
    innerProduct1 = np.dot(Ax.flatten('F'), y)
    innerProduct2 = np.dot(x, Aty.flatten('F'))
    error = abs(innerProduct1 - innerProduct2) / abs(innerProduct1)
    if error >= 0.001:
        print('Invalid measurement operator:  At is not the adjoint of A.  Error = {0}'.format(
            str(error)))
        assert(error < 0.001, 'Invalid measurement operator:  At is not the adjoint of A.  Error = {0}'.format(
            str(error)))
    return
