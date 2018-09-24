#                           solveCoordinateDescent.m

#  Implementation of the Coordinate Descent algorithm proposed in the paper.

# I/O
#  Inputs:
#     A:    m x n matrix or a function handle to a method that
#           returns A*x.
#     At:   The adjoint (transpose) of 'A'. If 'A' is a function handle, 'At'
#           must be provided.
#     b0:   m x 1 real,non-negative vector consists of all the measurements.
#     x0:   n x 1 vector. It is the initial guess of the unknown signal x.
#     opts: A struct consists of the options for the algorithm. For details,
#           see header in solvePhaseRetrieval.m or the User Guide.

#     Note: When a function handle is used, the
#     value of 'At' (a function handle for the adjoint of 'A') must be
#     supplied.
#
#  Outputs:
#     sol:  n x 1 vector. It is the estimated signal.
#     outs: A struct consists of the convergence info. For details,
#           see header in solvePhaseRetrieval.m or the User Guide.
#
#
#  See the script 'testCoordinateDescent.m' for an example of proper usage
#  of this function.

# Notations
#  Notations mainly follow the paper's notation.

# Algorithm Description
#  CD is an iterative procedure that successively minimizes the objective
#  function along coordinate directions. A single unknown is solved at each
#  iteration while all other variables are kept fixed. As a result, only
#  minimization of a univariate quartic polynomial is needed which is
#  easily achieved by finding the closed-form roots of a cubic equation.
#
#  Specifically, the method has the following steps: It keeps running until
#  the normalized gradient becomes smaller than the tolerance (1) At each
#  iteration, use the selected rule to choose an index i. (2) Minimize the
#  objective f with respect to the ith variable while keeping
#      all other 2n-1 (both real and imaginary parts are variables so there
#      are 2n in total) variables fixed by solving the cubic equation to
#      get alpha that minimize the objective along the ith variable.
#  (3) update the estimate along the ith variable by alpha.
#
# References
#  Paper Title:   Coordinate Descent Algorithms for Phase Retrieval
#  Place:         Chapter II.B
#  Authors:       Wen-Jun Zeng, H. C. So
#  arXiv Address: https://arxiv.org/abs/1706.03474
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

# -----------------------------START-----------------------------------
import math
import struct
import sys
import time

import numpy as np
from numpy import dot
from numpy.linalg import norm
from numpy.random import randn

from check_if_number import checkIfNumber
from display_verbose_output import displayVerboseOutput
from generate_outputs import generateOutputs
from stop_now import stopNow

sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/solvers')
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/initializers')
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/util')
from check_if_in_list import checkIfInList


def solveCoordinateDescent(A=None, At=None, b0=None, x0=None, opts=None, *args, **kwargs):
    validateOptions(opts)

    # Initialization
    m = len(b0)
    n = len(x0)
    sol = x0
    Ax = A(sol)

    currentTime = []
    currentResid = []
    currentReconError = []
    currentMeasurementError = []

    solveTimes, measurementErrors, reconErrors, residuals = initializeContainers(
        opts, nargout=4)
    maxDiff = - np.inf
    C = np.zeros(m, 3)

    # choice rule)
    f = lambda z=None: np.multiply((abs(z) ** 2 - b0 ** 2), z)
    startTime = time.time

    for iter in arange(1, opts.maxIters).reshape(-1):
        if 'random' == lower(opts.indexChoice):
            if opts.isComplex:
                index = randi(dot(2, n))
            else:
                index = randi(n)
        else:
            if 'cyclic' == lower(opts.indexChoice):
                if opts.isComplex:
                    index = mod(iter - 1, dot(2, n)) + 1
                else:
                    index = mod(iter - 1, n) + 1
            else:
                if 'greedy' == lower(opts.indexChoice):
                    grad = At(f(A(sol)))
                    if opts.isComplex:
                        grad_bar = concat([[real(grad)], [imag(grad)]])
                    else:
                        grad_bar = grad
                    __, index = max(abs(grad_bar), nargout=2)
        vals = conj(A(double(arange(1, n == mod(index - 1, n) + 1)).T))
        for j in arange(1, m).reshape(-1):
            if index > n:
                C[j, 2] = dot(2, imag(dot(Ax(j), vals(j))))
            else:
                C[j, 2] = dot(2, real(dot(Ax(j), vals(j))))
            C[j, 3] = abs(vals(j)) ** 2
            C[j, 1] = abs(Ax(j)) ** 2
        d_4 = sum(C(arange(), 3) ** 2)
        d_3 = sum(multiply(dot(2, C(arange(), 3)), C(arange(), 2)))
        d_2 = sum(C(arange(), 2) ** 2 +
                  multiply(dot(2, C(arange(), 3)), (C(arange(), 1) - b0 ** 2)))
        d_1 = sum(multiply(dot(2, C(arange(), 2)), (C(arange(), 1) - b0 ** 2)))
        alphas = roots(concat([dot(4, d_4), dot(3, d_3), dot(2, d_2), d_1])).T
        alphas = alphas(imag(alphas) == 0)
        g = lambda x=None: dot(d_4, x ** 4) + dot(d_3,
                                                  x ** 3) + dot(d_2, x ** 2) + dot(d_1, x)
        __, idx = min(g(alphas), nargout=2)
        alpha = alphas(idx)
        if (index > n):
            a_j = A(double(arange(1, n == index - n)).T)
            sol[index - n] = sol(index - n) + dot(1j, alpha)
            Ax = Ax + dot((dot(1j, alpha)), a_j)
        else:
            a_j = A(double(arange(1, n == index)).T)
            sol[index] = sol(index) + alpha
            Ax = Ax + dot(alpha, a_j)
        diff = abs(alpha)
        maxDiff = max(diff, maxDiff)
        # If xt is provided, reconstruction error will be computed and used for stopping
        # condition. Otherwise, residual will be computed and used for stopping
        # condition.
        if logical_not(isempty(opts.xt)):
            x = sol
            xt = opts.xt
            alpha = (dot(ravel(x).T, ravel(xt))) / (dot(ravel(x).T, ravel(x)))
            x = dot(alpha, x)
            currentReconError = norm(x - xt) / norm(xt)
            if opts.recordReconErrors:
                reconErrors[iter] = currentReconError
        if logical_or(isempty(opts.xt), opts.recordResiduals):
            currentResid = diff / max(maxDiff, 1e-30)
        if opts.recordResiduals:
            residuals[iter] = currentResid
        currentTime = time.time
        if opts.recordTimes:
            solveTimes[iter] = currentTime
        if opts.recordMeasurementErrors:
            currentMeasurementError = norm(abs(A(sol)) - b0) / norm(b0)
            measurementErrors[iter] = currentMeasurementError
        # Display verbose output if specified
        if opts.verbose == 2:
            displayVerboseOutput(
                iter, currentTime, currentResid, currentReconError, currentMeasurementError)
        #  Test stopping criteria.
        if stopNow(opts, currentTime, currentResid, currentReconError):
            break

    # Create output according to the options chosen by user
    outs = generateOutputs(opts, iter, solveTimes,
                           measurementErrors, reconErrors, residuals)

    if opts.verbose == 1:
        displayVerboseOutput(iter, currentTime, currentResid,
                             currentReconError, currentMeasurementError)

    return sol, outs

    # Validate algorithm-specific options


def validateOptions(opts=None, *args, **kwargs):

    validIndexChoices = cellarray(['cyclic', 'random', 'greedy'])
    checkIfInList('indexChoice', opts.indexChoice, validIndexChoices)
    return
