# ------------------------solveKaczmarzSimple.m--------------------------------------

# Solver for the Kaczmarz method as given in Algorithm 3 of the Kaczmarz
# paper. Refer to the userguide for a detailed usage of the package.

#  See the script 'testKaczmarzSimple.m' for an example of proper usage of
#  this function.

# PAPER TITLE:
#              Solving systems of phaseless equations via Kaczmarz methods:
#              A proof of concept study.

# ARXIV LINK:
#              https://arxiv.org/pdf/1502.01822.pdf

# INPUTS:
#         A:   Function handle/numerical matrix for data matrix A. The rows
#              of this matrix are the measurement vectors that produce
#              amplitude measurements '\psi'.
#         At:  Function handle/numerical matrix for A transpose.
#         b0:  Observed data vector consisting of amplitude measurements
#              generated from b0 = |A*x|. We assign it to 'psi' to be
#              consistent with the notation in the paper.
#         x0:  The initial vector to be used by any solver.
#        opts: struct consists of the options for the algorithm. For
#              details,see header in solvePhaseRetrieval.m or the User
#              Guide.

# OUPTUT :
#         sol: n x 1 vector. It is the estimated signal.
#        outs: A struct consists of the convergence info. For details,
#              see header in solvePhaseRetrieval.m or the User Guide.

# Note:        When a function handle is used, the value of 'n' (the length
#              of the unknown signal) and 'At' (a function handle for the
#              adjoint of 'A') must be supplied. When 'A' is numeric, the
#              values of 'At' and 'n' are ignored and inferred from the
#              arguments

# DESCRIPTION:
#             The kaczmarz method is an iterative method based on an
#             algebraic reconstruction technique where an intial guess is
#             chosen and the next iterate is obtained by projecting the
#             current iterate onto the hyperplane <a_r, x> = y_r. a_r is
#             the rth measurement row of the matrix A.

# METHOD:
#         1.) The method is described in algorithm 3 in the paper.

# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

# -----------------------------START-----------------------------------
import sys
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/solvers')
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/initializers')
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/util')

import time
import struct

import numpy as np
from numpy import dot
from numpy.linalg import norm
from display_verbose_output import displayVerboseOutput
from generate_outputs import generateOutputs
from check_if_number import checkIfNumber
from stop_now import stopNow
from check_if_in_list import checkIfInList


def solveKaczmarzSimple(A=None, At=None, b0=None, x0=None, opts=None, *args, **kwargs):

    validateOptions(opts)

    # Initialization
    m = len(b0)

    sol = x0
    maxDiff = - np.inf

    currentTime = []
    currentResid = []
    currentReconError = []
    currentMeasurementError = []

    solveTimes, measurementErrors, reconErrors, residuals = initializeContainers(
        opts, nargout=4)
    startTime = time.time

    for iter in range(1, opts.maxIters):
        if 'random' == opts.indexChoice.lower():
            index = randi(m)
        else:
            if 'cyclic' == opts.indexChoice.lower():
                index = mod(iter - 1, m) + 1
        # Obtain measurement vector a_i as column vector
        a = At(double(arange(1, m == index)).T)
        y = b0(index)
        product = dot(a, sol)
        newSol = sol + \
            dot(((dot(y, np.sign(product)) - product) / norm(a) ** 2), a)
        diff = norm(newSol - sol)
        sol = newSol
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
    validIndexChoices = cellarray(['cyclic', 'random'])
    checkIfInList('indexChoice', opts.indexChoice, validIndexChoices)
    return
