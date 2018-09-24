#                           solvePhaseLift.m

#  Implementation of the PhaseLift algorithm using full-scale semidefinite
#  programming.

# I/O
#  Inputs:
#     A:    m x n matrix or a function handle to a method that
#           returns A*x.
#     At:   The adjoint (transpose) of 'A'. If 'A' is a function handle,
#           'At' must be provided.
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
#  See the script 'runPhaseLift.m' for an example of proper usage of this
#  function.

# Notations
#  X = x*x' is the lifted version of the unknown signal x.
#  b = b0.^2 is the element-wise square of the measurements b0.
#  AL is a function handle that takes X as input and outputs b.
#
# Algorithm Description
#  PhaseLift algorithm reformualates the PR problem as a convex problem by
#  lifting the dimension of the unknown signal x. The problem becomes:
#  minimize Tr(X) subject to AL(X)=b and X is a positive semidefinite
#  matrix.
#
#  More specifically,
#  Solve the problem
#           min  mu||X||_nuc +.5||A(X)-b||^2
#                X>=0
#  where X is a square symmetric matrix,||X||_nuc is the nuclear (trace)
#  norm, A is a linear operator, and X>=0 denotes that X
#  must lie in the positive semidefinite cone.

#  The unknown signal x can be recovered from its lifted version X by
#  factorization.

#  The solver has three steps
#  (1) Lifting x to X=xx' and create AL and its transpose AtL using A and At.
#  (2) Use FASTA to solve this convex optimization problem.
#  (3) Take the principle eigenvector of X and re-scale it to capture
#      the correct ammount of energy.
#
#  For a detailed explanation, see the PhaseLift paper referenced below.
#  For more details about FASTA, see the FASTA user guide, or the paper "A
#  field guide to forward-backward splitting with a FASTA implementation."

# References
#  Paper Title:   PhaseLift: Exact and Stable Signal Recovery from
#                 Magnitude Measurements via Convex Programming
#  Place:         Chapter 2.3
#  Authors:       Emmanuel J. Candes, Thomas Strohmer, Vladislav Voroninski
#  arXiv Address: https://arxiv.org/abs/1109.4499
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017


# -----------------------------START-----------------------------------

import sys
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/solvers')
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/initializers')
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/util')
import math
import struct
import time

import numpy as np
from numpy import diag as diag
from numpy import dot as dot
from numpy.linalg import norm as norm

from fasta import fasta
from display_verbose_output import displayVerboseOutput
from generate_outputs import generateOutputs
from stop_now import stopNow
from check_if_number import checkIfNumber

def solvePhaseLift(A=None, At=None, b0=None, x0=None, opts=None, *args, **kwargs):
    # Initialization
    muFinal = opts.regularizationPara

    m = len(b0)

    n = np.size(x0)

    b = b0 ** 2

    X0 = np.dot(x0, x0.T)

    #  We need direct access to the entires of A, so convert a function
    #  handle to a dense matrix
    if not(A.isnumeric()):
        A = A(np.eye(np.size(x0, 1)))

    # Initialize values potentially computed at each round.
    iter = 0
    currentTime = []
    currentResid = []
    currentReconError = []
    currentMeasurementError = []

    solveTimes, measurementErrors, reconErrors, residuals = initializeContainers(
        opts, nargout=4)

    # These function compute the linear measurement operator on the lifted
    # matrix, and it's adjoint
    AL = lambda X=None: sum(np.multiply((dot(A, X)), conj(A)), 2)
    AtL = lambda y=None: dot(A.T, (np.multiply(A, (dot(y, np.ones(1, n))))))

    # norm term).  Note that norm(eig(AtL(b0)),1) is the regularizer that
    # produces a zero solution.
    mu = dot(norm(eig(AtL(b0)), 1), opts.tol)

    # than the vectorized matrix version.
    # It can probably be further optimized by
    # 1.loop roll-out.
    # 2.AL0 can take advantage of the factorization of X.

    # Define ingredients for FASTA
    # m x 1 -> 1 x 1
    # f(y) = .5 ||y - b||^2
    f = lambda y=None: dot(0.5, norm(ravel(y) - ravel(b)) ** 2)

    gradf = lambda y=None: ravel(y) - ravel(b)

    # g(z) = mu||X||_nuc, plus characteristic function of the SDP cone
    g = lambda X=None: dot(mu, norm(eig(X), 1))

    # proxg(z,t) = argmin t*mu*nuc(x)+.5||x-z||^2, with x in SDP cone
    proxg = lambda X=None, t=None: projectSemiDefCone(X, dot(mu, t))

    fastaOpts = struct
    fastaOpts.tol = opts.tol
    fastaOpts.maxIters = opts.maxIters
    fastaOpts.accelerate = False
    fastaOpts.stopNow = lambda x=None, iter=None, resid=None, normResid=None, maxResid=None, opts=None: processIteration(
        x, normResid)

    # solveTime, residual and error at each iteration.

    # Call solver
    startTime = time.time
    Xest, outs = fasta(AL, AtL, f, gradf, g, proxg, X0, fastaOpts, nargout=2)
    sol = recoverSignal(Xest, n)

    outs = generateOutputs(opts, iter, solveTimes,
                           measurementErrors, reconErrors, residuals)

    if opts.verbose == 1:
        displayVerboseOutput(iter, currentTime, currentResid,
                             currentReconError, currentMeasurementError)

    # Runs code upon each FASTA iteration. Returns whether FASTA should
    # terminate.

    def processIteration(x=None, normResid=None, *args, **kwargs):
        iter = iter + 1

        # If xt is provided, reconstruction error will be computed and used for stopping
        # condition. Otherwise, residual will be computed and used for stopping
        # condition.
        if not(opts.xt == None):
            xt = opts.xt
            xt = dot(xt, xt.T)
            alpha = (dot(ravel(x).T, ravel(xt))) / (dot(ravel(x).T, ravel(x)))
            x = dot(alpha, x)
            currentReconError = norm(x - xt) / norm(xt)
            if opts.recordReconErrors:
                reconErrors[iter] = currentReconError

        if opts.xt == None | opts.recordResiduals:
            currentResid = normResid

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
            displayVerboseOutput(iter, currentTime, currentResid,
                                 currentReconError, currentMeasurementError)

        # Test stopping criteria.
        stop = stopNow(opts, currentTime, currentResid, currentReconError)

        return stop

        # Recover solution using method recommended by PhaseLift authors
        #  The solution matrix may not be rank 1.  In this case, we use the
        #  principle eigenvector and re-scale it to capture the correct ammount
        #  of energy.

    def recoverSignal(Xest=None, n=None, *args, **kwargs):
        #  Extract X
        V, D = eig(reshape(Xest, concat([n, n])), nargout=2)

        val, ind = max(diag(D), nargout=2)
        recovered = dot(V(arange(), ind), math.sqrt(D(ind, ind)))

        # have other eigenvectors
        lifted = dot(recovered, recovered.T)
        scale = norm(b) / norm(AL(lifted))

        sol = dot(recovered, scale)

        return sol
    return sol, outs


# n x n -> n x n
# proxg(z,t) = argmin t*mu*nuc(x)+.5||x-z||^2, with x in SDP cone
def projectSemiDefCone(X=None, delta=None, *args, **kwargs):
    V, D = eig(X, nargout=2)
    D = max(D.real() - delta, 0)
    X = dot(dot(V, D), V.T)

    return X


# Check the validify of algorithm specific options
def validateOptions(opts=None, *args, **kwargs):
    checkIfNumber('regularizationPara', opts.regularizationPara)

    return
