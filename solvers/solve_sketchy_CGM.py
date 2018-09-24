#                           solveSketchyCGM.m

#  Solver for the skechyCGM algorithm.
#
# I/O
#  Inputs:
#     A:    m x n matrix or a function handle to a method that
#           returns A*x.
#     At:   The adjoint (transpose) of 'A'. If 'A' is a function handle,
#           'At' must be provided.
#     b0:   m x 1 real,non-negative vector consists of all the
#           measurements.
#     x0:   n x 1 vector. It is the initial guess of the unknown signal x.
#     opts: A struct consists of the options for the algorithm.

# For details, see header in solvePhaseRetrieval.m or the User Guide.

#     Note: When a function handle is used, the value of 'At' (a function
#     handle for the adjoint of 'A') must be supplied.
#
#  Outputs:
#     sol:  n x 1 vector. It is the estimated signal.
#     outs: A struct consists of the convergence info.

# For details, see header in solvePhaseRetrieval.m or the User Guide.
#
#
#  See the script 'testSketchyCGM.m' for an example of proper usage of this
#  function.

# Notations
#  B.3.1 X = xx' is the lifted version of the unknown signal x. The linear
#  operator curly_A: R(nxn) -> R(d). And its adjoint, curly_As: R(d) ->
#  R(nxn)
#                 curly_A(X) = diag(A*X*A') curly_As(z) = A' * diag(z)' A

# Algorithm Description
#  SketchyCGM modifies a standard convex optimization scheme, the
#  conditional gradient method, to store only a small randomized sketch of
#  the matrix variable. After the optimization terminates, the algorithm
#  extracts a low-rank approximation of the solution from the sketch.

#  This algorithm solves the problem:
#                 minimize f(curly_A(X)) s.t. ||X||_N <= \alpha where
#                 ||X||_N is the nuclear norm of X i.e. trace(X) and \alpha
#                 is a constant value (see section 1.2 of SketchCGM paper
#                 for it).
#
#  Specifically, the method has the following five steps(section 5.1 of the
#  paper): (1) Initialize iterate z, two random matrices Psi and Omega, two
#  sketches Y and W. (2) At each iteration, compute an update direction via
#  Lanczos or via randomized SVD (3) Update the iterate and the two
#  sketches Y and W (4) The iteration continues until it triggers the
#  stopping condition(i.e. the normalized gradient is smaller than a given
#  tolerance). (5) Form a rank-r approximate solution X of the model
#  problem and reconstruct final solution x from it.
#
#  Note: The suboptimality eps used in the paper for stopping condition
#  doesn't work well in our tests so we use normalized gradient instead.
#
#  For a detailed explanation, see the paper referenced below.

# References
#  Paper Title:   Sketchy Decisions: Convex Low-Rank Matrix Optimization
#  with Optimal Storage Place:         Chapter 5.4, Algorithm 1 Authors:
#  Alp Yurtsever, Madeleine Udell, Joel A. Tropp, Volkan Cevher arXiv
#  Address: https://arxiv.org/abs/1702.06838
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein Copyright (c) University of Maryland,
# 2017

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


def solveSketchyCGM(A=None, At=None, b0=None, x0=None, opts=None, *args, **kwargs):
    validateOptions(opts)

    # Initialize test setup
    n = len(x0)

    m = len(b0)

    b = b0 ** 2

    r = opts.rank

    # "sketch" Y and W and random matrices Psi and Omega.
    # So the approximate solution will have rank r.
    z = np.zeros(m, 1)

    alpha = sum(b) / m

    maxNormGradfz = 1e-30

    # Initialize values potentially computed at each round.
    currentTime = []
    currentResid = []
    currentReconError = []
    currentMeasurementError = []
    sol = x0

    solveTimes, measurementErrors, reconErrors, residuals = initializeContainers(
        opts, nargout=4)

    Omega, Psi, Y, W = sketchyInit(m, n, r, nargout=4)

    curly_A_of_uut = lambda u=None: sum(abs(A(u)) ** 2, 2)

    # where X = uu'
    curly_As = lambda z=None: lambda x=None: At(np.multiply(z, A(x)))

    # for the operator/matrix A'*diag(z)*A
    # gradient of the objective function
    # .5 * ||z - b||^2
    grad_f = lambda z=None: z - b

    Y0 = Y
    W0 = W

    eta = opts.eta
    failCounter = 0
    startTime = time.time

    # Start SketchyCGM iterations
    for iter in range(1, opts.maxIters):
        grad_fz = grad_f(z)
        normGradfz = norm(grad_fz)
        if normGradfz > maxNormGradfz:
            maxNormGradfz = normGradfz
        # Return the minimal eigenvalue and eigenvector
        lambda_, u = MinEig(curly_As(grad_fz), n, nargout=2)
        if lambda_ > 0:
            h = 0
            u = 0
        else:
            h = dot(alpha, curly_A_of_uut(u))
        # Else continue update
        z = dot((1 - eta), z) + dot(eta, h)
        grad_fz = grad_f(z)
        Y, W = sketchyUpdate(dot(- alpha, u), - u, eta, Y,
                             W, Omega, Psi, nargout=2)
        currentResid = norm(grad_fz) / maxNormGradfz
        residuals[iter] = currentResid
        # Select stepsize
        # During optimization, cut the stepsize if the errors don't go
        # down.
        if iter > 1 and residuals(iter) > residuals(iter - 1):
            failCounter = failCounter + 1
        if failCounter > 3:
            eta = eta / 2
            failCounter = 0
        # ---------------------------------------------------------------------------
        # Record convergence information and check stopping condition
        # If xt is provided, reconstruction error will be computed and used for stopping
        # condition. Otherwise, residual will be computed and used for stopping
        # condition.
        if logical_not(isempty(opts.xt)):
            if iter == 1:
                sol = x0
            else:
                sol = recoverSignal(Y, W, Psi, r)
            x = sol
            xt = opts.xt
            alpha2 = (dot(ravel(x).T, ravel(xt))) / (dot(ravel(x).T, ravel(x)))
            x = dot(alpha2, x)
            currentReconError = norm(x - xt) / norm(xt)
            if opts.recordReconErrors:
                reconErrors[iter] = currentReconError
        currentTime = time.time
        if opts.recordTimes:
            solveTimes[iter] = currentTime
        if opts.recordMeasurementErrors:
            if iter == 1:
                sol = x0
            else:
                sol = recoverSignal(Y, W, Psi, r)
            currentMeasurementError = norm(abs(A(sol)) - b0) / norm(b0)
            measurementErrors[iter] = currentMeasurementError
        # Display verbose output if specified
        if opts.verbose == 2:
            displayVerboseOutput(
                iter, currentTime, currentResid, currentReconError, currentMeasurementError)
        #  Test stopping criteria.
        if stopNow(opts, currentTime, currentResid, currentReconError):
            break
        # ---------------------------------------------------------------------------

    # Recover the signal from two sketches Y, W and matrix Psi
    sol = recoverSignal(Y, W, Psi, r)

    outs = generateOutputs(opts, iter, solveTimes,
                           measurementErrors, reconErrors, residuals)

    if opts.verbose == 1:
        displayVerboseOutput(iter, currentTime, currentResid,
                             currentReconError, currentMeasurementError)

    return sol, outs

    # Helper Functions
# Initialize Sketchy parameters: random matrices Omega and Psi, sketches Y and W


def sketchyInit(m=None, n=None, r=None, *args, **kwargs):

    k = dot(2, r) + 1
    l = dot(4, r) + 3
    Omega = randn(n, k)
    Psi = randn(l, n)
    Y = np.zeros(n, k)
    W = np.zeros(l, n)
    return Omega, Psi, Y, W

    # Update the sketch


def sketchyUpdate(u=None, v=None, eta=None, Y=None, W=None, Omega=None, Psi=None, *args, **kwargs):

    Y = dot((1 - eta), Y) + dot(dot(eta, u), (dot(v.T, Omega)))
    W = dot((1 - eta), W) + dot(dot(eta, (dot(Psi, u))), v.T)
    return Y, W

    # Reconstruct the lifted solution U*S*V = xx' from sketch


def sketchyReconstruct(Y=None, W=None, Psi=None, r=None, *args, **kwargs):
    Q = orth(Y)
    B = np.linalg.solve((dot(Psi, Q)), W)
    U, S, V = svds(B, r, nargout=3)
    U = dot(Q, U)
    return U, S, V

    #  Return the minimal (hopefully negative) eigenvalue and eigenvector


def MinEig(matrix=None, n=None, *args, **kwargs):

    eigs_opts.isreal = False
    u, lambda_ = eigs(matrix, n, 1, 'LM', eigs_opts, nargout=2)

    if lambda_ > 0:
        u, lambda1 = eigs(lambda x=None: matrix(
            x) - dot(lambda_, x), n, 1, 'LM', eigs_opts, nargout=2)
        lambda_ = lambda1 + lambda_

    return lambda_, u

    # Recover the signal from two sketches Y, W and matrix Psi


def recoverSignal(Y=None, W=None, Psi=None, r=None, *args, **kwargs):
    # Reconstruct the lifted solution X_hat = xx*
    U, S, V = sketchyReconstruct(Y, W, Psi, r, nargout=3)

    sol = dot(U, math.sqrt(S))
    return sol

    # Check the validify of algorithm specific options


def validateOptions(opts=None, *args, **kwargs):

    checkIfNumber('rank for sketchuyCGM', opts.rank)
    return
