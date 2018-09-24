
#                           solveGerchbergSaxton.m

#  Solver for Gerchberg-Saxton algorithm.
#
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
#  See the script 'testGerchbergSaxton.m' for an example of proper usage of
#  this function.

# Notations
#  z is the spectral magnitude on the fourier domain(z = b0.*sign(Ax)).
#  It has the phase of the fourier transform Ax of the estimation
#  and the magnitude of the measurements b0.
#
# Algorithm Description
#  One simply transforms back and forth between the two domains, satisfying
#  the constraints in one before returning to the other. The method has
#  three steps
# (1) Left multipy the current estimation x by the measurement
#  matrix A and get Ax.
# (2) Keep phase, update the magnitude using the
#  measurements b0, z = b0.*sign(Ax).
# (3) Solve the least-squares problem
#           sol = \argmin ||Ax-z||^2
#      to get our new estimation x. We use Matlab built-in solver lsqr()
#      for this least square problem.
#  (4) Impose temporal constraints on x(This step is ignored now since we
#      don't assume constraints on our unknown siganl)

#  For a detailed explanation, see the paper referenced below.

# References
#  The paper that this implementation follows
#  Paper Title:   Fourier Phase Retrieval: Uniqueness and Algorithms
#  Place:         Chapter 4.1 Alternating projection algorithms, Algorithm 1
#  Authors:       Tamir Bendory, Robert Beinert, Yonina C. Eldar
#  arXiv Address: https://arxiv.org/abs/1705.09590
#
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


def solveGerchbergSaxton(A=None, At=None, b0=None, x0=None, opts=None, *args, **kwargs):
    validateOptions(opts)

    # Initialization
    sol = x0

    # Initialize values potentially computed at each round.
    currentTime = []
    currentResid = []
    currentReconError = []
    currentMeasurementError = []

    solveTimes, measurementErrors, reconErrors, residuals = initializeContainers(
        opts, nargout=4)

    def Afun(x=None, transp_flag=None, *args, **kwargs):

        if strcmp(transp_flag, 'transp'):
            y = At(x)
        else:
            if strcmp(transp_flag, 'notransp'):
                y = A(x)

        return y

    startTime = time.time

    z = np.multiply(b0, np.sign(A(sol)))

    for iter in range(1, opts.maxIters):
        # Solve the least-squares problem
        #  sol = \argmin ||Ax-z||^2.
        # If A is a matrix,
        #  sol = inv(A)*z
        # If A is a fourier transform( and measurements are not oversampled i.e. m==n),
        #  sol = inverse fourier transform of z
        # Use the evalc() to capture text output, thus preventing
        # the conjugate gradient solver from printing to the screen.
        evalc('sol=lsqr(@Afun,z,opts.tol/100,opts.maxInnerIters,[],[],sol)')
        Ax = A(sol)
        z = np.multiply(b0, np.sign(Ax))
        # Record convergence information and check stopping condition
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
        if isempty(opts.xt) or opts.recordResiduals:
            currentResid = norm(At(Ax - z)) / norm(z)
        if opts.recordResiduals:
            residuals[iter] = currentResid
        currentTime = time.time
        if opts.recordTimes:
            solveTimes[iter] = currentTime
        if opts.recordMeasurementErrors:
            currentMeasurementError = norm(abs(Ax) - b0) / norm(b0)
            measurementErrors[iter] = currentMeasurementError
        # Display verbose output if specified
        if opts.verbose == 2:
            displayVerboseOutput(
                iter, currentTime, currentResid, currentReconError, currentMeasurementError)
        # Test stopping criteria.
        if stopNow(opts, currentTime, currentResid, currentReconError):
            break

    # Create output according to the options chosen by user
    outs = generateOutputs(opts, iter, solveTimes,
                           measurementErrors, reconErrors, residuals)

    if opts.verbose == 1:
        displayVerboseOutput(iter, currentTime, currentResid,
                             currentReconError, currentMeasurementError)

    return y

    # Check the validify of algorithm specific options


def validateOptions(opts=None, *args, **kwargs):
    checkIfNumber('maxInnerIters', opts.maxInnerIters)
    return
