#                           solveFienup.m

#  Solver for Fienup algorithm.
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
#  See the script 'testFienup.m' for an example of proper usage of this
#  function.

# Notations
#  The notations mainly follow those used in Section 2 of the Fienup paper.
#  gk:    g_k   the guess to the signal before the k th round
#  gkp:   g_k'  the approximation to the signal after the k th round of
#         iteration
#  gknew: g_k+1 the guess to the signal before the k+1 th round
#  Gkp:   G_k'  the approximation to fourier transfor of the signal after
#               satisfying constraints on fourier-domain
#  beta:  \beta the Tuning parameter for object-domain update
#
# Algorithm Description
#  Fienup Algorithm is the same as Gerchberg-Saxton Algorithm except when
#  the signal is real and non-negative (or has constraint in general). When
#  this happens, the update on the object domain is different.
#
#  Like Gerchberg-Saxton, Fienup transforms back and forth between the two
#  domains, satisfying the constraints in one before returning to the
#  other. The method has four steps (1) Left multipy the current estimation
#  x by the measurement matrix A and get Ax. (2) Keep phase, update the
#  magnitude using the measurements b0, z = b0.*sign(Ax). (3) Solve the
#  least-squares problem
#           sol = \argmin ||Ax-z||^2
#      to get our new estimation x. We use Matlab built-in solver lsqr()
#      for this least square problem.
#  (4) Impose temporal constraints on x(This step is ignored when there is
#  no constraints)

#  For a detailed explanation, see the Fienup paper referenced below.

# References
#  Paper Title:   Phase retrieval algorithms: a comparison
#  Place:         Section II for notation and Section V for the
#                 Input-Output Algorithm
#  Authors:       J. R. Fienup
#  Address: https://www.osapublishing.org/ao/abstract.cfm?uri=ao-21-15-2758
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


def solveFienup(A=None, At=None, b0=None, x0=None, opts=None, *args, **kwargs):
    validateOptions(opts)

    # Initialization
    gk = x0

    gkp = x0

    gknew = x0

    beta = opts.FienupTuning

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

    for iter in range(1, opts.maxIters):
        Ax = A(gk)
        Gkp = np.multiply(b0, np.sign(Ax))
        # -----------------------------------------------------------------------
        # Record convergence information and check stopping condition
        # If xt is provided, reconstruction error will be computed and used for stopping
        # condition. Otherwise, residual will be computed and used for stopping
        # condition.
        if logical_not(isempty(opts.xt)):
            x = gk
            xt = opts.xt
            alpha = (dot(ravel(x).T, ravel(xt))) / (dot(ravel(x).T, ravel(x)))
            x = dot(alpha, x)
            currentReconError = norm(x - xt) / norm(xt)
            if opts.recordReconErrors:
                reconErrors[iter] = currentReconError
        if isempty(opts.xt) or opts.recordResiduals:
            currentResid = norm(At(Ax - Gkp)) / norm(Gkp)
        if opts.recordResiduals:
            residuals[iter] = currentResid
        currentTime = time.time
        if opts.recordTimes:
            solveTimes[iter] = currentTime
        if opts.recordMeasurementErrors:
            currentMeasurementError = norm(abs(A(gk)) - b0) / norm(b0)
            measurementErrors[iter] = currentMeasurementError
        # Display verbose output if specified
        if opts.verbose == 2:
            displayVerboseOutput(
                iter, currentTime, currentResid, currentReconError, currentMeasurementError)
        #  Test stopping criteria.
        if stopNow(opts, currentTime, currentResid, currentReconError):
            break
        # -----------------------------------------------------------------------
        # Solve the least-squares problem
        #  gkp = \argmin ||Ax-Gkp||^2.
        # If A is a matrix,
        #  gkp = inv(A)*Gkp
        # If A is a fourier transform( and measurements are not oversampled i.e. m==n),
        #  gkp = inverse fourier transform of Gkp
        # Use the evalc() to capture text output, thus preventing
        # the conjugate gradient solver from printing to the screen.
        evalc('gkp=lsqr(@Afun,Gkp,opts.tol/100,opts.maxInnerIters,[],[],gk)')
        # following the constraint
        if (opts.isComplex == False) & (opts.isNonNegativeOnly == True):
            inds = gkp < 0
            # May also need to check if isreal
            inds2 = not(inds)
            # hybrid input-output (see Section V, Equation (44))
            gknew[inds] = gk(inds) - dot(beta, gkp(inds))
            gknew[inds2] = gkp(inds2)
        else:
            gknew = gkp
        gk = gknew

    sol = gk

    outs = generateOutputs(opts, iter, solveTimes,
                           measurementErrors, reconErrors, residuals)

    if opts.verbose == 1:
        displayVerboseOutput(iter, currentTime, currentResid,
                             currentReconError, currentMeasurementError)

    return y


# Check the validify of algorithm specific options
def validateOptions(opts=None, *args, **kwargs):
    checkIfNumber('FienupTuning', opts.FienupTuning)
    return
