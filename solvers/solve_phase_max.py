#                           solvePhaseMax.m
#
#  Implementation of the PhaseMax algorithm proposed in the paper using
#  FASTA. Note: For this code to run, the solver "fasta.m" must be in your
#  path.
#
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
#
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
#  See the script 'testPhaseMaxGaussian.m' and 'testPhaseMaxFourier.m' for
#  two examples of proper usage of this function.
#
# Notations
#
#
# Algorithm Description
#  Solve the PhaseMax signal reconstruction problem
#         maximize <x0,x>
#         subject to |Ax|<=b0
#
#  The problem is solved by approximately enforcing the constraints using a
#  quadratic barrier function.  A continuation method is used to increase
#  the strength of the barrier function until a high level of numerical
#  accuracy is reached.
#  The objective plus the quadratic barrier has the form
#     <-x0,x> + 0.5*max{|Ax|-b,0}^2.
#
#  For a detailed explanation, see the PhaseMax paper referenced below. For
#  more details about FASTA, see the FASTA user guide, or the paper "A
#  field guide to forward-backward splitting with a FASTA implementation."
#
# References
#  Paper Title:   PhaseMax: Convex Phase Retrieval via Basis Pursuit
#  Authors:       Tom Goldstein, Christoph Studer
#  arXiv Address: https://arxiv.org/abs/1610.07531
#
#  Copyright Goldstein & Studer, 2016.  For more details, visit
#  https://www.cs.umd.edu/~tomg/projects/phasemax/

import sys
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/solvers')
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/initializers')
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/util')
import struct
import numpy as np
from numpy import dot
import math
from display_verbose_output import displayVerboseOutput
import time
from generate_outputs import generateOutputs
from initialize_containers import initializeContainers
from fasta import fasta


def solvePhaseMax(A=None, At=None, b0=None, x0=None, opts=None, *args, **kwargs):
    # Initialization
    m = len(b0)
    n = len(x0)
    remainIters = opts.maxIters
    # print(opts.maxIters, "opts.maxIters")

    # It's initialized to opts.maxIters.
    #  Normalize the initial guess relative to the number of measurements
    # print(type(x0),type(b0))
    x0 = dot(dot(dot((x0 / np.linalg.norm(x0.flatten('F'))),
                     np.mean(b0.flatten('F'))), (m / n)), 100)

    #  re-scale the initial guess so that it approximately satisfies |Ax|=b
    # sol = x0.* min(b0./abs(A(x0)))
    # print(A.shape,x0.shape,b0.shape,dot(A, x0).shape)
    # print(np.min(b0 / np.abs(dot(A, x0))))
    sol = np.multiply(x0, np.min(b0 / np.abs(dot(A, x0))))
    ending = 0
    itera = 0
    currentTime = []
    currentResid = []
    currentReconError = []
    currentMeasurementError = []

    solveTimes, measurementErrors, reconErrors, residuals = initializeContainers(
        opts)

    f = lambda z=None: dot(0.5, np.linalg.norm(np.max(np.abs(z) - b0, 0)) ** 2)

    gradf = lambda z=None: (np.multiply(np.sign(z), np.max(np.abs(z) - b0, 0)))

    # Options to hand to fasta
    fastaOpts = struct
    fastaOpts.maxIters = opts.maxIters
    # fastaOpts.stopNow = lambda x=None, iter=None, resid=None,
    #                          normResid = None, maxResid = None, opts = None: processIteration(x, resid)

    # solveTime, residual and error at each iteration.
    fastaOpts.verbose = 0
    startTime = time.time

    constraintError = np.linalg.norm(abs(dot(A, sol)) - b0)

    while remainIters > (0 & (not (ending))):
        g = lambda x=None: - np.real(dot(x0.T, x))
        # proxg = @(x,t) x+t*x0; 
        proxg = lambda x=None, t=None: x + t*x0
        fastaOpts.tol = np.linalg.norm(x0) / 100
        # Call FASTA to solve the inner minimization problem
        #  [sol, fastaOuts] = fasta(A, At, f, gradf, g, proxg, sol, fastaOpts);  % Call a solver to minimize the quadratic barrier problem
        # def fasta(A=None, At=None, f=None, gradf=None, g=None, proxg=None, x0=None, opts=None, *args, **kwargs):
        
        sol, _, fastaOuts = fasta(A, At, f, gradf, g, proxg, sol, fastaOpts)
        # fastaOpts.tau = fastaOuts.stepsizes(end);     % Record the most recent stepsize for recycling.
        fastaOpts.tau = fastaOuts.stepsizes
        x0 = x0 / 10
        # Update the max number of iterations for fasta
        remainIters = remainIters - fastaOuts.iterationCount
        fastaOpts.maxIters = min(opts.maxIters, remainIters)
        # newConstraintError = np.linalg.norm(max(abs(A(sol)) - b0, 0))
        newConstraintError = np.linalg.norm(np.max(np.abs(dot(A,sol)) - b0, 0))

        relativeChange = abs(
            constraintError - newConstraintError) / np.linalg.norm(b0)
        if relativeChange < opts.tol:
            break
        constraintError = newConstraintError

    # Create output according to the options chosen by user
    outs = generateOutputs(opts, itera, solveTimes,
                           measurementErrors, reconErrors, residuals)

    if opts.verbose == 1:
        displayVerboseOutput(itera, currentTime, currentResid,
                             currentReconError, currentMeasurementError)

    # Runs code upon each FASTA iteration. Returns whether FASTA should
    # terminate.

    def processIteration(x=None, residual=None):
        itera = itera + 1

        # If xt is provided, reconstruction error will be computed and used for stopping
        # condition. Otherwise, residual will be computed and used for stopping
        # condition.
        if not(isempty(opts.xt)):
            xt = opts.xt
            alpha = (dot(ravel(x).T, ravel(xt))) / \
                (dot(ravel(x).T, ravel(x)))
            x = dot(alpha, x)
            currentReconError = np.linalg.norm(x - xt) / np.linalg.norm(xt)
            if opts.recordReconErrors:
                reconErrors[itera] = currentReconError

        if opts.xt:
            currentResid = residual

        if opts.recordResiduals:
            residuals[itera] = residual

        currentTime = time.time

        if opts.recordTimes:
            solveTimes[itera] = currentTime

        if opts.recordMeasurementErrors:
            currentMeasurementError = np.linalg.norm(
                abs(A(sol)) - b0) / np.linalg.norm(b0)
            measurementErrors[itera] = currentMeasurementError

        # Display verbose output if specified
        if opts.verbose == 2:
            displayVerboseOutput(
                itera, currentTime, currentResid, currentReconError, currentMeasurementError)

        # Test stopping criteria.
        stop = False
        if currentTime > opts.maxTime:
            stop = True

        if not(isempty(opts.xt)):
            assert(not(isempty(currentReconError)),
                   'If xt is provided, currentReconError must be provided.')
            stop = stop or currentReconError < opts.tol
            ending = stop

        stop = stop or residual < fastaOpts.tol

        return stop
    return sol, outs
