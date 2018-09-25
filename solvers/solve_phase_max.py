import sys
sys.path.append(u'../util')
sys.path.append(u'../initializers')
sys.path.append(u'../solvers')

import struct
import numpy as np
from numpy import dot
from numpy.linalg import norm
import math
from display_verbose_output import displayVerboseOutput
import time
from generate_outputs import generateOutputs
from initialize_containers import initializeContainers
from fasta import fasta


def solvePhaseMax(A=None, At=None, b0=None, x0=None, opts=None):
    # Initialization
    m = len(b0)
    n = len(x0)
    remainIters = opts.maxIters

    # It's initialized to opts.maxIters.
    # %  Normalize the initial guess relative to the number of measurements
    #  Normalize the initial guess relative to the number of measurements
    x0 = dot((x0 / norm(x0.flatten('F'))),
             np.mean(b0.flatten('F'))) * (m / n)*100
    #  re-scale the initial guess so that it approximately satisfies |Ax|=b
    sol = np.multiply(x0, np.min(b0 / np.abs(dot(A, x0))))
    ending = 0
    itera = 0
    currentTime = []
    currentResid = []
    currentReconError = []
    currentMeasurementError = []

    solveTimes, measurementErrors, reconErrors, residuals = initializeContainers(
        opts)

    f = lambda z=None: dot(0.5, norm(np.max(np.abs(z) - b0, 0)) ** 2)

    gradf = lambda z=None: (np.multiply(np.sign(z), np.max(np.abs(z) - b0, 0)))

    # Options to hand to fasta
    fastaOpts = struct
    fastaOpts.maxIters = opts.maxIters
    fastaOpts.stopNow = lambda x=None, itera=None, resid=None, normResid = None, maxResid = None, opts = None: processIteration(
        x, resid)

    startTime = time.time  # Start timer
    fastaOpts.verbose = 0
    constraintError = norm(abs(dot(A, sol)) - b0)
    while (remainIters > 0) & (not (ending)):
        g = lambda x=None: - np.real(dot(x0.T, x))
        proxg = lambda x=None, t=None: x + t*x0
        fastaOpts.tol = norm(x0) / 100
        # Call FASTA to solve the inner minimization problem
        sol, _, fastaOuts = fasta(A, At, f, gradf, g, proxg, sol, fastaOpts)
        fastaOpts.tau = fastaOuts.stepsizes
        x0 = x0 / 10
        # Update the max number of iterations for fasta
        remainIters = remainIters - fastaOuts.iterationCount
        fastaOpts.maxIters = min(opts.maxIters, remainIters)
        newConstraintError = norm(np.max(np.abs(dot(A, sol)) - b0, 0))

        relativeChange = abs(constraintError - newConstraintError) / norm(b0)
        if relativeChange < opts.tol:
            break
        constraintError = newConstraintError

    # Create output according to the options chosen by user
    outs = generateOutputs(opts, itera, solveTimes,
                           measurementErrors, reconErrors, residuals)

    if opts.verbose == 1:
        displayVerboseOutput(itera, currentTime, currentResid,
                             currentReconError, currentMeasurementError)

    # Runs code upon each FASTA iteration. Returns whether FASTA should terminate.
    def processIteration(x=None, residual=None):
        itera = itera + 1

        # Record convergence information and check stopping condition, If xt is provided, reconstruction error will be computed and used for stopping condition. Otherwise, residual will be computed and used for stopping condition.
        if opts.xt:
            xt = opts.xt
            # Compute optimal rotation
            alpha = (dot(x.flatten('F').T, xt.flatten('F'))) / \
                (dot(x.flatten('F').T, x.flatten('F')))
            x = dot(alpha, x)
            currentReconError = norm(x - xt) / norm(xt)
            if opts.recordReconErrors:
                reconErrors[itera] = currentReconError

        if opts.xt == None:
            currentResid = residual

        if opts.recordResiduals:
            residuals[itera] = residual

        currentTime = time.time - startTime  # Record elapsed time so far

        if opts.recordTimes:
            solveTimes[itera] = currentTime

        if opts.recordMeasurementErrors:
            currentMeasurementError = norm(
                abs(A(sol)) - b0) / norm(b0)
            measurementErrors[itera] = currentMeasurementError

        # Display verbose output if specified
        if opts.verbose == 2:
            displayVerboseOutput(
                itera, currentTime, currentResid, currentReconError, currentMeasurementError)

        # Test stopping criteria.
        stop = False
        if currentTime > opts.maxTime:  # Stop if we're run over the max runtime
            stop = True

        if not(opts.xt == None):  # If true solution is specified, terminate when close to true solution
            # assert(not((currentReconError==None),'If xt is provided, currentReconError must be provided.')
            stop = stop
            ending = stop  # When true, this flag will terminate outer loop
        stop = stop

        return stop
    return sol, outs
