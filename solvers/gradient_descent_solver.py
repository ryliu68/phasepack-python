# -------------------------gradientDescentSolver.m--------------------------------
#
# General routine used by phase retrieval algorithms that function by using
# line search methods. This function is internal and should not be called
# by any code outside of this software package.

# The line search approach first finds a descent direction along which the
# objective function f will be reduced and then computes a step size that
# determines how far x  should move along that direction. The descent
# direction can be computed by various methods, such as gradient descent,
# Newton's method and Quasi-Newton method. The step size can be determined
# either exactly or inexactly.

# This line search algorithm implements the steepest descent, non linear
# conjugate gradient, and the LBFGS method. Set the option accordingly as
# described below.
#
# Aditional Parameters
# The following are additional parameters that are to be passed as fields
# of the struct 'opts':
#
#     maxIters (required) - The maximum number of iterations that are
#     allowed to
#         occur.
#
#     maxTime (required) - The maximum amount of time in seconds the
#     algorithm
#         is allowed to spend solving before terminating.
#
#     tol (required) - Positive real number representing how precise the
#     final
#         estimate should be. Lower values indicate to the solver that a
#         more precise estimate should be obtained.
#
#     verbose (required) - Integer representing whether / how verbose
#         information should be displayed via output. If verbose == 0, no
#         output is displayed. If verbose == 1, output is displayed only
#         when the algorithm terminates. If verbose == 2, output is
#         displayed after every iteration.
#
#     recordTimes (required) - Whether the algorithm should store the total
#         processing time upon each iteration in a list to be obtained via
#         output.
#
#     recordResiduals (required) - Whether the algorithm should store the
#         relative residual values upon each iteration in a list to be
#         obtained via output.
#
#     recordMeasurementErrors (required) - Whether the algorithm should
#     store
#         the relative measurement errors upon each iteration in a list to
#         be obtained via output.
#
#     recordReconErrors (required) - Whether the algorithm should store the
#         relative reconstruction errors upon each iteration in a list to
#         be obtained via output. This parameter can only be set 'true'
#         when the additional parameter 'xt' is non-empty.
#
#     xt (required) - The true signal to be estimated, or an empty vector
#     if the
#         true estimate was not provided.
#
#     searchMethod (optional) - A string representing the method used to
#         determine search direction upon each iteration. Must be one of
#         {'steepestDescent', 'NCG', 'LBFGS'}. If equal to
#         'steepestDescent', then the steepest descent search method is
#         used. If equal to 'NCG', a nonlinear conjugate gradient method is
#         used. If equal to 'LBFGS', a Limited-Memory BFGS method is used.
#         Default value is 'steepestDescent'.
#
#     updateObjectivePeriod (optional) - The maximum number of iterations
#     that
#         are allowed to occur between updates to the objective function.
#         Default value is infinite (no limit is applied).
#
#     tolerancePenaltyLimit (optional) - The maximum tolerable penalty
#     caused by
#         surpassing the tolerance threshold before terminating. Default
#         value is 3.
#
#     betaChoice (optional) - A string representing the choice of the value
#         'beta' when a nonlinear conjugate gradient method is used. Must
#         be one of {'HS', 'FR', 'PR', 'DY'}. If equal to 'HS', the
#         Hestenes-Stiefel method is used. If equal to 'FR', the
#         Fletcher-Reeves method is used. If equal to 'PR', the
#         Polak-Ribi�re method is used. If equal to 'DY', the Dai-Yuan
#         method is used. This field is only used when searchMethod is set
#         to 'NCG'. Default value is 'HS'.
#
#     ncgResetPeriod (optional) - The maximum number of iterations that are
#         allowed to occur between resettings of a nonlinear conjugate
#         gradient search direction. This field is only used when
#         searchMethod is set to 'NCG'. Default value is 100.
#
#     storedVectors (optional) - The maximum number of previous iterations
#     of
#         which to retain LBFGS-specific iteration data. This field is only
#         used when searchMethod is set to 'LBFGS'. Default value is 5.
import sys
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/solvers')
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/initializers')
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/util')

import struct

import numpy as np
from numpy import dot
from numpy.linalg import norm
import time
from display_verbose_output import displayVerboseOutput


def gradientDescentSolver(A=None, At=None, x0=None, b0=None, updateObjective=None, opts=None, *args, **kwargs):
    setDefaultOpts()
    # Length of input signal
    n = len(x0)
    if not(opts.xt):
        residualTolerance = 1e-13
    else:
        residualTolerance = opts.tol

    # Create output
    # Iteration number of last objective update
    lastObjectiveUpdateIter = 0
    # Total penalty caused by surpassing tolerance threshold
    tolerancePenalty = 0
    # Whether to update objective function upon next iteration
    updateObjectiveNow = True
    # Maximum norm of differences between consecutive estimates
    maxDiff = - np.inf
    currentSolveTime = 0
    currentMeasurementError = []
    currentResidual = []
    currentReconError = []
    if opts.recordTimes:
        solveTimes = np.zeros(opts.maxIters, 1)

    if opts.recordResiduals:
        residuals = np.zeros(opts.maxIters, 1)

    if opts.recordMeasurementErrors:
        measurementErrors = np.zeros(opts.maxIters, 1)

    if opts.recordReconErrors:
        reconErrors = np.zeros(opts.maxIters, 1)

    x1 = x0
    d1 = A(x1)
    startTime = time.time
    for iter in range(1, opts.maxIters):
        # Signal to update objective function after fixed number of iterations
        # have passed
        if iter - lastObjectiveUpdateIter == opts.updateObjectivePeriod:
            updateObjectiveNow = True
        # Update objective if flag is set
        if updateObjectiveNow:
            updateObjectiveNow = False
            lastObjectiveUpdateIter = iter
            f, gradf = updateObjective(x1, d1, nargout=2)
            f1 = f(d1)
            gradf1 = At(gradf(d1))
            if strcmpi(opts.searchMethod, 'lbfgs'):
                # Perform LBFGS initialization
                yVals = np.zeros(n, opts.storedVectors)
                sVals = np.zeros(n, opts.storedVectors)
                rhoVals = np.zeros(1, opts.storedVectors)
            else:
                if strcmpi(opts.searchMethod, 'ncg'):
                    # Perform NCG initialization
                    lastNcgResetIter = iter
                    unscaledSearchDir = np.zeros(n, 1)
            searchDir1 = determineSearchDirection()
            tau1 = determineInitialStepsize()
        else:
            gradf1 = At(gradf(d1))
            Dg = gradf1 - gradf0
            if strcmpi(opts.searchMethod, 'lbfgs'):
                # Update LBFGS stored vectors
                sVals = concat(
                    [Dx, sVals(arange(), arange(1, opts.storedVectors - 1))])
                yVals = concat(
                    [Dg, yVals(arange(), arange(1, opts.storedVectors - 1))])
                rhoVals = concat(
                    [1 / real(dot(Dg.T, Dx)), rhoVals(arange(), arange(1, opts.storedVectors - 1))])
            searchDir1 = determineSearchDirection()
            updateStepsize()
        x0 = x1
        f0 = f1
        gradf0 = gradf1
        tau0 = tau1
        searchDir0 = searchDir1
        x1 = x0 + dot(tau0, searchDir0)
        Dx = x1 - x0
        d1 = A(x1)
        f1 = f(d1)
        # Armijo-Goldstein condition
        backtrackCount = 0
        while backtrackCount < 20:

            tmp = f0 + dot(dot(0.1, tau0), real(dot(searchDir0.T, gradf0)))
            # by error)
        # Avoids division by zero
            if f1 < tmp:
                break
            backtrackCount = backtrackCount + 1
            tau0 = dot(tau0, 0.2)
            x1 = x0 + dot(tau0, searchDir0)
            Dx = x1 - x0
            d1 = A(x1)
            f1 = f(d1)

        # Handle processing of current iteration estimate
        stopNow = processIteration()
        if stopNow:
            break

    # Create output

    sol = x1
    outs = struct
    outs.iterationCount = iter
    if opts.recordTimes:
        outs.solveTimes = solveTimes

    if opts.recordResiduals:
        outs.residuals = residuals

    if opts.recordMeasurementErrors:
        outs.measurementErrors = measurementErrors

    if opts.recordReconErrors:
        outs.reconErrors = reconErrors

    if opts.verbose == 1:
        displayVerboseOutput()


# Assigns default options for any options that were not provided by the
# client


    def setDefaultOpts(*args, **kwargs):
        if not(isfield(opts, 'updateObjectivePeriod')):
            # Objective function is never updated by default
            opts.updateObjectivePeriod = inf
        if not(isfield(opts, 'tolerancePenaltyLimit')):
            opts.tolerancePenaltyLimit = 3

        if not(isfield(opts, 'searchMethod')):
            opts.searchMethod = 'steepestDescent'

        if strcmpi(opts.searchMethod, 'lbfgs'):
            if logical_not(isfield(opts, 'storedVectors')):
                opts.storedVectors = 5

        if strcmpi(opts.searchMethod, 'ncg'):
            if logical_not(isfield(opts, 'betaChoice')):
                opts.betaChoice = 'HS'
            if logical_not(isfield(opts, 'ncgResetPeriod')):
                opts.ncgResetPeriod = 100

        return

    # Determine reasonable initial stepsize of current objective function
    # (adapted from FASTA.m)

    def determineInitialStepsize(*args, **kwargs):
        x_1 = np.random.randn(size(x0))
        x_2 = np.random.randn(size(x0))
        gradf_1 = At(gradf(A(x_1)))
        gradf_2 = At(gradf(A(x_2)))
        L = norm(gradf_1 - gradf_2) / norm(x_2 - x_1)
        L = max(L, 1e-30)
        tau = 25.0 / L
        return tau


# Determine search direction for next iteration based on specified search
# method


    def determineSearchDirection(*args, **kwargs):
        if 'steepestdescent' == opts.searchMethod.lower():
            searchDir = - gradf1
        else:
            if 'ncg' == opts.searchMethod.lower():
                searchDir = - gradf1
                # passed
                if iter - lastNcgResetIter == opts.ncgResetPeriod:
                    unscaledSearchDir = np.zeros(n, 1)
                    lastNcgResetIter = iter
                # Proceed only if reset has not just occurred
                if iter != lastNcgResetIter:
                    if 'hs' == opts.betaChoice.lower():
                        # Hestenes-Stiefel
                        beta = - real(dot(gradf1.T, Dg)) / \
                            real(dot(unscaledSearchDir.T, Dg))
                    else:
                        if 'fr' == opts.betaChoice.lower():
                            # Fletcher-Reeves
                            beta = norm(gradf1) ** 2 / norm(gradf0) ** 2
                        else:
                            if 'pr' == opts.betaChoice.lower():
                                # Polak-Ribi�re
                                beta = real(dot(gradf1.T, Dg)) / \
                                    norm(gradf0) ** 2
                            else:
                                if 'dy' == opts.betaChoice.lower():
                                    # Dai-Yuan
                                    beta = norm(gradf1) ** 2 / \
                                        real(dot(unscaledSearchDir.T, Dg))
                    searchDir = searchDir + dot(beta, unscaledSearchDir)
                unscaledSearchDir = searchDir
            else:
                if 'lbfgs' == opts.searchMethod.lower():
                    searchDir = - gradf1
                    iters = min(iter - lastObjectiveUpdateIter,
                                opts.storedVectors)
                    if iters > 0:
                        alphas = np.zeros(iters, 1)
                        for j in range(1, iters):
                            alphas[j] = dot(rhoVals(j), real(
                                dot(sVals(arange(), j).T, searchDir)))
                            searchDir = searchDir - \
                                dot(alphas(j), yVals(arange(), j))
                        # Scaling of search direction
                        gamma = real(dot(Dg.T, Dx)) / (dot(Dg.T, Dg))
                        searchDir = dot(gamma, searchDir)
                        for j in range(iters, 1, - 1):
                            beta = dot(rhoVals(j), real(
                                dot(yVals(arange(), j).T, searchDir)))
                            searchDir = searchDir + \
                                dot((alphas(j) - beta), sVals(arange(), j))
                        searchDir = dot(1 / gamma, searchDir)
                        searchDir = dot(
                            norm(gradf1) / norm(searchDir), searchDir)

        # Change search direction to steepest descent direction if current
            # direction is invalid
        if any(np.isnan(searchDir)) or any(np.isinf(searchDir)):
            searchDir = - gradf1

        # Scale current search direction match magnitude of gradient
        searchDir = dot(norm(gradf1) / norm(searchDir), searchDir)
        return searchDir


# Update stepsize when objective update has not just occurred (adopted from
# FASTA.m)


    def updateStepsize(*args, **kwargs):
        Ds = searchDir0 - searchDir1
        dotprod = real(dot(Dx, Ds))
        tauS = norm(Dx) ** 2 / dotprod

        tauM = dotprod / norm(Ds) ** 2

        tauM = max(tauM, 0)
        if dot(2, tauM) > tauS:
            tau1 = tauM
        else:
            tau1 = tauS - tauM / 2

        if tau1 < 0 or isinf(tau1) or isnan(tau1):
            tau1 = dot(tau0, 1.5)

        return

    def processIteration(*args, **kwargs):
        currentSolveTime = time.time
        maxDiff = max(norm(Dx), maxDiff)
        currentResidual = norm(Dx) / maxDiff

        if not(isempty(opts.xt)):
            reconEstimate = dot(
                (dot(x1.T, opts.xt)) / (dot(x1.T, x1)), x1)
            currentReconError = norm(opts.xt - reconEstimate) / norm(opts.xt)

        if opts.recordTimes:
            solveTimes[iter] = currentSolveTime

        if opts.recordResiduals:
            residuals[iter] = currentResidual

        if opts.recordMeasurementErrors:
            currentMeasurementError = norm(abs(d1) - b0) / norm(b0)
            measurementErrors[iter] = currentMeasurementError

        if opts.recordReconErrors:
            assert_(not(isempty(opts.xt)), concat(['You must specify the ground truth solution ',
                                                   'if the "recordReconErrors" flag is set to true.  Turn ', 'this flag off, or specify the ground truth solution.']))
            reconErrors[iter] = currentReconError

        if opts.verbose == 2:
            displayVerboseOutput()

        # Terminate if solver surpasses maximum allocated timespan
        if currentSolveTime > opts.maxTime:
            stopNow = True
            return stopNow

        # If user has supplied actual solution, use recon error to determine
            # termination
        if not(currentReconError):
            if currentReconError < opts.tol:
                stopNow = true
                return stopNow

        if currentResidual < residualTolerance:
            # Give algorithm chance to recover if stuck at local minimum by
                # forcing update of objective function
            updateObjectiveNow = True
            tolerancePenalty = tolerancePenalty + 1
            if tolerancePenalty > opts.tolerancePenaltyLimit:
                stopNow = True
                return stopNow

        stopNow = False
        return stopNow


# Display output to user based on provided options


    def displayVerboseOutput(*args, **kwargs):
        print('Iter = %d', iter)
        print(' | IterationTime = %.3f', currentSolveTime)
        print(' | Resid = %.3e', currentResidual)
        print(' | Stepsize = %.3e', tau0)
        if not(currentMeasurementError):
            print(' | M error = %.3e', currentMeasurementError)

        if not(currentReconError):
            print(' | R error = %.3e', currentReconError)

        print('\n')
        return
    return sol, outs
