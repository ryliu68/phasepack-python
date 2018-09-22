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
'''
function [sol, outs] = solvePhaseMax(A,At,b0,x0,opts)
    # Initialization
    m = length(b0)              # number of measurements
    n = length(x0)              # length of the unknown signal
    remainIters = opts.maxIters # The remaining fasta iterations we have.
                                 # It's initialized to opts.maxIters.

    #  Normalize the initial guess relative to the number of measurements
    x0 = (x0/norm(x0(:)))*mean(b0(:))*(m/n)*100

    #  re-scale the initial guess so that it approximately satisfies |Ax|=b
    sol = x0.* min(b0./abs(A(x0)))

    # Initialize values potentially computed at each round.
    # Indicate whether any of the ending condition (except maxIters) has been met in FASTA.
    ending = 0
    iter = 0
    currentTime = []
    currentResid = []
    currentReconError = []
    currentMeasurementError = []

    # Initialize vectors for recording convergence information
    [solveTimes,measurementErrors,reconErrors,
        residuals] = initializeContainers(opts)

    # Define objective function components for the gradient descent method FASTA
    # f(z) = 0.5*max{|x|-b,0}^2 : This is the quadratic penalty function
    f = @(z) 0.5*norm(max(abs(z)-b0,0))^2
    # The gradient of the quadratic penalty
    gradf = @(z)  (sign(z).*max(abs(z)-b0,0))

    # Options to hand to fasta
    fastaOpts.maxIters = opts.maxIters
    fastaOpts.stopNow = @(x, iter, resid, normResid, maxResid, opts) ...
        processIteration(x, resid)  # Use customized stopNow in order to get
                                         # solveTime, residual and error at each iteration.
    fastaOpts.verbose=0
    startTime = tic                    # Start timer
    # Keep track of the current error in the solution.
    constraintError = norm(abs(A(sol))-b0)
    while remainIters > 0 & ~ending     # Iterate over continuation steps
        g = @(x) -real(x0'*x)          # The linear part of the objective
        # The proximal operator of the linear objective
        proxg = @(x,t) x+t*x0
        # use a tighter tolerance when the solution is more exact
        fastaOpts.tol = norm(x0)/100
        # Call FASTA to solve the inner minimization problem
        # Call a solver to minimize the quadratic barrier problem
        [sol, fastaOuts] = fasta(A, At, f, gradf, g, proxg, sol, fastaOpts)

        # Record the most recent stepsize for recycling.
        fastaOpts.tau = fastaOuts.stepsizes(end)
        # do continuation - this makes the quadratic penalty stronger
        x0 = x0/10

        # Update the max number of iterations for fasta
        remainIters = remainIters - fastaOuts.iterationCount
        fastaOpts.maxIters = min(opts.maxIters, remainIters)

        # Monitor convergence and check stopping conditions
        newConstraintError = norm(max(abs(A(sol))-b0,0))
        relativeChange = abs(constraintError-newConstraintError)/norm(b0)
        if relativeChange <opts.tol   # terminate when error reduction stalls
            break
        end
        constraintError = newConstraintError
    end


    # Create output according to the options chosen by user
    outs = generateOutputs(opts, iter, solveTimes,
                           measurementErrors, reconErrors, residuals)

    # Display verbose output if specified
    if opts.verbose == 1
        displayVerboseOutput(iter, currentTime, currentResid,
                             currentReconError, currentMeasurementError)
    end


    # Runs code upon each FASTA iteration. Returns whether FASTA should
    # terminate.
    function stop = processIteration(x, residual)
        iter = iter + 1
        # Record convergence information and check stopping condition
        # If xt is provided, reconstruction error will be computed and used for stopping
        # condition. Otherwise, residual will be computed and used for stopping
        # condition.
        if ~isempty(opts.xt)
            xt = opts.xt
            #  Compute optimal rotation
            alpha = (x(:)'*xt(:))/(x(:)'*x(:))
            x = alpha*x
            currentReconError = norm(x-xt)/norm(xt)
            if opts.recordReconErrors
                reconErrors(iter) = currentReconError
            end
        end

        if isempty(opts.xt)
            currentResid = residual
        end

        if opts.recordResiduals
            residuals(iter) = residual
        end

        # Record elapsed time so far
        currentTime = toc(startTime)
        if opts.recordTimes
            solveTimes(iter) = currentTime
        end
        if opts.recordMeasurementErrors
            currentMeasurementError = norm(abs(A(sol)) - b0) / norm(b0)
            measurementErrors(iter) = currentMeasurementError
        end

        # Display verbose output if specified
        if opts.verbose == 2
            displayVerboseOutput(
                iter, currentTime, currentResid, currentReconError, currentMeasurementError)
        end

        # Test stopping criteria.
        stop = false
        if currentTime >= opts.maxTime # Stop if we're run over the max runtime
            stop = true
        end
        # If true solution is specified, terminate when close to true solution
        if ~isempty(opts.xt)
            assert(~isempty(currentReconError),
                   'If xt is provided, currentReconError must be provided.')
            stop = stop || currentReconError < opts.tol
            ending = stop  # When true, this flag will terminate outer loop
        end
        # Stop FASTA is the tolerance is reached
        stop = stop || residual < fastaOpts.tol
    end

end
'''
import numpy as np


def solvePhaseMax(A=None, At=None, b0=None, x0=None, opts=None, *args, **kwargs):
    # Initialization
    m = len(b0)
    n = len(x0)
    remainIters = opts.maxIters

    # It's initialized to opts.maxIters.
    #  Normalize the initial guess relative to the number of measurements
    x0 = np.dot(
        np.dot(np.dot((x0 / norm(ravel(x0))), mean(ravel(b0))), (m / n)), 100)
    sol = np.multiply(x0, min(b0 / abs(A(x0))))
    ending = 0
    iter = 0
    currentTime = []
    currentResid = []
    currentReconError = []
    currentMeasurementError = []

    solveTimes, measurementErrors, reconErrors, residuals = initializeContainers(
        opts, nargout=4)

    f = lambda z=None: np.dot(0.5, norm(max(abs(z) - b0, 0)) ** 2)

    gradf = lambda z=None: (np.multiply(np.sign(z), max(abs(z) - b0, 0)))

    # Options to hand to fasta
    fastaOpts.maxIters = opts.maxIters
    # fastaOpts.stopNow = lambda x=None, iter=None, resid=None,
    #                          normResid = None, maxResid = None, opts = None: processIteration(x, resid)

    # solveTime, residual and error at each iteration.
    fastaOpts.verbose = 0
    startTime = tic

    constraintError = norm(abs(A(sol)) - b0)

    while remainIters > logical_and(0, not(ending)):

        g = lambda x=None: - real(np.dot(x0.T, x))
        proxg = lambda x=None, t=None: x + np.dot(t, x0)
        fastaOpts.tol = norm(x0) / 100
        # Call FASTA to solve the inner minimization problem
        sol, fastaOuts = fasta(A, At, f, gradf, g, proxg,
                               sol, fastaOpts, nargout=2)
        fastaOpts.tau = fastaOuts.stepsizes(end())
        x0 = x0 / 10
        # Update the max number of iterations for fasta
        remainIters = remainIters - fastaOuts.iterationCount
        fastaOpts.maxIters = min(opts.maxIters, remainIters)
        newConstraintError = norm(max(abs(A(sol)) - b0, 0))
        relativeChange = abs(constraintError - newConstraintError) / norm(b0)
        if relativeChange < opts.tol:
            break
        constraintError = newConstraintError

    # Create output according to the options chosen by user
    outs = generateOutputs(opts, iter, solveTimes,
                           measurementErrors, reconErrors, residuals)

    if opts.verbose == 1:
        displayVerboseOutput(iter, currentTime, currentResid,
                             currentReconError, currentMeasurementError)

    # Runs code upon each FASTA iteration. Returns whether FASTA should
    # terminate.

    def processIteration(x=None, residual=None, *args, **kwargs):
        iter = iter + 1

        # If xt is provided, reconstruction error will be computed and used for stopping
        # condition. Otherwise, residual will be computed and used for stopping
        # condition.
        if not(isempty(opts.xt)):
            xt = opts.xt
            alpha = (np.dot(ravel(x).T, ravel(xt))) / \
                (np.dot(ravel(x).T, ravel(x)))
            x = np.dot(alpha, x)
            currentReconError = norm(x - xt) / norm(xt)
            if opts.recordReconErrors:
                reconErrors[iter] = currentReconError

        if opts.xt:
            currentResid = residual

        if opts.recordResiduals:
            residuals[iter] = residual

        currentTime = toc(startTime)

        if opts.recordTimes:
            solveTimes[iter] = currentTime

        if opts.recordMeasurementErrors:
            currentMeasurementError = norm(abs(A(sol)) - b0) / norm(b0)
            measurementErrors[iter] = currentMeasurementError

        # Display verbose output if specified
        if opts.verbose == 2:
            displayVerboseOutput(
                iter, currentTime, currentResid, currentReconError, currentMeasurementError)

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
