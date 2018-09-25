import numpy as np
from numpy import dot
from numpy.linalg import norm
import time
from numpy.random import randn
import math
import struct

# fasta(A, At, f, gradf, g, proxg,sol, fastaOpts, nargout=2)


def fasta(A=None, At=None, f=None, gradf=None, g=None, proxg=None, x0=None, opts=None):
    # print('x0',x0)

    # Check whether we have function handles or matrices
    '''
    if not(A.isnumeric()):
        assert(not(At.isnumeric()),
               'If A is a function handle, then At must be a handle as well.')
    '''

    #  If we have matrices, create functions so we only have to treat one case
    # if A.isnumeric():
    #     At = lambda x=None: dot(A.T, x)
    #     A = lambda x=None: dot(A, x)

    # Check preconditions, fill missing optional entries in 'opts'
    if opts == None:
        opts = struct

    opts = setDefaults(opts, A, At, x0, gradf)
    # print('opts.tau',opts.tau)

    # Verify that At=A'
    checkAdjoint(A, At, x0)
    # print('opts.tau',opts.tau)
    if opts.verbose:
        print('%sFASTA:\tmode = %s\n\tmaxIters = %i,\ttol = %1.2d\n',
              opts.stringHeader, opts.mode, opts.maxIters, opts.tol)

    # Record some frequently used information from opts
    tau1 = opts.tau  # initial stepsize
    # print('tau1',tau1)
    # print('opts.tau',opts.tau)
    max_iters = opts.maxIters  # maximum iterations before automatic termination
    W = opts.window  # lookback window for non-montone line search

    # Allocate memory
    residual = np.zeros((max_iters, 1))
    normalizedResid = np.zeros((max_iters, 1))
    taus = np.zeros((max_iters, 1))
    fVals = np.zeros((max_iters, 1))
    objective = np.zeros((max_iters+1, 1))
    funcValues = np.zeros((max_iters, 1))
    totalBacktracks = 0
    backtrackCount = 0

    # Intialize array values
    x1 = x0
    d1 = dot(A, x1)

    f1 = f(d1)
    fVals[1] = f1
    gradf1 = dot(A.T, gradf(d1))

    if opts.accelerate:
        x_accel1 = x0
        d_accel1 = d1
        alpha1 = 1

    #  To handle non-monotonicity
    maxResidual = - np.Inf

    minObjectiveValue = np.Inf

    #  If user has chosen to record objective, then record initial value
    if opts.recordObjective:
        objective[1] = f1 + g(x0)

    # # Begin recording solve time
    # start_time = time.time

    # Begin Loop
    for i in range(max_iters):
        # Rename iterates relative to loop index.  "0" denotes index i, and "1" denotes index i+1
        x0 = x1
        gradf0 = gradf1
        tau0 = tau1
        # FBS step: obtain x_{i+1} from x_i
        x1hat = x0 - tau0*gradf0
        x1 = proxg(x1hat, tau0)
        # Non-monotone backtracking line search
        Dx = x1 - x0
        d1 = dot(A, x1)

        f1 = f(d1)
        if opts.backtrack:
            # Get largest of last 10 values of 'f'
            M = np.sort(fVals[-10])
            backtrackCount = 0
            while (f1 - 1e-12) > (M + np.real(dot(Dx.flatten('F'), gradf0.flatten('F'))) + norm(Dx.flatten('F')) ** 2) / (dot(2, tau0)) and backtrackCount < 20 or not(np.isreal(f1)):
                tau0 = dot(tau0, opts.stepsizeShrink)
                x1hat = x0 - dot(tau0, gradf0)
                x1 = proxg(x1hat, tau0)
                d1 = dot(A, x1)
                f1 = f(d1)
                Dx = x1 - x0
                backtrackCount = backtrackCount + 1

            totalBacktracks = totalBacktracks + backtrackCount
        if opts.verbose and backtrackCount > 10:
            print('%s\tWARNING: excessive backtracking (%d steps), current stepsize is %0.2d\n',
                  opts.stringHeader, backtrackCount, tau0)

        # Record convergence information
        taus[i] = tau0

        residual[i] = norm((np.array(Dx)).reshape(-1))/tau0

        maxResidual = max(maxResidual, residual[i])
        normalizer = max(norm(gradf0.flatten('F')), norm(
            x1.flatten('F') - x1hat.flatten('F')) / tau0) + opts.eps_n
        normalizedResid[i] = residual[i] / normalizer
        fVals[i] = f1
        funcValues[i] = opts.function(x1)
        if opts.recordObjective:
            objective[i + 1] = f1 + g(x1)
            newObjectiveValue = objective(i + 1)
        else:
            newObjectiveValue = residual[i]
        if opts.recordIterates:
            iterates[i] = x1
        if newObjectiveValue < minObjectiveValue:
            bestObjectiveIterate = x1
            bestObjectiveIterateHat = x1hat
            minObjectiveValue = newObjectiveValue
        if opts.verbose > 1:
            print('%s%d: resid = %0.2d, backtrack = %d, tau = %d',
                  opts.stringHeader, i, residual(i), backtrackCount, tau0)
            if opts.recordObjective:
                print(', objective = %d\n', objective(i + 1))
            else:
                print('\n')
        # Test stopping criteria
        #  If we stop, then record information in the output struct
        if opts.stopNow(x1, i, residual[i], normalizedResid[i], maxResidual, opts) or (i > max_iters):

            outs = struct
            outs.solveTime = time.time
            outs.residuals = residual[1:i]
            outs.stepsizes = taus[1:i]
            outs.normalizedResiduals = normalizedResid[1:i]
            outs.objective = objective[1:i]
            outs.funcValues = funcValues[1:i]
            outs.backtracks = totalBacktracks
            outs.L = opts.L
            outs.initialStepsize = opts.tau
            outs.iterationCount = i
            if not(opts.recordObjective):
                outs.objective = 'Not Recorded'
            if opts.recordIterates:
                outs.iterates = iterates
            outs.bestObjectiveIterateHat = bestObjectiveIterateHat
            sol = bestObjectiveIterate

            if opts.verbose:
                print('%s\tDone:  time = %0.3f secs, iterations = %i\n',
                      opts.stringHeader, time.time, outs.iterationCount)
            return sol, outs, opts

        if opts.adaptive and not(opts.accelerate):
            # Compute stepsize needed for next iteration using BB/spectral method
            gradf1 = At(gradf(d1))
            Dg = gradf1 + (x1hat - x0) / tau0
            dotprod = dot(Dx.flatten('F'), Dg.flatten('F')).real()
            # First BB stepsize rule
            tau_s = norm(Dx.flatten('F')) ** 2 / dotprod
            # Alternate BB stepsize rule
            tau_m = dotprod / norm(Dg.flatten('F')) ** 2
            tau_m = max(tau_m, 0)
            if dot(2, tau_m) > tau_s:
                tau1 = tau_m
            else:
                tau1 = tau_s - dot(0.5, tau_m)
            if tau1 < 0 or np.isinf(tau1) or np.isnan(tau1):
                tau1 = dot(tau0, 1.5)
        if opts.accelerate:
            # Use FISTA-type acceleration
            x_accel0 = x_accel1
            d_accel0 = d_accel1
            alpha0 = alpha1
            x_accel1 = x1
            d_accel1 = d1
            #  Check to see if the acceleration needs to be restarted
            if opts.restart and dot((x0.flatten('F') - x1.flatten('F')).T, (x1) - x_accel0.flatten('F')) > 0:
                alpha0 = 1
            #  Calculate acceleration parameter
            alpha1 = (1 + math.sqrt(1 + dot(4, alpha0 ** 2))) / 2
            x1 = x_accel1 + dot((alpha0 - 1) / alpha1, (x_accel1 - x_accel0))
            d1 = d_accel1 + dot((alpha0 - 1) / alpha1, (d_accel1 - d_accel0))
            gradf1 = At(gradf(d1))
            fVals[i] = f(d1)
            tau1 = tau0
        if not(opts.adaptive) and not(opts.accelerate):
            gradf1 = At(gradf(d1))
            tau1 = tau0
    return sol, outs, opts


def checkAdjoint(A=None, At=None, x=None, *args, **kwargs):
    x = randn(np.shape(x)[0])
    Ax = dot(A, x)
    y = randn(np.shape(Ax)[0])
    Aty = dot(A.T, y)

    innerProduct1 = dot(Ax.flatten('F').T, y.flatten('F'))
    innerProduct2 = dot(x.flatten('F').T, Aty.flatten('F'))
    error = abs(innerProduct1 - innerProduct2) / \
        max(abs(innerProduct1), abs(innerProduct2))
    assert(error < 0.001, '"At" is not the adjoint of "A".  Check the definitions of these operators. Error=' + str(error))

    return


# Fill in the struct of options with the default values
def setDefaults(opts=None, A=None, At=None, x0=None, gradf=None, *args, **kwargs):
    #  maxIters: The maximum number of iterations
    if not(hasattr(opts, 'maxIters')):
        setattr(opts, "maxIters", 1000)

    # tol:  The relative decrease in the residuals before the method stops
    if not(hasattr(opts, 'tol')):
        setattr(opts, "tol", 0.001)

    # verbose:  If 'True' then print status information on every iteration
    if not(hasattr(opts, 'verbose')):
        setattr(opts, "verbose", False)

    # recordObjective:  If 'True' then evaluate objective at every iteration
    if not(hasattr(opts, 'recordObjective')):
        setattr(opts, "recordObjective", False)

    # recordIterates:  If 'True' then record iterates in cell array
    if not(hasattr(opts, 'recordIterates')):
        setattr(opts, "recordIterates", False)

    # adaptive:  If 'True' then use adaptive method.
    if not(hasattr(opts, 'adaptive')):
        opts.adaptive = True

    # accelerate:  If 'True' then use FISTA-type adaptive method.
    if not(hasattr(opts, 'accelerate')):
        setattr(opts, "accelerate", False)

    # restart:  If 'True' then restart the acceleration of FISTA.
    #   This only has an effect when opts.accelerate=True
    if not(hasattr(opts, 'restart')):
        setattr(opts, "restart", True)

    # backtrack:  If 'True' then use backtracking line search
    if not(hasattr(opts, 'backtrack')):
        setattr(opts, "backtrack", True)

    # stepsizeShrink:  Coefficient used to shrink stepsize when backtracking
    # kicks in
    if not(hasattr(opts, 'stepsizeShrink')):
        setattr(opts, "stepsizeShrink", 0.2)
        if not(opts.adaptive) or opts.accelerate:
            setattr(opts, "stepsizeShrink", 0.5)

    #  Create a mode string that describes which variant of the method is used
    opts.mode = 'plain'
    if opts.adaptive:
        opts.mode = 'adaptive'

    if opts.accelerate:
        if opts.restart:
            opts.mode = 'accelerated(FISTA)+restart'
        else:
            opts.mode = 'accelerated(FISTA)'

    # W:  The window to look back when evaluating the max for the line search
    if not(hasattr(opts, 'window')):
        setattr(opts, "window", 10)

    # eps_r:  Epsilon to prevent ratio residual from dividing by zero
    if not(hasattr(opts, 'eps_r')):
        setattr(opts, "eps_r", 1e-08)

    # eps_n:  Epsilon to prevent normalized residual from dividing by zero
    if not(hasattr(opts, 'eps_n')):
        setattr(opts, "eps_n", 1e-08)

    #  L:  Lipschitz constant for smooth term.  Only needed if tau has not been
    #   set, in which case we need to approximate L so that tau can be
    #   computed.
    if (not(hasattr(opts, 'L')) or opts.L < 0) and (not(hasattr(opts, 'tau')) or opts.tau < 0):
        x1 = randn(np.shape(x0)[0])
        x2 = randn(np.shape(x0)[0])

        gradf1 = dot(A.T, gradf(dot(A, x1)))
        gradf2 = dot(A.T, gradf(dot(A, x2)))
        opts.L = norm(gradf1.flatten('F') - gradf2.flatten('F')) / \
            norm(x2.flatten('F') - x1.flatten('F'))

        opts.L = max(opts.L, 1e-06)
        opts.tau = 2 / opts.L / 10

    assert(opts.tau > 0, 'Invalid step size: '+str(opts.tau))
    #  Set tau if L was set by user
    if (not(hasattr(opts, 'tau')) or opts.tau < 0):
        opts.tau = 1.0 / opts.L
    else:
        opts.L = 1 / opts.tau

    # function:  An optional function that is computed and stored after every
    # iteration
    if not(hasattr(opts, 'function')):
        setattr(opts, "function", lambda x=None: 0)

    # stringHeader:  Append this string to beginning of all output
    if not(hasattr(opts, 'stringHeader')):
        setattr(opts, "stringHeader", '')

    #  The code below is for stopping rules
    #  The field 'stopNow' is a function that returns 'True' if the iteration
    #  should be terminated.  The field 'stopRule' is a string that allows the
    #  user to easily choose default values for 'stopNow'.  The default
    #  stopping rule terminates when the relative residual gets small.
    if hasattr(opts, 'stopNow'):
        setattr(opts, "stopNow", 'custom')

    if not(hasattr(opts, 'stopRule')):
        setattr(opts, "stopRule", 'hybridResidual')

    if opts.stopRule == 'residual':
        opts.stopNow = lambda x1=None, iter=None, resid=None, normResid=None, maxResidual=None, opts=None: resid < opts.tol

    if opts.stopRule == 'iterations':
        opts.stopNow = lambda x1=None, iter=None, resid=None, normResid=None, maxResidual=None, opts=None: iter > opts.maxIters

    # Stop when normalized residual is small
    if opts.stopRule == 'normalizedResidual':
        opts.stopNow = lambda x1=None, iter=None, resid=None, normResid=None, maxResidual=None, opts=None: normResid < opts.tol

    # Divide by residual at iteration k by maximum residual over all iterations.
# Terminate when this ratio gets small.
    if opts.stopRule == 'ratioResidual':
        opts.stopNow = lambda x1=None, iter=None, resid=None, normResid=None, maxResidual=None, opts=None: resid / \
            (maxResidual + opts.eps_r) < opts.tol

    # Default behavior:  Stop if EITHER normalized or ration residual is small
    if opts.stopRule == 'hybridResidual':
        opts.stopNow = lambda x1=None, iter=None, resid=None, normResid=None, maxResidual=None, opts=None: resid / \
            (maxResidual + opts.eps_r) < opts.tol or normResid < opts.tol

    assert(hasattr(opts, 'stopNow'),
           'Invalid choice for stopping rule: ' + opts.stopRule)
    return opts
