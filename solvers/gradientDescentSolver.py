# Generated with SMOP  0.41
from libsmop import *
# gradientDescentSolver.m

    ## -------------------------gradientDescentSolver.m--------------------------------
    
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
    
    ## Aditional Parameters
# The following are additional parameters that are to be passed as fields
# of the struct 'opts':
    
    #     maxIters (required) - The maximum number of iterations that are
#     allowed to
#         occur.
    
    #     maxTime (required) - The maximum amount of time in seconds the
#     algorithm
#         is allowed to spend solving before terminating.
    
    #     tol (required) - Positive real number representing how precise the
#     final
#         estimate should be. Lower values indicate to the solver that a
#         more precise estimate should be obtained.
    
    #     verbose (required) - Integer representing whether / how verbose
#         information should be displayed via output. If verbose == 0, no
#         output is displayed. If verbose == 1, output is displayed only
#         when the algorithm terminates. If verbose == 2, output is
#         displayed after every iteration.
    
    #     recordTimes (required) - Whether the algorithm should store the total
#         processing time upon each iteration in a list to be obtained via
#         output.
    
    #     recordResiduals (required) - Whether the algorithm should store the
#         relative residual values upon each iteration in a list to be
#         obtained via output.
    
    #     recordMeasurementErrors (required) - Whether the algorithm should
#     store
#         the relative measurement errors upon each iteration in a list to
#         be obtained via output.
    
    #     recordReconErrors (required) - Whether the algorithm should store the
#         relative reconstruction errors upon each iteration in a list to
#         be obtained via output. This parameter can only be set 'true'
#         when the additional parameter 'xt' is non-empty.
    
    #     xt (required) - The true signal to be estimated, or an empty vector
#     if the
#         true estimate was not provided.
    
    #     searchMethod (optional) - A string representing the method used to
#         determine search direction upon each iteration. Must be one of
#         {'steepestDescent', 'NCG', 'LBFGS'}. If equal to
#         'steepestDescent', then the steepest descent search method is
#         used. If equal to 'NCG', a nonlinear conjugate gradient method is
#         used. If equal to 'LBFGS', a Limited-Memory BFGS method is used.
#         Default value is 'steepestDescent'.
    
    #     updateObjectivePeriod (optional) - The maximum number of iterations
#     that
#         are allowed to occur between updates to the objective function.
#         Default value is infinite (no limit is applied).
    
    #     tolerancePenaltyLimit (optional) - The maximum tolerable penalty
#     caused by
#         surpassing the tolerance threshold before terminating. Default
#         value is 3.
    
    #     betaChoice (optional) - A string representing the choice of the value
#         'beta' when a nonlinear conjugate gradient method is used. Must
#         be one of {'HS', 'FR', 'PR', 'DY'}. If equal to 'HS', the
#         Hestenes-Stiefel method is used. If equal to 'FR', the
#         Fletcher-Reeves method is used. If equal to 'PR', the
#         Polak-Ribi�re method is used. If equal to 'DY', the Dai-Yuan
#         method is used. This field is only used when searchMethod is set
#         to 'NCG'. Default value is 'HS'.
    
    #     ncgResetPeriod (optional) - The maximum number of iterations that are
#         allowed to occur between resettings of a nonlinear conjugate
#         gradient search direction. This field is only used when
#         searchMethod is set to 'NCG'. Default value is 100.
    
    #     storedVectors (optional) - The maximum number of previous iterations
#     of
#         which to retain LBFGS-specific iteration data. This field is only
#         used when searchMethod is set to 'LBFGS'. Default value is 5.
    
    
@function
def gradientDescentSolver(A=None,At=None,x0=None,b0=None,updateObjective=None,opts=None,*args,**kwargs):
    varargin = gradientDescentSolver.varargin
    nargin = gradientDescentSolver.nargin

    setDefaultOpts()
    # Length of input signal
    n=length(x0)
# gradientDescentSolver.m:104
    if logical_not(isempty(opts.xt)):
        residualTolerance=1e-13
# gradientDescentSolver.m:106
    else:
        residualTolerance=opts.tol
# gradientDescentSolver.m:108
    
    # Iteration number of last objective update
    lastObjectiveUpdateIter=0
# gradientDescentSolver.m:112
    # Total penalty caused by surpassing tolerance threshold
    tolerancePenalty=0
# gradientDescentSolver.m:114
    # Whether to update objective function upon next iteration
    updateObjectiveNow=copy(true)
# gradientDescentSolver.m:116
    # Maximum norm of differences between consecutive estimates
    maxDiff=- inf
# gradientDescentSolver.m:118
    currentSolveTime=0
# gradientDescentSolver.m:120
    currentMeasurementError=[]
# gradientDescentSolver.m:121
    currentResidual=[]
# gradientDescentSolver.m:122
    currentReconError=[]
# gradientDescentSolver.m:123
    if opts.recordTimes:
        solveTimes=zeros(opts.maxIters,1)
# gradientDescentSolver.m:126
    
    if opts.recordResiduals:
        residuals=zeros(opts.maxIters,1)
# gradientDescentSolver.m:129
    
    if opts.recordMeasurementErrors:
        measurementErrors=zeros(opts.maxIters,1)
# gradientDescentSolver.m:132
    
    if opts.recordReconErrors:
        reconErrors=zeros(opts.maxIters,1)
# gradientDescentSolver.m:135
    
    x1=copy(x0)
# gradientDescentSolver.m:138
    d1=A(x1)
# gradientDescentSolver.m:139
    startTime=copy(tic)
# gradientDescentSolver.m:141
    for iter in arange(1,opts.maxIters).reshape(-1):
        # Signal to update objective function after fixed number of iterations
    # have passed
        if iter - lastObjectiveUpdateIter == opts.updateObjectivePeriod:
            updateObjectiveNow=copy(true)
# gradientDescentSolver.m:146
        # Update objective if flag is set
        if updateObjectiveNow:
            updateObjectiveNow=copy(false)
# gradientDescentSolver.m:150
            lastObjectiveUpdateIter=copy(iter)
# gradientDescentSolver.m:151
            f,gradf=updateObjective(x1,d1,nargout=2)
# gradientDescentSolver.m:152
            f1=f(d1)
# gradientDescentSolver.m:153
            gradf1=At(gradf(d1))
# gradientDescentSolver.m:154
            if strcmpi(opts.searchMethod,'lbfgs'):
                # Perform LBFGS initialization
                yVals=zeros(n,opts.storedVectors)
# gradientDescentSolver.m:158
                sVals=zeros(n,opts.storedVectors)
# gradientDescentSolver.m:159
                rhoVals=zeros(1,opts.storedVectors)
# gradientDescentSolver.m:160
            else:
                if strcmpi(opts.searchMethod,'ncg'):
                    # Perform NCG initialization
                    lastNcgResetIter=copy(iter)
# gradientDescentSolver.m:163
                    unscaledSearchDir=zeros(n,1)
# gradientDescentSolver.m:164
            searchDir1=determineSearchDirection()
# gradientDescentSolver.m:166
            tau1=determineInitialStepsize()
# gradientDescentSolver.m:168
        else:
            gradf1=At(gradf(d1))
# gradientDescentSolver.m:170
            Dg=gradf1 - gradf0
# gradientDescentSolver.m:171
            if strcmpi(opts.searchMethod,'lbfgs'):
                # Update LBFGS stored vectors
                sVals=concat([Dx,sVals(arange(),arange(1,opts.storedVectors - 1))])
# gradientDescentSolver.m:175
                yVals=concat([Dg,yVals(arange(),arange(1,opts.storedVectors - 1))])
# gradientDescentSolver.m:176
                rhoVals=concat([1 / real(dot(Dg.T,Dx)),rhoVals(arange(),arange(1,opts.storedVectors - 1))])
# gradientDescentSolver.m:177
            searchDir1=determineSearchDirection()
# gradientDescentSolver.m:180
            updateStepsize()
        x0=copy(x1)
# gradientDescentSolver.m:184
        f0=copy(f1)
# gradientDescentSolver.m:185
        gradf0=copy(gradf1)
# gradientDescentSolver.m:186
        tau0=copy(tau1)
# gradientDescentSolver.m:187
        searchDir0=copy(searchDir1)
# gradientDescentSolver.m:188
        x1=x0 + dot(tau0,searchDir0)
# gradientDescentSolver.m:190
        Dx=x1 - x0
# gradientDescentSolver.m:191
        d1=A(x1)
# gradientDescentSolver.m:192
        f1=f(d1)
# gradientDescentSolver.m:193
        # Armijo-Goldstein condition
        backtrackCount=0
# gradientDescentSolver.m:198
        while backtrackCount < 20:

            tmp=f0 + dot(dot(0.1,tau0),real(dot(searchDir0.T,gradf0)))
# gradientDescentSolver.m:201
            # by error)
        # Avoids division by zero
            if f1 < tmp:
                break
            backtrackCount=backtrackCount + 1
# gradientDescentSolver.m:210
            tau0=dot(tau0,0.2)
# gradientDescentSolver.m:212
            x1=x0 + dot(tau0,searchDir0)
# gradientDescentSolver.m:213
            Dx=x1 - x0
# gradientDescentSolver.m:214
            d1=A(x1)
# gradientDescentSolver.m:215
            f1=f(d1)
# gradientDescentSolver.m:216

        # Handle processing of current iteration estimate
        stopNow=processIteration()
# gradientDescentSolver.m:220
        if stopNow:
            break
    
    # Create output
    sol=copy(x1)
# gradientDescentSolver.m:227
    outs=copy(struct)
# gradientDescentSolver.m:228
    outs.iterationCount = copy(iter)
# gradientDescentSolver.m:229
    if opts.recordTimes:
        outs.solveTimes = copy(solveTimes)
# gradientDescentSolver.m:231
    
    if opts.recordResiduals:
        outs.residuals = copy(residuals)
# gradientDescentSolver.m:234
    
    if opts.recordMeasurementErrors:
        outs.measurementErrors = copy(measurementErrors)
# gradientDescentSolver.m:237
    
    if opts.recordReconErrors:
        outs.reconErrors = copy(reconErrors)
# gradientDescentSolver.m:240
    
    if opts.verbose == 1:
        displayVerboseOutput()
    
    # Assigns default options for any options that were not provided by the
# client
    
@function
def setDefaultOpts(*args,**kwargs):
    varargin = setDefaultOpts.varargin
    nargin = setDefaultOpts.nargin

    if logical_not(isfield(opts,'updateObjectivePeriod')):
        # Objective function is never updated by default
        opts.updateObjectivePeriod = copy(inf)
# gradientDescentSolver.m:252
    
    if logical_not(isfield(opts,'tolerancePenaltyLimit')):
        opts.tolerancePenaltyLimit = copy(3)
# gradientDescentSolver.m:255
    
    if logical_not(isfield(opts,'searchMethod')):
        opts.searchMethod = copy('steepestDescent')
# gradientDescentSolver.m:258
    
    if strcmpi(opts.searchMethod,'lbfgs'):
        if logical_not(isfield(opts,'storedVectors')):
            opts.storedVectors = copy(5)
# gradientDescentSolver.m:262
    
    if strcmpi(opts.searchMethod,'ncg'):
        if logical_not(isfield(opts,'betaChoice')):
            opts.betaChoice = copy('HS')
# gradientDescentSolver.m:267
        if logical_not(isfield(opts,'ncgResetPeriod')):
            opts.ncgResetPeriod = copy(100)
# gradientDescentSolver.m:270
    
    return
    
if __name__ == '__main__':
    pass
    
    # Determine reasonable initial stepsize of current objective function
# (adapted from FASTA.m)
    
@function
def determineInitialStepsize(*args,**kwargs):
    varargin = determineInitialStepsize.varargin
    nargin = determineInitialStepsize.nargin

    x_1=randn(size(x0))
# gradientDescentSolver.m:278
    x_2=randn(size(x0))
# gradientDescentSolver.m:279
    gradf_1=At(gradf(A(x_1)))
# gradientDescentSolver.m:280
    gradf_2=At(gradf(A(x_2)))
# gradientDescentSolver.m:281
    L=norm(gradf_1 - gradf_2) / norm(x_2 - x_1)
# gradientDescentSolver.m:282
    L=max(L,1e-30)
# gradientDescentSolver.m:283
    tau=25.0 / L
# gradientDescentSolver.m:284
    return tau
    
if __name__ == '__main__':
    pass
    
    # Determine search direction for next iteration based on specified search
# method
    
@function
def determineSearchDirection(*args,**kwargs):
    varargin = determineSearchDirection.varargin
    nargin = determineSearchDirection.nargin

    if 'steepestdescent' == lower(opts.searchMethod):
        searchDir=- gradf1
# gradientDescentSolver.m:292
    else:
        if 'ncg' == lower(opts.searchMethod):
            searchDir=- gradf1
# gradientDescentSolver.m:294
            # passed
            if iter - lastNcgResetIter == opts.ncgResetPeriod:
                unscaledSearchDir=zeros(n,1)
# gradientDescentSolver.m:299
                lastNcgResetIter=copy(iter)
# gradientDescentSolver.m:300
            # Proceed only if reset has not just occurred
            if iter != lastNcgResetIter:
                if 'hs' == lower(opts.betaChoice):
                    # Hestenes-Stiefel
                    beta=- real(dot(gradf1.T,Dg)) / real(dot(unscaledSearchDir.T,Dg))
# gradientDescentSolver.m:308
                else:
                    if 'fr' == lower(opts.betaChoice):
                        # Fletcher-Reeves
                        beta=norm(gradf1) ** 2 / norm(gradf0) ** 2
# gradientDescentSolver.m:311
                    else:
                        if 'pr' == lower(opts.betaChoice):
                            # Polak-Ribi�re
                            beta=real(dot(gradf1.T,Dg)) / norm(gradf0) ** 2
# gradientDescentSolver.m:314
                        else:
                            if 'dy' == lower(opts.betaChoice):
                                # Dai-Yuan
                                beta=norm(gradf1) ** 2 / real(dot(unscaledSearchDir.T,Dg))
# gradientDescentSolver.m:317
                searchDir=searchDir + dot(beta,unscaledSearchDir)
# gradientDescentSolver.m:319
            unscaledSearchDir=copy(searchDir)
# gradientDescentSolver.m:321
        else:
            if 'lbfgs' == lower(opts.searchMethod):
                searchDir=- gradf1
# gradientDescentSolver.m:323
                iters=min(iter - lastObjectiveUpdateIter,opts.storedVectors)
# gradientDescentSolver.m:324
                if iters > 0:
                    alphas=zeros(iters,1)
# gradientDescentSolver.m:327
                    for j in arange(1,iters).reshape(-1):
                        alphas[j]=dot(rhoVals(j),real(dot(sVals(arange(),j).T,searchDir)))
# gradientDescentSolver.m:331
                        searchDir=searchDir - dot(alphas(j),yVals(arange(),j))
# gradientDescentSolver.m:332
                    # Scaling of search direction
                    gamma=real(dot(Dg.T,Dx)) / (dot(Dg.T,Dg))
# gradientDescentSolver.m:336
                    searchDir=dot(gamma,searchDir)
# gradientDescentSolver.m:337
                    for j in arange(iters,1,- 1).reshape(-1):
                        beta=dot(rhoVals(j),real(dot(yVals(arange(),j).T,searchDir)))
# gradientDescentSolver.m:341
                        searchDir=searchDir + dot((alphas(j) - beta),sVals(arange(),j))
# gradientDescentSolver.m:342
                    searchDir=dot(1 / gamma,searchDir)
# gradientDescentSolver.m:345
                    searchDir=dot(norm(gradf1) / norm(searchDir),searchDir)
# gradientDescentSolver.m:346
    
    
    # Change search direction to steepest descent direction if current
        # direction is invalid
    if any(isnan(searchDir)) or any(isinf(searchDir)):
        searchDir=- gradf1
# gradientDescentSolver.m:353
    
    
    # Scale current search direction match magnitude of gradient
    searchDir=dot(norm(gradf1) / norm(searchDir),searchDir)
# gradientDescentSolver.m:357
    return searchDir
    
if __name__ == '__main__':
    pass
    
    # Update stepsize when objective update has not just occurred (adopted from
# FASTA.m)
    
@function
def updateStepsize(*args,**kwargs):
    varargin = updateStepsize.varargin
    nargin = updateStepsize.nargin

    Ds=searchDir0 - searchDir1
# gradientDescentSolver.m:363
    dotprod=real(dot(Dx,Ds))
# gradientDescentSolver.m:364
    tauS=norm(Dx) ** 2 / dotprod
# gradientDescentSolver.m:365
    
    tauM=dotprod / norm(Ds) ** 2
# gradientDescentSolver.m:366
    
    tauM=max(tauM,0)
# gradientDescentSolver.m:367
    if dot(2,tauM) > tauS:
        tau1=copy(tauM)
# gradientDescentSolver.m:369
    else:
        tau1=tauS - tauM / 2
# gradientDescentSolver.m:371
    
    if tau1 < 0 or isinf(tau1) or isnan(tau1):
        tau1=dot(tau0,1.5)
# gradientDescentSolver.m:374
    
    return
    
if __name__ == '__main__':
    pass
    
    
@function
def processIteration(*args,**kwargs):
    varargin = processIteration.varargin
    nargin = processIteration.nargin

    currentSolveTime=toc(startTime)
# gradientDescentSolver.m:379
    maxDiff=max(norm(Dx),maxDiff)
# gradientDescentSolver.m:380
    currentResidual=norm(Dx) / maxDiff
# gradientDescentSolver.m:381
    
    if logical_not(isempty(opts.xt)):
        reconEstimate=dot((dot(x1.T,opts.xt)) / (dot(x1.T,x1)),x1)
# gradientDescentSolver.m:385
        currentReconError=norm(opts.xt - reconEstimate) / norm(opts.xt)
# gradientDescentSolver.m:386
    
    
    if opts.recordTimes:
        solveTimes[iter]=currentSolveTime
# gradientDescentSolver.m:390
    
    if opts.recordResiduals:
        residuals[iter]=currentResidual
# gradientDescentSolver.m:393
    
    if opts.recordMeasurementErrors:
        currentMeasurementError=norm(abs(d1) - b0) / norm(b0)
# gradientDescentSolver.m:396
        measurementErrors[iter]=currentMeasurementError
# gradientDescentSolver.m:397
    
    if opts.recordReconErrors:
        assert_(logical_not(isempty(opts.xt)),concat(['You must specify the ground truth solution ','if the "recordReconErrors" flag is set to true.  Turn ','this flag off, or specify the ground truth solution.']))
        reconErrors[iter]=currentReconError
# gradientDescentSolver.m:403
    
    
    if opts.verbose == 2:
        displayVerboseOutput()
    
    
    # Terminate if solver surpasses maximum allocated timespan
    if currentSolveTime > opts.maxTime:
        stopNow=copy(true)
# gradientDescentSolver.m:412
        return stopNow
    
    
    # If user has supplied actual solution, use recon error to determine
        # termination
    if logical_not(isempty(currentReconError)):
        if currentReconError < opts.tol:
            stopNow=copy(true)
# gradientDescentSolver.m:420
            return stopNow
    
    if currentResidual < residualTolerance:
        # Give algorithm chance to recover if stuck at local minimum by
            # forcing update of objective function
        updateObjectiveNow=copy(true)
# gradientDescentSolver.m:428
        tolerancePenalty=tolerancePenalty + 1
# gradientDescentSolver.m:430
        if tolerancePenalty > opts.tolerancePenaltyLimit:
            stopNow=copy(true)
# gradientDescentSolver.m:432
            return stopNow
    
    
    stopNow=copy(false)
# gradientDescentSolver.m:437
    return stopNow
    
if __name__ == '__main__':
    pass
    
    # Display output to user based on provided options
    
@function
def displayVerboseOutput(*args,**kwargs):
    varargin = displayVerboseOutput.varargin
    nargin = displayVerboseOutput.nargin

    fprintf('Iter = %d',iter)
    fprintf(' | IterationTime = %.3f',currentSolveTime)
    fprintf(' | Resid = %.3e',currentResidual)
    fprintf(' | Stepsize = %.3e',tau0)
    if logical_not(isempty(currentMeasurementError)):
        fprintf(' | M error = %.3e',currentMeasurementError)
    
    if logical_not(isempty(currentReconError)):
        fprintf(' | R error = %.3e',currentReconError)
    
    fprintf('\n')
    return
    
if __name__ == '__main__':
    pass
    
    return
    
if __name__ == '__main__':
    pass
    