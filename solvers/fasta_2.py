# Generated with SMOP  0.41
from libsmop import *
# fasta.m

    #                               FASTA.M
#      This method solves the problem
#                        minimize f(Ax)+g(x)
#   Where A is a matrix, f is differentiable, and both f and g are convex.  
#   The algorithm is an adaptive/accelerated forward-backward splitting.  
#   The user supplies function handles that evaluate 'f' and 'g'.  The user 
#   also supplies a function that evaluates the gradient of 'f' and the
#   proximal operator of 'g', which is given by
#                proxg(z,t) = argmin t*g(x)+.5||x-z||^2.
    
    #  Inputs:
#    A     : A matrix (or optionally a function handle to a method) that 
#             returns A*x
#    At    : The adjoint (transpose) of 'A.' Optionally, a function handle
#             may be passed.  
#    gradf : A function of z, computes the gradient of f at z
#    proxg : A function of z and t, the proximal operator of g with
#             stepsize t.  
#    x0    : The initial guess, usually a vector of zeros
#    f     : A function of x, computes the value of f
#    g     : A function of x, computes the value of g
#    opts  : An optional struct with options.  The commonly used fields
#             of 'opts' are:
#               maxIters : (integer, default=1e4) The maximum number of 
#                          iterations allowed before termination.
#               tol      : (double, default=1e-3) The stopping tolerance.  
#                               A smaller value of 'tol' results in more
#                               iterations.
#               verbose  : (boolean, default=false)  If true, print out
#                               convergence information on each iteration.
#               recordObjective:  (boolean, default=false) Compute and
#                               record the objective of each iterate.
#               recordIterates :  (boolean, default=false) Record every
#                               iterate in a cell array.
#            To use these options, set the corresponding field in 'opts'. 
#            For example:  
#                      >> opts.tol=1e-8;
#                      >> opts.maxIters = 100;
    
    #  Outputs:
#    sol  : The approximate solution
#    outs : A struct with convergence information
#    opts : A complete struct of options, containing all the values 
#           (including defaults) that were used by the solver.
    
    #   For more details, see the FASTA user guide, or the paper "A field guide
#   to forward-backward splitting with a FASTA implementation."
    
    #   Copyright: Tom Goldstein, 2014.
    
    
@function
def fasta(A=None,At=None,f=None,gradf=None,g=None,proxg=None,x0=None,opts=None,*args,**kwargs):
    varargin = fasta.varargin
    nargin = fasta.nargin

    ##  Check whether we have function handles or matrices
    if logical_not(isnumeric(A)):
        assert_(logical_not(isnumeric(At)),'If A is a function handle, then At must be a handle as well.')
    
    #  If we have matrices, create functions so we only have to treat one case
    if isnumeric(A):
        At=lambda x=None: dot(A.T,x)
# fasta.m:60
        A=lambda x=None: dot(A,x)
# fasta.m:61
    
    
    ## Check preconditions, fill missing optional entries in 'opts'
    if logical_not(exist('opts','var')):
        opts=[]
# fasta.m:66
    
    opts=setDefaults(opts,A,At,x0,gradf)
# fasta.m:68
    
    # Verify that At=A'
    checkAdjoint(A,At,x0)
    if opts.verbose:
        fprintf('%sFASTA:\tmode = %s\n\tmaxIters = %i,\ttol = %1.2d\n',opts.stringHeader,opts.mode,opts.maxIters,opts.tol)
    
    
    ## Record some frequently used information from opts
    tau1=opts.tau
# fasta.m:78
    
    max_iters=opts.maxIters
# fasta.m:79
    
    W=opts.window
# fasta.m:80
    
    
    ## Allocate memory
    residual=zeros(max_iters,1)
# fasta.m:83
    
    normalizedResid=zeros(max_iters,1)
# fasta.m:84
    
    taus=zeros(max_iters,1)
# fasta.m:85
    
    fVals=zeros(max_iters,1)
# fasta.m:86
    
    objective=zeros(max_iters + 1,1)
# fasta.m:87
    
    funcValues=zeros(max_iters,1)
# fasta.m:88
    
    totalBacktracks=0
# fasta.m:89
    
    backtrackCount=0
# fasta.m:90
    
    ## Intialize array values
    x1=copy(x0)
# fasta.m:93
    d1=A(x1)
# fasta.m:94
    f1=f(d1)
# fasta.m:95
    fVals[1]=f1
# fasta.m:96
    gradf1=At(gradf(d1))
# fasta.m:97
    
    if opts.accelerate:
        x_accel1=copy(x0)
# fasta.m:101
        d_accel1=copy(d1)
# fasta.m:102
        alpha1=1
# fasta.m:103
    
    
    #  To handle non-monotonicity
    maxResidual=- Inf
# fasta.m:107
    
    minObjectiveValue=copy(Inf)
# fasta.m:108
    
    
    #  If user has chosen to record objective, then record initial value
    if opts.recordObjective:
        objective[1]=f1 + g(x0)
# fasta.m:112
    
    
    tic
    
    ## Begin Loop
    for i in arange(1,max_iters).reshape(-1):
        ##  Rename iterates relative to loop index.  "0" denotes index i, and "1" denotes index i+1
        x0=copy(x1)
# fasta.m:119
        gradf0=copy(gradf1)
# fasta.m:120
        tau0=copy(tau1)
# fasta.m:121
        ##  FBS step: obtain x_{i+1} from x_i
        x1hat=x0 - dot(tau0,gradf0)
# fasta.m:124
        x1=proxg(x1hat,tau0)
# fasta.m:125
        ##  Non-monotone backtracking line search
        Dx=x1 - x0
# fasta.m:128
        d1=A(x1)
# fasta.m:129
        f1=f(d1)
# fasta.m:130
        if opts.backtrack:
            M=max(fVals(arange(max(i - W,1),max(i - 1,1))))
# fasta.m:132
            backtrackCount=0
# fasta.m:133
            while f1 - 1e-12 > M + real(dot(ravel(Dx),ravel(gradf0))) + norm(ravel(Dx)) ** 2 / (dot(2,tau0)) and backtrackCount < 20 or logical_not(isreal(f1)):

                tau0=dot(tau0,opts.stepsizeShrink)
# fasta.m:136
                x1hat=x0 - dot(tau0,gradf0)
# fasta.m:137
                x1=proxg(x1hat,tau0)
# fasta.m:138
                d1=A(x1)
# fasta.m:139
                f1=f(d1)
# fasta.m:140
                Dx=x1 - x0
# fasta.m:141
                backtrackCount=backtrackCount + 1
# fasta.m:142

            totalBacktracks=totalBacktracks + backtrackCount
# fasta.m:144
        if opts.verbose and backtrackCount > 10:
            fprintf('%s\tWARNING: excessive backtracking (%d steps), current stepsize is %0.2d\n',opts.stringHeader,backtrackCount,tau0)
        ## Record convergence information
        taus[i]=tau0
# fasta.m:153
        residual[i]=norm(ravel(Dx)) / tau0
# fasta.m:154
        maxResidual=max(maxResidual,residual(i))
# fasta.m:155
        normalizer=max(norm(ravel(gradf0)),norm(ravel(x1) - ravel(x1hat)) / tau0) + opts.eps_n
# fasta.m:156
        normalizedResid[i]=residual(i) / normalizer
# fasta.m:157
        fVals[i]=f1
# fasta.m:158
        funcValues[i]=opts.function(x1)
# fasta.m:160
        if opts.recordObjective:
            objective[i + 1]=f1 + g(x1)
# fasta.m:162
            newObjectiveValue=objective(i + 1)
# fasta.m:163
        else:
            newObjectiveValue=residual(i)
# fasta.m:165
        if opts.recordIterates:
            iterates[i]=x1
# fasta.m:169
        if newObjectiveValue < minObjectiveValue:
            bestObjectiveIterate=copy(x1)
# fasta.m:173
            bestObjectiveIterateHat=copy(x1hat)
# fasta.m:174
            minObjectiveValue=copy(newObjectiveValue)
# fasta.m:175
        if opts.verbose > 1:
            fprintf('%s%d: resid = %0.2d, backtrack = %d, tau = %d',opts.stringHeader,i,residual(i),backtrackCount,tau0)
            if opts.recordObjective:
                fprintf(', objective = %d\n',objective(i + 1))
            else:
                fprintf('\n')
        ## Test stopping criteria
    #  If we stop, then record information in the output struct
        if opts.stopNow(x1,i,residual(i),normalizedResid(i),maxResidual,opts) or i > max_iters:
            outs=[]
# fasta.m:191
            outs.solveTime = copy(toc)
# fasta.m:192
            outs.residuals = copy(residual(arange(1,i)))
# fasta.m:193
            outs.stepsizes = copy(taus(arange(1,i)))
# fasta.m:194
            outs.normalizedResiduals = copy(normalizedResid(arange(1,i)))
# fasta.m:195
            outs.objective = copy(objective(arange(1,i)))
# fasta.m:196
            outs.funcValues = copy(funcValues(arange(1,i)))
# fasta.m:197
            outs.backtracks = copy(totalBacktracks)
# fasta.m:198
            outs.L = copy(opts.L)
# fasta.m:199
            outs.initialStepsize = copy(opts.tau)
# fasta.m:200
            outs.iterationCount = copy(i)
# fasta.m:201
            if logical_not(opts.recordObjective):
                outs.objective = copy('Not Recorded')
# fasta.m:203
            if opts.recordIterates:
                outs.iterates = copy(iterates)
# fasta.m:206
            outs.bestObjectiveIterateHat = copy(bestObjectiveIterateHat)
# fasta.m:208
            sol=copy(bestObjectiveIterate)
# fasta.m:209
            if opts.verbose:
                fprintf('%s\tDone:  time = %0.3f secs, iterations = %i\n',opts.stringHeader,toc,outs.iterationCount)
            return sol,outs,opts
        if opts.adaptive and logical_not(opts.accelerate):
            ## Compute stepsize needed for next iteration using BB/spectral method
            gradf1=At(gradf(d1))
# fasta.m:218
            Dg=gradf1 + (x1hat - x0) / tau0
# fasta.m:219
            dotprod=real(dot(ravel(Dx),ravel(Dg)))
# fasta.m:220
            tau_s=norm(ravel(Dx)) ** 2 / dotprod
# fasta.m:221
            tau_m=dotprod / norm(ravel(Dg)) ** 2
# fasta.m:222
            tau_m=max(tau_m,0)
# fasta.m:223
            if dot(2,tau_m) > tau_s:
                tau1=copy(tau_m)
# fasta.m:225
            else:
                tau1=tau_s - dot(0.5,tau_m)
# fasta.m:227
            if tau1 < 0 or isinf(tau1) or isnan(tau1):
                tau1=dot(tau0,1.5)
# fasta.m:230
        if opts.accelerate:
            ## Use FISTA-type acceleration
            x_accel0=copy(x_accel1)
# fasta.m:236
            d_accel0=copy(d_accel1)
# fasta.m:237
            alpha0=copy(alpha1)
# fasta.m:238
            x_accel1=copy(x1)
# fasta.m:239
            d_accel1=copy(d1)
# fasta.m:240
            if opts.restart and dot((ravel(x0) - ravel(x1)).T,(ravel(x1) - ravel(x_accel0))) > 0:
                alpha0=1
# fasta.m:243
            #  Calculate acceleration parameter
            alpha1=(1 + sqrt(1 + dot(4,alpha0 ** 2))) / 2
# fasta.m:246
            x1=x_accel1 + dot((alpha0 - 1) / alpha1,(x_accel1 - x_accel0))
# fasta.m:248
            d1=d_accel1 + dot((alpha0 - 1) / alpha1,(d_accel1 - d_accel0))
# fasta.m:249
            gradf1=At(gradf(d1))
# fasta.m:251
            fVals[i]=f(d1)
# fasta.m:252
            tau1=copy(tau0)
# fasta.m:253
        if logical_not(opts.adaptive) and logical_not(opts.accelerate):
            gradf1=At(gradf(d1))
# fasta.m:257
            tau1=copy(tau0)
# fasta.m:258
    
    return sol,outs,opts
    
@function
def checkAdjoint(A=None,At=None,x=None,*args,**kwargs):
    varargin = checkAdjoint.varargin
    nargin = checkAdjoint.nargin

    x=randn(size(x))
# fasta.m:267
    Ax=A(x)
# fasta.m:268
    y=randn(size(Ax))
# fasta.m:269
    Aty=At(y)
# fasta.m:270
    innerProduct1=dot(ravel(Ax).T,ravel(y))
# fasta.m:272
    innerProduct2=dot(ravel(x).T,ravel(Aty))
# fasta.m:273
    error=abs(innerProduct1 - innerProduct2) / max(abs(innerProduct1),abs(innerProduct2))
# fasta.m:274
    assert_(error < 0.001,concat(['"At" is not the adjoint of "A".  Check the definitions of these operators. Error=',num2str(error)]))
    return
    ## Fill in the struct of options with the default values
    
@function
def setDefaults(opts=None,A=None,At=None,x0=None,gradf=None,*args,**kwargs):
    varargin = setDefaults.varargin
    nargin = setDefaults.nargin

    #  maxIters: The maximum number of iterations
    if logical_not(isfield(opts,'maxIters')):
        opts.maxIters = copy(1000)
# fasta.m:286
    
    # tol:  The relative decrease in the residuals before the method stops
    if logical_not(isfield(opts,'tol')):
        opts.tol = copy(0.001)
# fasta.m:291
    
    # verbose:  If 'true' then print status information on every iteration
    if logical_not(isfield(opts,'verbose')):
        opts.verbose = copy(false)
# fasta.m:296
    
    # recordObjective:  If 'true' then evaluate objective at every iteration
    if logical_not(isfield(opts,'recordObjective')):
        opts.recordObjective = copy(false)
# fasta.m:301
    
    # recordIterates:  If 'true' then record iterates in cell array
    if logical_not(isfield(opts,'recordIterates')):
        opts.recordIterates = copy(false)
# fasta.m:306
    
    # adaptive:  If 'true' then use adaptive method.
    if logical_not(isfield(opts,'adaptive')):
        opts.adaptive = copy(true)
# fasta.m:311
    
    # accelerate:  If 'true' then use FISTA-type adaptive method.
    if logical_not(isfield(opts,'accelerate')):
        opts.accelerate = copy(false)
# fasta.m:316
    
    # restart:  If 'true' then restart the acceleration of FISTA.
#   This only has an effect when opts.accelerate=true
    if logical_not(isfield(opts,'restart')):
        opts.restart = copy(true)
# fasta.m:322
    
    # backtrack:  If 'true' then use backtracking line search
    if logical_not(isfield(opts,'backtrack')):
        opts.backtrack = copy(true)
# fasta.m:327
    
    # stepsizeShrink:  Coefficient used to shrink stepsize when backtracking
# kicks in
    if logical_not(isfield(opts,'stepsizeShrink')):
        opts.stepsizeShrink = copy(0.2)
# fasta.m:333
        if logical_not(opts.adaptive) or opts.accelerate:
            opts.stepsizeShrink = copy(0.5)
# fasta.m:335
    
    #  Create a mode string that describes which variant of the method is used
    opts.mode = copy('plain')
# fasta.m:340
    if opts.adaptive:
        opts.mode = copy('adaptive')
# fasta.m:342
    
    if opts.accelerate:
        if opts.restart:
            opts.mode = copy('accelerated(FISTA)+restart')
# fasta.m:346
        else:
            opts.mode = copy('accelerated(FISTA)')
# fasta.m:348
    
    # W:  The window to look back when evaluating the max for the line search
    if logical_not(isfield(opts,'window')):
        opts.window = copy(10)
# fasta.m:355
    
    # eps_r:  Epsilon to prevent ratio residual from dividing by zero
    if logical_not(isfield(opts,'eps_r')):
        opts.eps_r = copy(1e-08)
# fasta.m:360
    
    # eps_n:  Epsilon to prevent normalized residual from dividing by zero
    if logical_not(isfield(opts,'eps_n')):
        opts.eps_n = copy(1e-08)
# fasta.m:365
    
    #  L:  Lipschitz constant for smooth term.  Only needed if tau has not been
#   set, in which case we need to approximate L so that tau can be
#   computed.
    if (logical_not(isfield(opts,'L')) or opts.L < 0) and (logical_not(isfield(opts,'tau')) or opts.tau < 0):
        x1=randn(size(x0))
# fasta.m:372
        x2=randn(size(x0))
# fasta.m:373
        gradf1=At(gradf(A(x1)))
# fasta.m:374
        gradf2=At(gradf(A(x2)))
# fasta.m:375
        opts.L = copy(norm(ravel(gradf1) - ravel(gradf2)) / norm(ravel(x2) - ravel(x1)))
# fasta.m:376
        opts.L = copy(max(opts.L,1e-06))
# fasta.m:377
        opts.tau = copy(2 / opts.L / 10)
# fasta.m:378
    
    assert_(opts.tau > 0,concat(['Invalid step size: ',num2str(opts.tau)]))
    #  Set tau if L was set by user
    if (logical_not(isfield(opts,'tau')) or opts.tau < 0):
        opts.tau = copy(1.0 / opts.L)
# fasta.m:384
    else:
        opts.L = copy(1 / opts.tau)
# fasta.m:386
    
    # function:  An optional function that is computed and stored after every
# iteration
    if logical_not(isfield(opts,'function')):
        opts.function = copy(lambda x=None: 0)
# fasta.m:392
    
    # stringHeader:  Append this string to beginning of all output
    if logical_not(isfield(opts,'stringHeader')):
        opts.stringHeader = copy('')
# fasta.m:397
    
    #  The code below is for stopping rules
#  The field 'stopNow' is a function that returns 'true' if the iteration
#  should be terminated.  The field 'stopRule' is a string that allows the
#  user to easily choose default values for 'stopNow'.  The default
#  stopping rule terminates when the relative residual gets small.
    if isfield(opts,'stopNow'):
        opts.stopRule = copy('custom')
# fasta.m:406
    
    if logical_not(isfield(opts,'stopRule')):
        opts.stopRule = copy('hybridResidual')
# fasta.m:410
    
    if strcmp(opts.stopRule,'residual'):
        opts.stopNow = copy(lambda x1=None,iter=None,resid=None,normResid=None,maxResidual=None,opts=None: resid < opts.tol)
# fasta.m:414
    
    if strcmp(opts.stopRule,'iterations'):
        opts.stopNow = copy(lambda x1=None,iter=None,resid=None,normResid=None,maxResidual=None,opts=None: iter > opts.maxIters)
# fasta.m:418
    
    # Stop when normalized residual is small
    if strcmp(opts.stopRule,'normalizedResidual'):
        opts.stopNow = copy(lambda x1=None,iter=None,resid=None,normResid=None,maxResidual=None,opts=None: normResid < opts.tol)
# fasta.m:423
    
    # Divide by residual at iteration k by maximum residual over all iterations.
# Terminate when this ratio gets small.
    if strcmp(opts.stopRule,'ratioResidual'):
        opts.stopNow = copy(lambda x1=None,iter=None,resid=None,normResid=None,maxResidual=None,opts=None: resid / (maxResidual + opts.eps_r) < opts.tol)
# fasta.m:429
    
    # Default behavior:  Stop if EITHER normalized or ration residual is small
    if strcmp(opts.stopRule,'hybridResidual'):
        opts.stopNow = copy(lambda x1=None,iter=None,resid=None,normResid=None,maxResidual=None,opts=None: resid / (maxResidual + opts.eps_r) < opts.tol or normResid < opts.tol)
# fasta.m:434
    
    assert_(isfield(opts,'stopNow'),concat(['Invalid choice for stopping rule: ',opts.stopRule]))
    return opts