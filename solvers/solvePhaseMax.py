# Generated with SMOP  0.41
from libsmop import *
# solvePhaseMax.m

    #                           solvePhaseMax.m
    
    #  Implementation of the PhaseMax algorithm proposed in the paper using
#  FASTA. Note: For this code to run, the solver "fasta.m" must be in your
#  path.
    
    ## I/O
#  Inputs:
#     A:    m x n matrix or a function handle to a method that
#           returns A*x.     
#     At:   The adjoint (transpose) of 'A'. If 'A' is a function handle, 'At'
#           must be provided.
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
#  See the script 'testPhaseMaxGaussian.m' and 'testPhaseMaxFourier.m' for 
#  two examples of proper usage of this function.
    
    ## Notations
#  
#  
## Algorithm Description
#  Solve the PhaseMax signal reconstruction problem
#         maximize <x0,x>
#         subject to |Ax|<b0 # <=
# 
#  The problem is solved by approximately enforcing the constraints using a
#  quadratic barrier function.  A continuation method is used to increase
#  the strength of the barrier function until a high level of numerical
#  accuracy is reached.
#  The objective plus the quadratic barrier has the form
#     <-x0,x> + 0.5*max{|Ax|-b,0}^2.
    
    #  For a detailed explanation, see the PhaseMax paper referenced below. For
#  more details about FASTA, see the FASTA user guide, or the paper "A
#  field guide to forward-backward splitting with a FASTA implementation."
    
    ## References
#  Paper Title:   PhaseMax: Convex Phase Retrieval via Basis Pursuit
#  Authors:       Tom Goldstein, Christoph Studer
#  arXiv Address: https://arxiv.org/abs/1610.07531
#  
#  Copyright Goldstein & Studer, 2016.  For more details, visit 
#  https://www.cs.umd.edu/~tomg/projects/phasemax/
    
    
@function
def solvePhaseMax(A=None,At=None,b0=None,x0=None,opts=None,*args,**kwargs):
    varargin = solvePhaseMax.varargin
    nargin = solvePhaseMax.nargin

    # Initialization
    m=length(b0)
# solvePhaseMax.m:60
    
    n=length(x0)
# solvePhaseMax.m:61
    
    remainIters=opts.maxIters
# solvePhaseMax.m:62
    
    # It's initialized to opts.maxIters.
    
    #  Normalize the initial guess relative to the number of measurements
    x0=dot(dot(dot((x0 / norm(ravel(x0))),mean(ravel(b0))),(m / n)),100)
# solvePhaseMax.m:66
    
    sol=multiply(x0,min(b0 / abs(A(x0))))
# solvePhaseMax.m:69
    
    ending=0
# solvePhaseMax.m:72
    
    iter=0
# solvePhaseMax.m:73
    currentTime=[]
# solvePhaseMax.m:74
    currentResid=[]
# solvePhaseMax.m:75
    currentReconError=[]
# solvePhaseMax.m:76
    currentMeasurementError=[]
# solvePhaseMax.m:77
    
    solveTimes,measurementErrors,reconErrors,residuals=initializeContainers(opts,nargout=4)
# solvePhaseMax.m:80
    
    f=lambda z=None: dot(0.5,norm(max(abs(z) - b0,0)) ** 2)
# solvePhaseMax.m:83
    
    gradf=lambda z=None: (multiply(sign(z),max(abs(z) - b0,0)))
# solvePhaseMax.m:84
    
    
    # Options to hand to fasta
    fastaOpts.maxIters = copy(opts.maxIters)
# solvePhaseMax.m:87
    fastaOpts.stopNow = copy(lambda x=None,iter=None,resid=None,normResid=None,maxResid=None,opts=None: processIteration(x,resid))
# solvePhaseMax.m:88
    
    # solveTime, residual and error at each iteration.
    fastaOpts.verbose = copy(0)
# solvePhaseMax.m:91
    startTime=copy(tic)
# solvePhaseMax.m:92
    
    constraintError=norm(abs(A(sol)) - b0)
# solvePhaseMax.m:93
    
    while remainIters > logical_and(0,logical_not(ending)):

        g=lambda x=None: - real(dot(x0.T,x))
# solvePhaseMax.m:95
        proxg=lambda x=None,t=None: x + dot(t,x0)
# solvePhaseMax.m:96
        fastaOpts.tol = copy(norm(x0) / 100)
# solvePhaseMax.m:97
        # Call FASTA to solve the inner minimization problem
        sol,fastaOuts=fasta(A,At,f,gradf,g,proxg,sol,fastaOpts,nargout=2)
# solvePhaseMax.m:99
        fastaOpts.tau = copy(fastaOuts.stepsizes(end()))
# solvePhaseMax.m:101
        x0=x0 / 10
# solvePhaseMax.m:102
        # Update the max number of iterations for fasta
        remainIters=remainIters - fastaOuts.iterationCount
# solvePhaseMax.m:105
        fastaOpts.maxIters = copy(min(opts.maxIters,remainIters))
# solvePhaseMax.m:106
        newConstraintError=norm(max(abs(A(sol)) - b0,0))
# solvePhaseMax.m:109
        relativeChange=abs(constraintError - newConstraintError) / norm(b0)
# solvePhaseMax.m:110
        if relativeChange < opts.tol:
            break
        constraintError=copy(newConstraintError)
# solvePhaseMax.m:114

    
    # Create output according to the options chosen by user
    outs=generateOutputs(opts,iter,solveTimes,measurementErrors,reconErrors,residuals)
# solvePhaseMax.m:119
    
    if opts.verbose == 1:
        displayVerboseOutput(iter,currentTime,currentResid,currentReconError,currentMeasurementError)
    
    # Runs code upon each FASTA iteration. Returns whether FASTA should
    # terminate.
    
@function
def processIteration(x=None,residual=None,*args,**kwargs):
    varargin = processIteration.varargin
    nargin = processIteration.nargin

    iter=iter + 1
# solvePhaseMax.m:130
    
    # If xt is provided, reconstruction error will be computed and used for stopping
        # condition. Otherwise, residual will be computed and used for stopping
        # condition.
    if logical_not(isempty(opts.xt)):
        xt=opts.xt
# solvePhaseMax.m:136
        alpha=(dot(ravel(x).T,ravel(xt))) / (dot(ravel(x).T,ravel(x)))
# solvePhaseMax.m:138
        x=dot(alpha,x)
# solvePhaseMax.m:139
        currentReconError=norm(x - xt) / norm(xt)
# solvePhaseMax.m:140
        if opts.recordReconErrors:
            reconErrors[iter]=currentReconError
# solvePhaseMax.m:142
    
    if isempty(opts.xt):
        currentResid=copy(residual)
# solvePhaseMax.m:147
    
    if opts.recordResiduals:
        residuals[iter]=residual
# solvePhaseMax.m:151
    
    currentTime=toc(startTime)
# solvePhaseMax.m:154
    
    if opts.recordTimes:
        solveTimes[iter]=currentTime
# solvePhaseMax.m:156
    
    if opts.recordMeasurementErrors:
        currentMeasurementError=norm(abs(A(sol)) - b0) / norm(b0)
# solvePhaseMax.m:159
        measurementErrors[iter]=currentMeasurementError
# solvePhaseMax.m:160
    
    
    # Display verbose output if specified
    if opts.verbose == 2:
        displayVerboseOutput(iter,currentTime,currentResid,currentReconError,currentMeasurementError)
    
    # Test stopping criteria.
    stop=copy(false)
# solvePhaseMax.m:169
    if currentTime > opts.maxTime:
        stop=copy(true)
# solvePhaseMax.m:171
    
    if logical_not(isempty(opts.xt)):
        assert_(logical_not(isempty(currentReconError)),'If xt is provided, currentReconError must be provided.')
        stop=stop or currentReconError < opts.tol
# solvePhaseMax.m:175
        ending=copy(stop)
# solvePhaseMax.m:176
    
    stop=stop or residual < fastaOpts.tol
# solvePhaseMax.m:178
    
    return stop
    
if __name__ == '__main__':
    pass
    
    return stop
    
if __name__ == '__main__':
    pass
    