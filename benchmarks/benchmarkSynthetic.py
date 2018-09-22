# Generated with SMOP  0.41
from libsmop import *
# benchmarkSynthetic.m

    #                           benchmarkSynthetic.m
    
    # A general framework for benchmarking different phase retrieval algorithms
# using synthetic signals and either synthetic or real meaurement matrices.
    
    # I/O
# Inputs
# xitem:  A string that describes the desired x-axis label of the plot that
#         that is produced by this benchmark.  Valid options are 
#         ['m/n', 'snr','iterations', 'time'].  When using a 2D image with
#         Fourier measurements, then 'masks' should be used instead of
#         'm/n' to conntrol the number of Fourier masks.
# xvalues: A list of scalar values for which the performance of each
#          algorithm is measured.
# yitem:  A string to appear as the y-axis label.  Value should be 
#         drawn from ['reconError', 'measurementError', 'correlation'];
# algorithms: a cell array of options structs,
#             where each struct is the same as the input parameter 'opts'
#             for solvePhaseRetrieval. See the example scripts for details.
# dataSet: The name of dataset used. Currently supported options are
#          ['1DGaussian', '2DImage', 'transmissionMatrix'].
    
    # params: a struct of options containing the following fields:
#          verbose(boolean, default=false):      
#                   If true, the result of each trial will be reported.
#          numTrials(integer, default=1):        
#                   The number of trials each algorithm/dataset 
#                   combination will run.
#          policy(string, default='median'):     
#                   How to compute the final yvalue used for ploting from 
#                   the values one gets by running numTrials trials. It 
#                   currently supports
#                   ['median','mean','best','successRate'].
#          successConstant(real number,defualt=1e-5): 
#                   If the yvalue of the current  trial is less than this,
#                   the trial will be counted as a success. This parameter
#                   will only be used when policy='successRate'.
#          maxTime(positive real number,default=120) :  
#                   Max time allowed for a single algorithm.
#          recordSignals(boolean, default=false):
#                   Whether to record the recovered signal at each trial.
    
    
    # Outputs
#          results : A 3D struct consisting of the errors(error
#                   metric is based on yitem chosen by the user) of all 
#                   trials of algorithm/dataset combinations. Coordinates 
#                   are (x-axis value, algorithm index, trial number).
# recoveredSignals: A 4D cell array consisting of the recovered signal at
#                   each trial for each algorithm. Coordinates are (x-axis
#                   value, algorithm index, current trial number, the index
#                   of the recovered signal).
    
    # For more details, please look at the Phasepack User Guide.
    
    
    # PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017
    
    ## -----------------------------START----------------------------------
    
    # Main benchmark interface
    
@function
def benchmarkSynthetic(xitem=None,xvalues=None,yitem=None,algorithms=None,dataSet=None,params=None,*args,**kwargs):
    varargin = benchmarkSynthetic.varargin
    nargin = benchmarkSynthetic.nargin

    # Check if the inputs are valid. Note: the fields of algorithms will be checked by
# solvePhaseRetrieval implicitly.
    checkValidityOfInputs(xitem,xvalues,yitem,dataSet)
    # If params is not provided, create it.
    if logical_not(exist('params','var')):
        params=copy(struct)
# benchmarkSynthetic.m:73
    
    # Provide default value/validate the params provided by the user.
    params=manageOptionsForBenchmark(dataSet,params)
# benchmarkSynthetic.m:77
    # Check if the params are valid for the dataSet chosen by the user.
    params=checkValidityOfParams(dataSet,params,xitem)
# benchmarkSynthetic.m:80
    # Get the labels for x, y axis, numTrials, policy, successConstant,
# and recordSignals.
# For details of what these represent, see the header in this file or the User Guide.
    numTrials=params.numTrials
# benchmarkSynthetic.m:85
    policy=params.policy
# benchmarkSynthetic.m:86
    successConstant=params.successConstant
# benchmarkSynthetic.m:87
    recordSignals=params.recordSignals
# benchmarkSynthetic.m:88
    # create struct to store results of each trial.
    results=zeros(length(xvalues),length(algorithms),numTrials)
# benchmarkSynthetic.m:91
    if recordSignals:
        recoveredSignals=cell(length(xvalues),length(algorithms),numTrials)
# benchmarkSynthetic.m:93
    
    fprintf('Benchmarking on dataset %s with %s as x axis and the %s\n %s of %d trials as y axis...\n\n',dataSet,xitem,policy,yitem,numTrials)
    # Loop over the xvalues
    for p in arange(1,length(xvalues)).reshape(-1):
        fprintf('Running trails: %s=%g\n',xitem,xvalues(p))
        # Loop over the algorithms
        for k in arange(1,length(algorithms)).reshape(-1):
            opts,params=setupTrialParameters(algorithms[k],xitem,xvalues(p),dataSet,params,nargout=2)
# benchmarkSynthetic.m:103
            fprintf('  %s:',algorithms[k].algorithm)
            if numTrials == 1:
                fprintf('\n')
            # Loop over the random trials
            for q in arange(1,numTrials).reshape(-1):
                if numTrials > 1 and params.verbose == 0:
                    fprintf('*')
                # Create a random test problem with the right dimensions and SNR
                A,At,b0,xt,plotter,params=createProblemData(dataSet,params,nargout=6)
# benchmarkSynthetic.m:115
                n=numel(xt)
# benchmarkSynthetic.m:116
                opts.xt = copy(xt)
# benchmarkSynthetic.m:117
                startTime=copy(tic)
# benchmarkSynthetic.m:120
                x,outs,opts=solvePhaseRetrieval(A,At,b0,n,opts,nargout=3)
# benchmarkSynthetic.m:121
                elapsedTime=toc(startTime)
# benchmarkSynthetic.m:122
                plotter(x)
                title(sprintf('%s (%s=%s)',opts.algorithm,xitem,num2str(xvalues(p))),'fontsize',16)
                drawnow()
                yvalue=evalY(yitem,x,xt,A,b0)
# benchmarkSynthetic.m:130
                results[p,k,q]=yvalue
# benchmarkSynthetic.m:132
                if recordSignals:
                    recoveredSignals[p,k,q]=x
# benchmarkSynthetic.m:134
                # Report results of a trial if verbose is true.
                if params.verbose:
                    reportResult(opts.initMethod,opts.algorithm,xitem,xvalues(p),yitem,yvalue,elapsedTime,q)
            fprintf('\n')
    
    # Get final results in order to plot a comparison graph for algorithms
# at different x values.
    finalResults=getFinalResults(results,policy,successConstant,yitem)
# benchmarkSynthetic.m:149
    # Plot the comparison of performance graph among different chosen algorithms and
# initial methods combinations.
    plotComparison(xitem,yitem,xvalues,algorithms,dataSet,finalResults,policy)
    return finalResults,results,recoveredSignals
    
if __name__ == '__main__':
    pass
    
    ## Helper functions
    
    # Set the parameters needed to solve a phase retrieval problem with the
# specified dataset, xtime, and yitem.
    
@function
def setupTrialParameters(opts=None,xitem=None,xval=None,dataSet=None,params=None,*args,**kwargs):
    varargin = setupTrialParameters.varargin
    nargin = setupTrialParameters.nargin

    if 'iterations' == lower(xitem):
        opts.maxIters = copy(xval)
# benchmarkSynthetic.m:166
        opts.tol = copy(1e-10)
# benchmarkSynthetic.m:167
        # small number in order to make it
        # run maxIters iterations.
        opts.maxTime = copy(params.maxTime)
# benchmarkSynthetic.m:170
        opts.isComplex = copy(params.isComplex)
# benchmarkSynthetic.m:171
        opts.isNonNegativeOnly = copy(params.isNonNegativeOnly)
# benchmarkSynthetic.m:172
    else:
        if 'm/n' == lower(xitem):
            if strcmp(lower(dataSet),'1dgaussian') or strcmp(lower(dataSet),'transmissionmatrix'):
                params.m = copy(round(dot(params.n,xval)))
# benchmarkSynthetic.m:175
            opts.maxTime = copy(params.maxTime)
# benchmarkSynthetic.m:177
        else:
            if 'snr' == lower(xitem):
                opts.maxTime = copy(params.maxTime)
# benchmarkSynthetic.m:179
                params.snr = copy(xval)
# benchmarkSynthetic.m:180
            else:
                if 'time' == lower(xitem):
                    opts.maxTime = copy(xval)
# benchmarkSynthetic.m:182
                    opts.tol = copy(1e-10)
# benchmarkSynthetic.m:183
                    opts.maxIters = copy(100000000.0)
# benchmarkSynthetic.m:184
                    opts.recordTimes = copy(false)
# benchmarkSynthetic.m:185
                    opts.recordResiduals = copy(false)
# benchmarkSynthetic.m:186
                else:
                    if 'masks' == lower(xitem):
                        params.numMasks = copy(xval)
# benchmarkSynthetic.m:188
                        opts.maxTime = copy(params.maxTime)
# benchmarkSynthetic.m:189
                    else:
                        if 'angle' == lower(xitem):
                            opts.maxTime = copy(params.maxTime)
# benchmarkSynthetic.m:191
                            opts.initMethod = copy('angle')
# benchmarkSynthetic.m:192
                            opts.initAngle = copy(xval)
# benchmarkSynthetic.m:193
                        else:
                            # Raise an error if the given label for x axis is invalid.
                            error(concat(['invalid x label: ',xitem]))
    
    return opts,params
    
if __name__ == '__main__':
    pass
    
    # Run a specified algorithm on a specific dataset and get results
# Inputs:
# dataSet and params are as defined in benchmarkPhaseRetrieval
# opts: a struct consists of options specified by user for running the algorithm
#       on a specified dataset.
# Outputs:
# x:  n x 1 vector, estimation of xt given by the PR algorithm the subrountine invokes.
# xt: n x 1 vector, the real unknown signal.
# A:  m x n matrix/function handle(n x 1 -> m x 1).
# b0: m x 1 vector, the measurement.
# opts: a struct consists of options finally used for running the algorithm on a specified
#       dataset.
    
@function
def createProblemData(dataSet=None,params=None,*args,**kwargs):
    varargin = createProblemData.varargin
    nargin = createProblemData.nargin

    if '1dgaussian' == lower(dataSet):
        A,At,b0,xt,plotter=experimentGaussian1D(params.n,params.m,params.isComplex,params.isNonNegativeOnly,nargout=5)
# benchmarkSynthetic.m:216
    else:
        if '2dimage' == lower(dataSet):
            A,At,b0,xt,plotter=experimentImage2D(params.numMasks,params.imagePath,nargout=5)
# benchmarkSynthetic.m:219
        else:
            if 'transmissionmatrix' == lower(dataSet):
                A,b0,xt,plotter=experimentTransMatrixWithSynthSignal(params.n,params.m,params.A_cached,nargout=4)
# benchmarkSynthetic.m:221
                params.A_cached = copy(A)
# benchmarkSynthetic.m:222
                At=[]
# benchmarkSynthetic.m:223
            else:
                error('unknown dataset: %s\n',dataSet)
    
    
    # Add noise to achieve specified SNR
    if params.snr != inf:
        noise=randn(params.m,1)
# benchmarkSynthetic.m:230
        noise=dot(noise / norm(noise),norm(b0)) / params.snr
# benchmarkSynthetic.m:231
        b0=max(b0 + noise,0)
# benchmarkSynthetic.m:232
    
    
    return A,At,b0,xt,plotter,params
    
if __name__ == '__main__':
    pass
    
    # Calculate how good the solution is. Use the metric specified by yitem.
    
@function
def evalY(yitem=None,x=None,xt=None,A=None,b0=None,*args,**kwargs):
    varargin = evalY.varargin
    nargin = evalY.nargin

    #  Compute optimal rotation
    if 'reconerror' == lower(yitem):
        # solve for least-squares solution:  alpha*x = xt
        alpha=(dot(ravel(x).T,ravel(xt))) / (dot(ravel(x).T,ravel(x)))
# benchmarkSynthetic.m:244
        x=dot(alpha,x)
# benchmarkSynthetic.m:245
        yvalue=norm(ravel(xt) - ravel(x)) / norm(ravel(xt))
# benchmarkSynthetic.m:246
    else:
        if 'measurementerror' == lower(yitem):
            # Transform A into function handle if A is a matrix
            if isnumeric(A):
                At=lambda x=None: dot(A.T,x)
# benchmarkSynthetic.m:250
                A=lambda x=None: dot(A,x)
# benchmarkSynthetic.m:251
            yvalue=norm(abs(A(x)) - ravel(b0)) / norm(ravel(b0))
# benchmarkSynthetic.m:253
        else:
            if 'correlation' == lower(yitem):
                yvalue=abs(dot(x.T,xt) / norm(x) / norm(xt))
# benchmarkSynthetic.m:255
            else:
                error(concat(['invalid y label: ',yitem]))
    
    return yvalue
    
if __name__ == '__main__':
    pass
    
    # Get final results by averaging across all trials. 
# The possible policies are the following:
# mean: take the mean of all trials.
# best: take the best of all trials. If yitem=='reconerror' or 'measurementerror',
#       min value will be taken; If yitem=='correlation', max value will be taken.
# median: take the median of all trials.
# successrate: A success rate will be calculated. If yitem=='reconerror' or
#              'measurementerror', it is the percentage of values that are
#              smaller than the successConstant. If yitem=='correlation', it is
#              the percentage of values that are larger than the successConstant.
# The input struct results has size length(xvalues) x length(algorithms) x numTrials
# The output struct finalResults has size length(xvalues) x length(algorithms)
    
@function
def getFinalResults(results=None,policy=None,successConstant=None,yitem=None,*args,**kwargs):
    varargin = getFinalResults.varargin
    nargin = getFinalResults.nargin

    if 'mean' == lower(policy):
        finalResults=mean(results,3)
# benchmarkSynthetic.m:276
    else:
        if 'best' == lower(policy):
            if cellarray(['reconerror','measurementerror']) == lower(yitem):
                finalResults=min(results,[],3)
# benchmarkSynthetic.m:280
            else:
                if 'correlation' == lower(yitem):
                    finalResults=max(results,[],3)
# benchmarkSynthetic.m:282
                else:
                    error('invalid yitem: %s',yitem)
        else:
            if 'median' == lower(policy):
                finalResults=median(results,3)
# benchmarkSynthetic.m:287
            else:
                if 'successrate' == lower(policy):
                    if cellarray(['reconerror','measurementerror']) == lower(yitem):
                        finalResults=mean(results < successConstant,3)
# benchmarkSynthetic.m:291
                    else:
                        if 'correlation' == lower(yitem):
                            finalResults=mean(results > successConstant,3)
# benchmarkSynthetic.m:293
                        else:
                            error('invalid yitem: %s',yitem)
                else:
                    error('Invalid policy: %s',policy)
    
    return finalResults
    
if __name__ == '__main__':
    pass
    
    # Plot a performance curve for each algorithm
    
@function
def plotComparison(xitem=None,yitem=None,xvalues=None,algorithms=None,dataSet=None,finalResults=None,policy=None,*args,**kwargs):
    varargin = plotComparison.varargin
    nargin = plotComparison.nargin

    algNames=cellarray([])
# benchmarkSynthetic.m:304
    
    for k in arange(1,length(algorithms)).reshape(-1):
        if isfield(algorithms[k],'label') and numel(algorithms[k].label) > 0:
            algNames[k]=algorithms[k].label
# benchmarkSynthetic.m:308
        else:
            algNames[k]=algorithms[k].algorithm
# benchmarkSynthetic.m:310
    
    autoplot(xvalues,finalResults,algNames)
    title(char(strcat(cellarray([xitem]),cellarray([' VS ']),cellarray([yitem]),cellarray([' on ']),cellarray([dataSet]))))
    
    xlabel(xitem)
    
    ylabel(char(strcat(cellarray([policy]),cellarray(['(']),cellarray([yitem]),cellarray([')']))))
    
    ax=copy(gca)
# benchmarkSynthetic.m:319
    
    ax.FontSize = copy(12)
# benchmarkSynthetic.m:320
    
    legend('show','Location','northeastoutside')
    
    return
    
if __name__ == '__main__':
    pass
    
    # Report the result at the end of each trial if verbose is true.
    
@function
def reportResult(initMethod=None,algorithm=None,xitem=None,xvalue=None,yitem=None,yvalue=None,time=None,currentTrialNum=None,*args,**kwargs):
    varargin = reportResult.varargin
    nargin = reportResult.nargin

    if currentTrialNum == 1:
        fprintf('\n')
    
    fprintf('Trial: %d, initial method: %s, algorithm: %s, %s: %d, %s: %f, time: %f\n',currentTrialNum,initMethod,algorithm,xitem,xvalue,yitem,yvalue,time)
    return
    
if __name__ == '__main__':
    pass
    
    # Check if the inputs are valid.
    
@function
def checkValidityOfInputs(xitem=None,xvalues=None,yitem=None,dataSet=None,*args,**kwargs):
    varargin = checkValidityOfInputs.varargin
    nargin = checkValidityOfInputs.nargin

    assert_(logical_not(isempty(xvalues)),'The list xvalues must not be empty.')
    
    yitemList=cellarray(['reconerror','measurementerror','correlation'])
# benchmarkSynthetic.m:340
    checkIfInList('yitem',yitem,yitemList)
    
    xitem=lower(xitem)
# benchmarkSynthetic.m:344
    if '1dgaussian' == lower(dataSet):
        supportedList=cellarray(['iterations','m/n','snr','time','angle'])
# benchmarkSynthetic.m:347
    else:
        if 'transmissionmatrix' == lower(dataSet):
            supportedList=cellarray(['iterations','m/n','snr','time','angle'])
# benchmarkSynthetic.m:349
        else:
            if '2dimage' == lower(dataSet):
                supportedList=cellarray(['masks','iterations','angle'])
# benchmarkSynthetic.m:351
            else:
                error('unknown dataset: %s\n',dataSet)
    
    checkIfInList('xitem',xitem,supportedList,strcat('For ',lower(dataSet),', '))
    return
    
if __name__ == '__main__':
    pass
    
    # Check if the params are valid
    
@function
def checkValidityOfParams(dataSet=None,params=None,xitem=None,*args,**kwargs):
    varargin = checkValidityOfParams.varargin
    nargin = checkValidityOfParams.nargin

    if '1dgaussian' == lower(dataSet):
        assert_(isfield(params,'n'),'User must specify signal dimension in params.n')
        checkIfNumber('params.n',params.n)
        if logical_not(strcmp(xitem,'m/n')):
            assert_(isfield(params,'m'),concat(['User must specify the number of measurements in params.m when using xlabel ',xitem]))
    else:
        if '2dimage' == lower(dataSet):
            assert_(logical_not(strcmp(xitem,'m/n')),'xlabel m/n is not supported when using 2D images.  Instead use "masks"')
            if logical_not((xitem == 'masks')):
                assert_(exist('params.numMasks'),concat(['User must specify the number of Fourier masks in params.numMasks when using xlabel ',xitem]))
        else:
            if 'transmissionmatrix' == lower(dataSet):
                params.A_cached = copy([])
# benchmarkSynthetic.m:374
                assert_(isfield(params,'n'),'User must specify signal dimension in params.n.  Options are {256,1600,4096} for the tranmissionMatrix dataset.')
                assert_(params.n == 256 or params.n == 1600 or params.n == 4096,concat(['Invalid choice (',num2str(params.n),') ofr params.n for the transmissionMatrix dataset. Valid choices are {256,1600,4096}.']))
            else:
                error('unknown dataset: %s\n',dataSet)
    
    if logical_not(isfield(params,'snr')):
        params.snr = copy(Inf)
# benchmarkSynthetic.m:383
    
    return params
    
if __name__ == '__main__':
    pass
    