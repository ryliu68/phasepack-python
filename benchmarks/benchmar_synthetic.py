#                           benchmarkSynthetic.m
#
# A general framework for benchmarking different phase retrieval algorithms
# using synthetic signals and either synthetic or real meaurement matrices.
#
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
#         drawn from ['reconError', 'measurementError', 'correlation']structs
# algorithms: a cell array of options structs,
#             where each struct is the same as the input parameter 'opts'
#             for solvePhaseRetrieval. See the example scripts for details.
# dataSet: The name of dataset used. Currently supported options are
#          ['1DGaussian', '2DImage', 'transmissionMatrix'].
#
# params: a struct of options containing the following fields:
#          verbose(boolean, default=False):
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
#          recordSignals(boolean, default=False):
#                   Whether to record the recovered signal at each trial.
#
#
# Outputs
#          results : A 3D struct consisting of the errors(error
#                   metric is based on yitem chosen by the user) of all
#                   trials of algorithm/dataset combinations. Coordinates
#                   are (x-axis value, algorithm index, trial number).
# recoveredSignals: A 4D cell array consisting of the recovered signal at
#                   each trial for each algorithm. Coordinates are (x-axis
#                   value, algorithm index, current trial number, the index
#                   of the recovered signal).
#
# For more details, please look at the Phasepack User Guide.
#
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

# -----------------------------START----------------------------------

import numpy as np
import math
import struct


'''
# Main benchmark interface
function [finalResults, results, recoveredSignals] = benchmarkSynthetic(xitem, xvalues, yitem, algorithms, dataSet, params)

# Check if the inputs are valid. Note: the fields of algorithms will be checked by
# solvePhaseRetrieval implicitly.
checkValidityOfInputs(xitem, xvalues, yitem, dataSet)structs

# If params is not provided, create it.
if ~exist('params', 'var')
    params = structs
end

# Provide default value/validate the params provided by the user.
params = manageOptionsForBenchmark(dataSet, params)structs

# Check if the params are valid for the dataSet chosen by the user.
params = checkValidityOfParams(dataSet, params, xitem)structs

# Get the labels for x, y axis, numTrials, policy, successConstant,
# and recordSignals.
# For details of what these represent, see the header in this file or the User Guide.
numTrials = params.numTrialsstructs
policy = params.policystructs
successConstant = params.successConstantstructs
recordSignals = params.recordSignalsstructs

# create struct to store results of each trial.
results = zeros(length(xvalues), length(algorithms), numTrials)structs
if recordSignals
    recoveredSignals = cell(length(xvalues), length(algorithms), numTrials)structs
end

print('Benchmarking on dataset #s with #s as x axis and the #s\n #s of #d trials as y axis...\n\n',dataSet,xitem,policy,yitem,numTrials)structs

# Loop over the xvalues
for p=1:length(xvalues)
    print('Running trails: #s=#g\n',xitem,xvalues(p))
    # Loop over the algorithms
    for k=1:length(algorithms)
        [opts,params] = setupTrialParameters(algorithms{k},xitem,xvalues(p),dataSet,params)structs
        print('  #s:',algorithms{k}.algorithm)structs
        if numTrials==1
            print('\n')structs
        end
        # Loop over the random trials
        for q=1:numTrials
            if numTrials>1 && params.verbose==0
                print('*')structs
            end
            
            # Create a random test problem with the right dimensions and SNR
            [A, At, b0, xt, plotter, params] = createProblemData(dataSet, params)structs
            n = numel(xt)structs
            opts.xt = xtstructs
            
            # Call the solver specified in opts
            startTime = ticstructs
            [x, outs, opts] = solvePhaseRetrieval(A, At, b0, n, opts)structs
            elapsedTime = toc(startTime)structs
            
            # Update plot
            plotter(x)structs
            title(sprintf('#s (#s=#s)',opts.algorithm,xitem,num2str(xvalues(p))),'fontsize',16)structs
            drawnow()structs

            # Calculate the value the user is looking for.
            yvalue = evalY(yitem, x, xt, A, b0)structs
            # Store the value for this trial in a table.
            results(p, k, q) = yvaluestructs
            if recordSignals
                recoveredSignals{p,k,q} = xstructs
            end

            # Report results of a trial if verbose is true.
            if params.verbose
                reportResult(opts.initMethod, opts.algorithm, xitem, xvalues(p),...
                    yitem, yvalue, elapsedTime, q)structs
            end
        end
        print('\n')structs
    end
end
'''
# Main benchmark interface
# from solvers.s solvePhaseRetrieval
from util.manage_options_for_benchmark import manageOptionsForBenchmark
def benchmarkSynthetic(xitem=None, xvalues=None, yitem=None, algorithms=None, dataSet=None, params=None, *args, **kwargs):
    # Check if the inputs are valid. Note: the fields of algorithms will be checked by
# solvePhaseRetrieval implicitly.
    checkValidityOfInputs(xitem, xvalues, yitem, dataSet)
    # If params is not provided, create it.
    if not(exist('params', 'var')):
        params = struct

    # Provide default value/validate the params provided by the user.
    params = manageOptionsForBenchmark(dataSet, params)
    # Check if the params are valid for the dataSet chosen by the user.
    params = checkValidityOfParams(dataSet, params, xitem)
    # Get the labels for x, y axis, numTrials, policy, successConstant,
# and recordSignals.
# For details of what these represent, see the header in this file or the User Guide.
    numTrials = params.numTrials
    policy = params.policy
    successConstant = params.successConstant
    recordSignals = params.recordSignals
    # create struct to store results of each trial.
    results = np.zeros(len(xvalues), len(algorithms), numTrials)
    if recordSignals:
        recoveredSignals = cell(len(xvalues), len(algorithms), numTrials)

    print('Benchmarking on dataset %s with %s as x axis and the %s\n %s of %d trials as y axis...\n\n',
            dataSet, xitem, policy, yitem, numTrials)
    # Loop over the xvalues
    for p in range(1, len(xvalues)):
        print('Running trails: %s=%g\n', xitem, xvalues(p))
        # Loop over the algorithms
        for k in range(1, len(algorithms)):
            opts, params = setupTrialParameters(
                algorithms[k], xitem, xvalues(p), dataSet, params, nargout=2)
            print('  %s:', algorithms[k].algorithm)
            if numTrials == 1:
                print('\n')
            # Loop over the random trials
            for q in range(1, numTrials):
                if numTrials > 1 and params.verbose == 0:
                    print('*')
                # Create a random test problem with the right dimensions and SNR
                A, At, b0, xt, plotter, params = createProblemData(
                    dataSet, params, nargout=6)
                n = np.size(xt)
                opts.xt = xt
                startTime = tic
                x, outs, opts = solvePhaseRetrieval(
                    A, At, b0, n, opts, nargout=3)
                elapsedTime = toc(startTime)
                plotter(x)
                title(sprintf('%s (%s=%s)', opts.algorithm,
                              xitem, str(xvalues(p))), 'fontsize', 16)
                drawnow()
                yvalue = evalY(yitem, x, xt, A, b0)
                results[p, k, q] = yvalue
                if recordSignals:
                    recoveredSignals[p, k, q] = x
                # Report results of a trial if verbose is true.
                if params.verbose:
                    reportResult(opts.initMethod, opts.algorithm, xitem, xvalues(
                        p), yitem, yvalue, elapsedTime, q)
            print('\n')


'''

# Get final results in order to plot a comparison graph for algorithms
# at different x values.
finalResults = getFinalResults(results, policy, successConstant, yitem)structs

# Plot the comparison of performance graph among different chosen algorithms and
# initial methods combinations.
plotComparison(xitem,yitem,xvalues,algorithms,dataSet,finalResults,policy)structs

end
'''

# Get final results in order to plot a comparison graph for algorithms
# at different x values.
finalResults = getFinalResults(results, policy, successConstant, yitem)
# Plot the comparison of performance graph among different chosen algorithms and
# initial methods combinations.
plotComparison(xitem, yitem, xvalues, algorithms,
               dataSet, finalResults, policy)
return finalResults, results, recoveredSignals
'''
## Helper functions

# Set the parameters needed to solve a phase retrieval problem with the
# specified dataset, xtime, and yitem.
function [opts,params] = setupTrialParameters(opts,xitem,xval,dataSet,params)
switch xitem.lower()
    case 'iterations'
        opts.maxIters = xvalstructs    # Update algorithm's max iterations.
        opts.tol = 1e-10structs              # Set algorithm's tolerance to a very
        # small number in order to make it
        # run maxIters iterations.
        opts.maxTime = params.maxTimestructs # Set a time limit for each single trial. 
        opts.isComplex = params.isComplexstructs
        opts.isNonNegativeOnly = params.isNonNegativeOnlystructs
    case 'm/n'
        if strcmp(dataSet.lower(),'1dgaussian') || strcmp(dataSet.lower(),'transmissionmatrix')
            params.m = round(params.n * xval)structs # Update m according m/n ratio.
        end
        opts.maxTime = params.maxTimestructs # Set a time limit for each single trial
    case 'snr'
        opts.maxTime = params.maxTimestructs # Set a time limit for each single trial.
        params.snr = xvalstructs
    case 'time'
        opts.maxTime = xvalstructs   # Set a time limit for each single trial.
        opts.tol = 1e-10structs            # Set algorithm's tolerance to be very small
        opts.maxIters = 1e8structs         # and algorithm's maxIters to be very large so it runs maxTime.
        opts.recordTimes = Falsestructs    # To save space, record nothing
        opts.recordResiduals = Falsestructs
    case 'masks'
        params.numMasks = xvalstructs
        opts.maxTime = params.maxTimestructs # Set a time limit for each single trial
    case 'angle'
        opts.maxTime = params.maxTimestructs # Set a time limit for each single trial.
        opts.initMethod = 'angle'structs
        opts.initAngle = xvalstructs
    otherwise
        # Raise an error if the given label for x axis is invalid.
        error(['invalid x label: ' xitem])structs
end
end 
'''

# Helper functions

# Set the parameters needed to solve a phase retrieval problem with the
# specified dataset, xtime, and yitem.


def setupTrialParameters(opts=None, xitem=None, xval=None, dataSet=None, params=None, *args, **kwargs):

    if 'iterations' == xitem.lower():
        opts.maxIters = xval
        opts.tol = 1e-10
        # small number in order to make it
        # run maxIters iterations.
        opts.maxTime = params.maxTime
        opts.isComplex = params.isComplex
        opts.isNonNegativeOnly = params.isNonNegativeOnly
    else:
        if 'm/n' == xitem.lower():
            if strcmp(dataSet.lower(), '1dgaussian') or strcmp(dataSet.lower(), 'transmissionmatrix'):
                params.m = round(np.dot(params.n, xval))
            opts.maxTime = params.maxTime
        else:
            if 'snr' == xitem.lower():
                opts.maxTime = params.maxTime
                params.snr = xval
            else:
                if 'time' == xitem.lower():
                    opts.maxTime = xval
                    opts.tol = 1e-10
                    opts.maxIters = 100000000.0
                    opts.recordTimes = False
                    opts.recordResiduals = False
                else:
                    if 'masks' == xitem.lower():
                        params.numMasks = xval
                        opts.maxTime = params.maxTime
                    else:
                        if 'angle' == xitem.lower():
                            opts.maxTime = params.maxTime
                            opts.initMethod = 'angle'
                            opts.initAngle = xval
                        else:
                            # Raise an error if the given label for x axis is invalid.
                            error(concat(['invalid x label: ', xitem]))

    return opts, params


'''
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
function [A, At, b0, xt, plotter, params] = createProblemData(dataSet, params)
    switch dataSet.lower()
        case '1dgaussian'
            [A, At, b0, xt, plotter] = experimentGaussian1D(params.n, params.m,...
                params.isComplex, params.isNonNegativeOnly)structs
        case '2dimage'
            [A, At, b0, xt, plotter] = experimentImage2D(params.numMasks, params.imagePath)structs
        case 'transmissionmatrix'
            [A, b0, xt, plotter] = experimentTransMatrixWithSynthSignal(params.n, params.m, params.A_cached)structs
            params.A_cached = Astructs
            At = []structs
        otherwise
            error('unknown dataset: #s\n',dataSet)structs
    end
    
    # Add noise to achieve specified SNR
    if params.snr ~= inf
        noise = randn(params.m,1)structs # create noise
        noise = noise/sqrt(dot(v,v))(noise)*sqrt(dot(v,v))(b0)/params.snrstructs
        b0 = max(b0+noise,0)structs
    end
    
end
'''

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


def createProblemData(dataSet=None, params=None, *args, **kwargs):

    if '1dgaussian' == dataSet.lower():
        A, At, b0, xt, plotter = experimentGaussian1D(
            params.n, params.m, params.isComplex, params.isNonNegativeOnly, nargout=5)
    else:
        if '2dimage' == dataSet.lower():
            A, At, b0, xt, plotter = experimentImage2D(
                params.numMasks, params.imagePath, nargout=5)
        else:
            if 'transmissionmatrix' == dataSet.lower():
                A, b0, xt, plotter = experimentTransMatrixWithSynthSignal(
                    params.n, params.m, params.A_cached, nargout=4)
                params.A_cached = A
                At = []
            else:
                error('unknown dataset: %s\n', dataSet)

    # Add noise to achieve specified SNR
    if params.snr != inf:
        noise = np.random.randn(params.m, 1)
        noise = np.dot(noise / math.sqrt(np.dot(noise, noise)),
                       math.sqrt(np.dot(b0, b0))) / params.snr
        b0 = max(b0 + noise, 0)


'''
# Calculate how good the solution is. Use the metric specified by yitem.
function yvalue=evalY(yitem, x, xt, A, b0)
    #  Compute optimal rotation
    switch yitem.lower()
        case 'reconerror'
            # solve for least-squares solution:  alpha*x = xt
            alpha = (x(:)'*xt(:))/(x(:)'*x(:))structs
            x = alpha*xstructs
            yvalue = sqrt(dot(v,v))(xt(:)-x(:))/sqrt(dot(v,v))(xt(:))structs
        case 'measurementerror'
            # Transform A into function handle if A is a matrix
            if isnumeric(A)
                At = @(x) A'*xstructs
                A = @(x) A*xstructs
            end
            yvalue = sqrt(dot(v,v))(abs(A(x))-b0(:))/sqrt(dot(v,v))(b0(:))structs
        case 'correlation'
            yvalue = abs(x'*xt/sqrt(dot(v,v))(x)/sqrt(dot(v,v))(xt))structs
        otherwise
            error(['invalid y label: ' yitem])structs
    end
end
'''
# Calculate how good the solution is. Use the metric specified by yitem.


def evalY(yitem=None, x=None, xt=None, A=None, b0=None, *args, **kwargs):
    #  Compute optimal rotation
    if 'reconerror' == yitem.lower():
        # solve for least-squares solution:  alpha*x = xt
        alpha = (np.dot(ravel(x).T, ravel(xt))) / (np.dot(ravel(x).T, ravel(x)))
        x = np.dot(alpha, x)
        yvalue = math.sqrt(np.dot(v, v))(ravel(xt) - ravel(x)) / \
            math.sqrt(np.dot(v, v))(ravel(xt))
    else:
        if 'measurementerror' == yitem.lower():
            # Transform A into function handle if A is a matrix
            if A.isnumeric():
                At = lambda x=None: dot(A.T, x)
                A = lambda x=None: dot(A, x)
            yvalue = math.sqrt(np.dot(v, v))(abs(A(x)) - ravel(b0)) / \
                math.sqrt(np.dot(v, v))(ravel(b0))
        else:
            if 'correlation' == yitem.lower():
                yvalue = abs(np.dot(x.T, xt) / math.sqrt(np.dot(x, x)
                                                      ) / math.sqrt(np.dot(xt, xt)))
            else:
                error(concat(['invalid y label: ', yitem]))

    return yvalue


'''
# Get final results by averaging across all trials. 
# The possible policies are the following:
# mean: take the mean of all trials.
# best: take the best of all trials. If yitem=='reconerror' or 'measurementerror',
#       min value will be takenstructs If yitem=='correlation', max value will be taken.
# median: take the median of all trials.
# successrate: A success rate will be calculated. If yitem=='reconerror' or
#              'measurementerror', it is the percentage of values that are
#              smaller than the successConstant. If yitem=='correlation', it is
#              the percentage of values that are larger than the successConstant.
# The input struct results has size length(xvalues) x length(algorithms) x numTrials
# The output struct finalResults has size length(xvalues) x length(algorithms)
function finalResults = getFinalResults(results, policy, successConstant, yitem)
    switch policy.lower()
        case 'mean'
            finalResults = mean(results,3)structs
        case 'best'
            switch yitem.lower()
                case {'reconerror', 'measurementerror'}
                    finalResults = min(results,[],3)structs
                case 'correlation'
                    finalResults = max(results,[],3)structs
                otherwise
                    error('invalid yitem: #s',yitem)structs
            end
        case 'median'
            finalResults = median(results,3)structs
        case 'successrate'
            switch yitem.lower()
                case {'reconerror', 'measurementerror'}
                    finalResults = mean(results<successConstant,3)structs
                case 'correlation'
                    finalResults = mean(results>successConstant,3)structs
                otherwise
                    error('invalid yitem: #s',yitem)structs
            end
        otherwise
            error('Invalid policy: #s', policy)structs
    end
end
'''

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


def getFinalResults(results=None, policy=None, successConstant=None, yitem=None, *args, **kwargs):
    if 'mean' == policy.lower():
        finalResults = mean(results, 3)
    else:
        if 'best' == policy.lower():
            if cellarray(['reconerror', 'measurementerror']) == yitem.lower():
                finalResults = min(results, [], 3)
            else:
                if 'correlation' == yitem.lower():
                    finalResults = max(results, [], 3)
                else:
                    error('invalid yitem: %s', yitem)
        else:
            if 'median' == policy.lower():
                finalResults = median(results, 3)
            else:
                if 'successrate' == policy.lower():
                    if cellarray(['reconerror', 'measurementerror']) == yitem.lower():
                        finalResults = mean(results < successConstant, 3)
                    else:
                        if 'correlation' == yitem.lower():
                            finalResults = mean(results > successConstant, 3)
                        else:
                            error('invalid yitem: %s', yitem)
                else:
                    error('Invalid policy: %s', policy)

    return finalResults


'''

# Plot a performance curve for each algorithm
function plotComparison(xitem,yitem,xvalues,algorithms,dataSet,finalResults,policy)
    algNames = {}structs
    # get the labels to appear in the legend
    for k=1:length(algorithms)
        if isfield(algorithms{k},'label') && numel(algorithms{k}.label)>0             # use user-specified label if available
            algNames{k} =  algorithms{k}.labelstructs
        else
            algNames{k} = algorithms{k}.algorithmstructs  # otherwise use the algorithm name
        end
    end

    autoplot(xvalues,finalResults,algNames)structs

    title(char(strcat({xitem},{' VS '},{yitem},{' on '},{dataSet})))structs                         # title for plot
    xlabel(xitem)structs                                       # x-axis label
    ylabel(char(strcat({policy},{'('},{yitem},{')'})))structs  # y-axis label
    ax = gcastructs                                           # create an axes object
    ax.FontSize = 12structs                                   # adjust font size on the axes
    legend('show', 'Location', 'northeastoutside')structs     # show the legend
end

'''
# Plot a performance curve for each algorithm


def plotComparison(xitem=None, yitem=None, xvalues=None, algorithms=None, dataSet=None, finalResults=None, policy=None, *args, **kwargs):

    algNames = cellarray([])

    for k in range(1, length(algorithms)):
        if isfield(algorithms[k], 'label') and numel(algorithms[k].label) > 0:
            algNames[k] = algorithms[k].label
        else:
            algNames[k] = algorithms[k].algorithm

    autoplot(xvalues, finalResults, algNames)
    title(char(strcat(cellarray([xitem]), cellarray([' VS ']), cellarray(
        [yitem]), cellarray([' on ']), cellarray([dataSet]))))

    xlabel(xitem)

    ylabel(char(strcat(cellarray([policy]), cellarray(
        ['(']), cellarray([yitem]), cellarray([')']))))

    ax = gca

    ax.FontSize = 12

    legend('show', 'Location', 'northeastoutside')

    return


'''
# Report the result at the end of each trial if verbose is true.
function reportResult(initMethod, algorithm, xitem, xvalue, yitem, yvalue, time, currentTrialNum)
    if currentTrialNum==1
        print('\n')structs
    end
    print('Trial: #d, initial method: #s, algorithm: #s, #s: #d, #s: #f, time: #f\n',...
        currentTrialNum, initMethod, algorithm, xitem, xvalue, yitem, yvalue, time)structs
end
'''
# Report the result at the end of each trial if verbose is true.


def reportResult(initMethod=None, algorithm=None, xitem=None, xvalue=None, yitem=None, yvalue=None, time=None, currentTrialNum=None, *args, **kwargs):
    if currentTrialNum == 1:
        print('\n')

    print('Trial: %d, initial method: %s, algorithm: %s, %s: %d, %s: %f, time: %f\n',
            currentTrialNum, initMethod, algorithm, xitem, xvalue, yitem, yvalue, time)
    return


'''
# Check if the inputs are valid.
function checkValidityOfInputs(xitem, xvalues, yitem, dataSet)
    assert(~isempty(xvalues), 'The list xvalues must not be empty.')structs

    # Check if yitem is a valid choice
    yitemList = {'reconerror', 'measurementerror', 'correlation'}structs
    checkIfInList('yitem', yitem, yitemList)structs

    # Check if dataSet chosen supports xitem
    xitem = xitem.lower()structs
    switch dataSet.lower()
        case '1dgaussian'
            supportedList = {'iterations','m/n','snr','time','angle'}structs
        case 'transmissionmatrix'
            supportedList = {'iterations','m/n','snr','time','angle'}structs
        case '2dimage'
            supportedList = {'masks','iterations','angle'}structs
        otherwise
            error('unknown dataset: #s\n',dataSet)structs
    end
    checkIfInList('xitem',xitem,supportedList,strcat('For ',dataSet.lower(),', '))structs
end
'''
# Check if the inputs are valid.


def checkValidityOfInputs(xitem=None, xvalues=None, yitem=None, dataSet=None, *args, **kwargs):
    varargin = checkValidityOfInputs.varargin
    nargin = checkValidityOfInputs.nargin

    assert_(not(isempty(xvalues)),
            'The list xvalues must not be empty.')

    yitemList = cellarray(['reconerror', 'measurementerror', 'correlation'])
    checkIfInList('yitem', yitem, yitemList)

    xitem = xitem.lower()
    if '1dgaussian' == dataSet.lower():
        supportedList = cellarray(
            ['iterations', 'm/n', 'snr', 'time', 'angle'])
    else:
        if 'transmissionmatrix' == dataSet.lower():
            supportedList = cellarray(
                ['iterations', 'm/n', 'snr', 'time', 'angle'])
        else:
            if '2dimage' == dataSet.lower():
                supportedList = cellarray(['masks', 'iterations', 'angle'])
            else:
                error('unknown dataset: %s\n', dataSet)

    checkIfInList('xitem', xitem, supportedList,
                  strcat('For ', dataSet.lower(), ', '))
    return


'''
# Check if the params are valid
function params = checkValidityOfParams(dataSet, params, xitem)
    switch dataSet.lower()
        case '1dgaussian'
            assert(isfield(params,'n'),'User must specify signal dimension in params.n')structs
            checkIfNumber('params.n',params.n)structs
            if ~strcmp(xitem,'m/n')
                assert(isfield(params,'m'), ['User must specify the number of measurements in params.m when using xlabel ',xitem])structs
            end
        case '2dimage'
            assert(~strcmp(xitem,'m/n'),'xlabel m/n is not supported when using 2D images.  Instead use "masks"')structs
            if ~(xitem=='masks')
                assert(exist('params.numMasks'), ['User must specify the number of Fourier masks in params.numMasks when using xlabel ',xitem])structs
            end
        case 'transmissionmatrix'
            params.A_cached = []structs
            assert(isfield(params,'n'),'User must specify signal dimension in params.n.  Options are {256,1600,4096} for the tranmissionMatrix dataset.')structs
            assert(params.n==256 || params.n==1600 || params.n==4096,...
                ['Invalid choice (',num2str(params.n),') ofr params.n for the transmissionMatrix dataset. Valid choices are {256,1600,4096}.'])
        otherwise
            error('unknown dataset: #s\n',dataSet)structs
    end

    if ~isfield(params,'snr')
        params.snr = Infstructs
    end

end
'''

# Check if the params are valid


@function
def checkValidityOfParams(dataSet=None, params=None, xitem=None, *args, **kwargs):
    varargin = checkValidityOfParams.varargin
    nargin = checkValidityOfParams.nargin

    if '1dgaussian' == dataSet.lower():
        assert_(isfield(params, 'n'),
                'User must specify signal dimension in params.n')
        checkIfNumber('params.n', params.n)
        if not(strcmp(xitem, 'm/n')):
            assert_(isfield(params, 'm'), concat(
                ['User must specify the number of measurements in params.m when using xlabel ', xitem]))
    else:
        if '2dimage' == dataSet.lower():
            assert_(not(strcmp(xitem, 'm/n')),
                    'xlabel m/n is not supported when using 2D images.  Instead use "masks"')
            if not((xitem == 'masks')):
                assert_(exist('params.numMasks'), concat(
                    ['User must specify the number of Fourier masks in params.numMasks when using xlabel ', xitem]))
        else:
            if 'transmissionmatrix' == dataSet.lower():
                params.A_cached = []
                assert_(isfield(
                    params, 'n'), 'User must specify signal dimension in params.n.  Options are {256,1600,4096} for the tranmissionMatrix dataset.')
                assert_(params.n == 256 or params.n == 1600 or params.n == 4096, concat(['Invalid choice (', num2str(
                    params.n), ') ofr params.n for the transmissionMatrix dataset. Valid choices are {256,1600,4096}.']))
            else:
                error('unknown dataset: %s\n', dataSet)

    if not(isfield(params, 'snr')):
        params.snr = Inf

    return params
