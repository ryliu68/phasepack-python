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

# Main benchmark interface
# from solvers.s solvePhaseRetrieval
from util.manage_options_for_benchmark import manageOptionsForBenchmark
def benchmarkSynthetic(xitem=None, xvalues=None, yitem=None, algorithms=None, dataSet=None, params=None, *args, **kwargs):
    # Check if the inputs are valid. Note: the fields of algorithms will be checked by
# solvePhaseRetrieval implicitly.
    checkValidityOfInputs(xitem, xvalues, yitem, dataSet)
    # If params is not provided, create it.
    if not params:
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


# Get final results in order to plot a comparison graph for algorithms
# at different x values.
finalResults = getFinalResults(results, policy, successConstant, yitem)
# Plot the comparison of performance graph among different chosen algorithms and
# initial methods combinations.
plotComparison(xitem, yitem, xvalues, algorithms,
               dataSet, finalResults, policy)
return finalResults, results, recoveredSignals


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

from numpy.linalg import norm as norm
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
        noise = noise/norm(noise)*norm(b0)/params.snr
        b0 = max(b0 + noise, 0)

# Calculate how good the solution is. Use the metric specified by yitem.


def evalY(yitem=None, x=None, xt=None, A=None, b0=None, *args, **kwargs):
    #  Compute optimal rotation
    if 'reconerror' == yitem.lower():
        # solve for least-squares solution:  alpha*x = xt
        alpha = (np.dot(ravel(x).T, ravel(xt))) / (np.dot(ravel(x).T, ravel(x)))
        x = np.dot(alpha, x)
        yvalue = norm((ravel(xt) - ravel(x))) / norm(ravel(xt))
    else:
        if 'measurementerror' == yitem.lower():
            # Transform A into function handle if A is a matrix
            if A.isnumeric():
                At = lambda x=None: dot(A.T, x)
                A = lambda x=None: dot(A, x)
            yvalue = norm(abs(A(x)) - ravel(b0)) / norm(ravel(b0))
        else:
            if 'correlation' == yitem.lower():
                yvalue = abs(np.dot(x.T, xt) / norm(x) / norm(xt))
            else:
                error(concat(['invalid y label: ', yitem]))

    return yvalue


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

# Report the result at the end of each trial if verbose is true.


def reportResult(initMethod=None, algorithm=None, xitem=None, xvalue=None, yitem=None, yvalue=None, time=None, currentTrialNum=None, *args, **kwargs):
    if currentTrialNum == 1:
        print('\n')

    print('Trial: %d, initial method: %s, algorithm: %s, %s: %d, %s: %f, time: %f\n',
            currentTrialNum, initMethod, algorithm, xitem, xvalue, yitem, yvalue, time)
    return

# Check if the inputs are valid.


def checkValidityOfInputs(xitem=None, xvalues=None, yitem=None, dataSet=None, *args, **kwargs):
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


# Check if the params are valid


@function
def checkValidityOfParams(dataSet=None, params=None, xitem=None, *args, **kwargs):
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
