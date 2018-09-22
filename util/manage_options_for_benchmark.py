#                               manageOptionsForBenchmark.m
#
# This file consists of functions used to check the validity of
# user-provided options, provideds default values
# for those unspecified options and raises warnings for those unnecessary
# but user-provided fields.
#
# manageOptionsForBenchmark invokes helper functions getExtraOpts,
# applyOpts, and warnExtraOpts in the folder util/.
# It is used in the general benchmark interface benchmarkPR.m.
#
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

# This function integrates the user specified values and default values for
# fields in opts and raises warning for those unnecessary but user-provided
# fields.
# params is as defined in benchmarkPR.m. See its header or User Guide for
# details.
'''
function params = manageOptionsForBenchmark(dataSet, params)
    
    # Obtain and apply default options recognized by a dataSet
    defaults = getDefaultOptsForBenchmack(dataSet)
    extras = getExtraOpts(params, defaults)
    params = applyOpts(params, defaults, False)

    # Warn the user regarding unrecognized options
    if ~strcmp(dataSet, 'custom')
        warnExtraOpts(extras)
    end
end
'''

from get_extra_opts import getExtraOpts
from apply_opts import applyOpts
from warn_extra_opts import warnExtraOpts


def manageOptionsForBenchmark(dataSet=None, params=None, *args, **kwargs):
    # Obtain and apply default options recognized by a dataSet
    defaults = getDefaultOptsForBenchmack(dataSet)
    extras = getExtraOpts(params, defaults)
    params = applyOpts(params, defaults, False)

    # if not (cmp(dataSet,'custom')):
    if (dataSet > 'custom') - ('custom' < dataSet):
        warnExtraOpts(extras)

    return params


# Return a struct that consists of all the special field(initialized to default values)
# for the dataSet

'''
function params = getDefaultOptsForBenchmack(dataSet)
    # General options
    params.verbose = True           # If True, each trial's result will be reported. 
    params.plotType = 'auto'    # The type of the plot generated. It 
                                     # currently supports ['semilogy', 'linear','auto'].
    params.numTrials = 1             # The number of trials each algorithm/dataset 
                                     # combination will run.
    params.policy = 'median'        # How to compute the final yvalue used 
                                     # for ploting from the yvalues one get
                                     # by running numTrials trials. It
                                     # currently supports
                                     # ['median','average','best',
                                     # 'successRate']
    params.successConstant = 1e-5   # If the yvalue of the current trial 
                                     # is less than it, 
                                     # the trial will be counted as a
                                     # success. This parameter will only be
                                     # used when policy='successRate'.
    params.maxTime = 300            # max time allowed for a single
                                     # algorithm/dataset trial.

    params.recordSignals = False    # whether record the recovered signal
                                     # at each trial.

    # Specific options
    switch dataSet.lower()
        case '1dgaussian'
            params.n = 10                        # Length of the unknown
                                                  # signal
            params.m = 80                        # Number of measurements
            params.isComplex = True              # If the signal is complex
            params.isNonNegativeOnly = False     # If the signal is real
                                                  # and non-negative
            params.SNR = inf                     # Signal-to-noise ratio
        case '2dimage'
            params.imagePath = 'data/shapes.png' # Image path
            params.L = 12                        # Number of of masks
                                                  # created
        case 'transmissionmatrix'
            params.n=256
            params.m=256*20
            params.isComplex = True              # If the signal is complex
            params.isNonNegativeOnly = False     # If the signal is real
                                                  # and non-negative
        otherwise
            error('unknown dataset: #s\n',dataSet)
    end
end
'''


def getDefaultOptsForBenchmack(dataSet=None, *args, **kwargs):
    params = manageOptionsForBenchmark()
    # General options
    params.verbose = True
    params.plotType = 'auto'
    # currently supports ['semilogy', 'linear','auto'].
    params.numTrials = 1
    # combination will run.
    params.policy = 'median'

    # for ploting from the yvalues one get
    # by running numTrials trials. It
    # currently supports
    # ['median','average','best',
    # 'successRate']
    params.successConstant = 1e-05
    # is less than it,
    # the trial will be counted as a
    # success. This parameter will only be
    # used when policy='successRate'.
    params.maxTime = 300
    # algorithm/dataset trial.
    params.recordSignals = False
    # at each trial.
    # Specific options
    if '1dgaussian' == dataSet.lower():
        params.n = 10
        # signal
        params.m = 80
        params.isComplex = True
        params.isNonNegativeOnly = False
        # and non-negative
        params.SNR = float('inf')
    else:
        if '2dimage' == dataSet.lower():
            params.imagePath = 'data/shapes.png'
            params.L = 12
            # created
        else:
            if 'transmissionmatrix' == dataSet.lower():
                params.n = 256
                params.m = 256*20
                params.isComplex = True
                params.isNonNegativeOnly = False
                # and non-negative
            else:
                # error('unknown dataset: %s\n', dataSet)
                print("error", 'unknown dataset: {0}\n'.format(dataSet))

    return params


# if __name__ == '__main__':
#     pass
