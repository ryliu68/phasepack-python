#                               manageOptions.m
#
# This file consists of functions used to check the validity of
# user-provided options, provideds default values
# for those unspecified options and raises warnings for those unnecessary
# but user-provided fields.
#
# manageOptions invokes helper functions getExtraOpts,applyOpts, and
# warnExtraOpts in the folder util/.
# It is used in the general solve PR interface solvePhaseRetrieval.m.
#
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

# This function integrates the user specified values and default values for
# fields in opts and raises warning for those unnecessary but user-provided
# fields.
# opts is as defined in solvePhaseRetrieval.m. See its header or User Guide
# for details.
'''
function opts = manageOptions(opts)
    # Set default algorithm if not specified
    if ~isfield(opts, 'algorithm')
        opts.algorithm = 'gerchbergsaxton'
    end

    # Obtain and apply default options that relevant to the specified
    # algorithm
    defaults = getDefaultOpts(opts.algorithm)
    extras = getExtraOpts(opts, defaults)
    opts = applyOpts(opts, defaults, False)

    # Warn the user regarding unrecognized options
    if ~strcmp(opts.algorithm, 'custom')
        warnExtraOpts(extras)
    end
end
'''

from get_extra_opts import getExtraOpts
from apply_opts import applyOpts
from warn_extra_opts import warnExtraOpts


def manageOptions(opts=None, *args, **kwargs):

    # Set default algorithm if not specified
    if not(opts.hasattr('algorithm')):
        opts.algorithm = 'gerchbergsaxton'

    # Obtain and apply default options that relevant to the specified
    # algorithm
    defaults = getDefaultOpts(opts.algorithm)
    extras = getExtraOpts(opts, defaults)
    opts = applyOpts(opts, defaults, False)

    if not(opts.algorithm == 'custom'):
        warnExtraOpts(extras)

    return opts


# This function returns a struct containing the default options for
# the specified solver algorithm.
'''
function opts = getDefaultOpts(algorithm)
    opts = struct

    # Following options are relevant to every algorithm

    opts.algorithm = algorithm
    # Initialization method
    opts.initMethod = 'optimal'
    # Whether the measurements consist entirely of real numbers.
    opts.isComplex = True
    # Whether the signal is non-negative only.
    opts.isNonNegativeOnly = False
    # Maximum number of iterations
    opts.maxIters = 10000
    # The maximum time the solver can run (unit: second)
    # Note: since elapsed time will be checked at the end of each iteration,
    # the real time the solver takes is the time of the iteration it
    # goes beyond this maxTime.
    opts.maxTime = 300
    # Stopping tolerance
    opts.tol = 1.0e-4
    # Choose from [0, 1, 2]. If 0, print out nothing. If 1, print out
    # status information in the end.
    # If 2, print print out status information every round.
    opts.verbose = 0
    # If the solver record time at each iteration
    opts.recordTimes = True
    # If the solver compute and record measurement errors
    # i.e. norm(abs(A*x-b0))/norm(b0) at each iteration
    opts.recordMeasurementErrors = False
    # If the solver record reconstruction errors i.e. norm(xt-x)/norm(x) at
    # each iteration
    opts.recordReconErrors = False
    # If the solver record residuals (metric varies across solvers) at each
    # iteration
    opts.recordResiduals = True
    # Can be used to choose a label for the algorithm to show up in the
    # legend of a plot.  This is used when plotting results of benchmarks.
    # The algorithm name is used by default if no label is specified.
    opts.label = []

    # Following three options are unused by default
    # The True signal. If it is provided, reconstruction error will be
    # computed and used for stopping condition
    opts.xt = []
    # Custom algorithm provided by user
    opts.customAlgorithm = []
    # Custom initializer provided by user
    opts.customx0 = []
    # When the angle initializer is used, you must specify the angle
    # between the True signal and the initializer
    opts.initAngle = []

    # Remaining options are algorithm-specific
    switch algorithm.lower()
        case 'custom'
            # No extra options
        case 'amplitudeflow'
            # Specifies how search direction for line search is chosen upon
            # each iteration
            opts.searchMethod = 'steepestDescent'
            # Specifies how beta value is chosen (only used when search
            #  method is NCG)
            opts.betaChoice = [] # Unused by default
        case 'coordinatedescent'
            opts.indexChoice = 'greedy'  # the rule for picking up index,
                                          # choose from ['cyclic','random'
                                          # ,'greedy'].
        case 'fienup'
            opts.FienupTuning = 0.5      # Tunning parameter for
                                          # Gerchberg-Saxton algorithm.
                                          # It influences the update of the
                                          # fourier domain value at each
                                          # iteration.
            opts.maxInnerIters = 10     # The max number of iterations
                                          # the inner-loop solver
                                          # will have.
        case 'gerchbergsaxton'
            opts.maxInnerIters = 10     # The max number of iterations
                                          # the inner-loop solver
                                          # will have.
        case 'kaczmarz'
            opts.indexChoice = 'cyclic'  # the rule for picking up index,
                                          # choose from ['cyclic','random'].
        case 'phasemax'

        case 'phaselamp'

        case 'phaselift'
            # This controls the weight of trace(X), where X=xx'
            # in the objective function (see phaseLift paper for details)
            opts.regularizationPara = 0.1
        case 'raf'
            # The maximum number of iterations that are allowed to occurr
            # between reweights of objective function
            opts.reweightPeriod = 20
            # Specifies how search direction for line search is chosen upon
            # each iteration
            opts.searchMethod = 'steepestDescent'
            # Specifies how beta value is chosen (only used when search
            # method is NCG)
            opts.betaChoice = 'HS' # Used only for NCG solver
        case 'rwf'
            # Constant used to reweight objective function (see RWF paper
            # for details)
            opts.eta = 0.9
            # The maximum number of iterations that are allowed to occurr
            # between reweights of objective function
            opts.reweightPeriod = 20
            # Specifies how search direction for line search is chosen upon
            # each iteration
            opts.searchMethod = 'steepestDescent'
            # Specifies how beta value is chosen (only used when search
            # method is NCG)
            opts.betaChoice = 'HS' # Used only for NCG solver
        case 'sketchycgm'
            opts.rank = 1                # rank parameter. For details see
                                          # Algorithm1 in the sketchyCGM
                                          # paper.
            opts.eta = 1                 # stepsize parameter
        case 'taf'
            # Constant used to truncate objective function (see paper for
            # details)
            opts.gamma = 0.7
            # The maximum number of iterations that are allowed to occurr
            # between truncations of objective function
            opts.truncationPeriod = 20
            # Specifies how search direction for line search is chosen upon
            # each iteration
            opts.searchMethod = 'steepestDescent'
            # Specifies how beta value is chosen (only used when search
            # method is NCG)
            opts.betaChoice = 'HS' # Used only for NCG solver
        case 'twf'
            # The maximum number of iterations that are allowed to occur
            # between truncations of objective function
            opts.truncationPeriod = 20
            # Specifies how search direction for line search is chosen upon
            # each iteration
            opts.searchMethod = 'steepestDescent'
            # Specifies how beta value is chosen (only used when search
            # method is NCG)
            opts.betaChoice = 'HS' # Used only for NCG solver
            # Truncation parameters. These default values are defined as
            # in the proposing paper for the case where line search is
            # used.
            opts.alpha_lb    = 0.1
            opts.alpha_ub    = 5
            opts.alpha_h     = 6
        case 'wirtflow'
            # Specifies how search direction for line search is chosen upon each
            # iteration
            opts.searchMethod = 'steepestDescent'
            # Specifies how beta value is chosen (only used when search method
            # is NCG)
            opts.betaChoice = 'HS' # Used only for NCG solver
        otherwise
            error('Invalid algorithm "#s" provided', algorithm)
    end
end
'''
import struct


def getDefaultOpts(algorithm=None, *args, **kwargs):
    opts = struct
    opts.algorithm = algorithm
    opts.initMethod = 'optimal'
    opts.isComplex = True
    opts.isNonNegativeOnly = False
    opts.maxIters = 10000

    # Note: since elapsed time will be checked at the end of each iteration,
    # the real time the solver takes is the time of the iteration it
    # goes beyond this maxTime.
    opts.maxTime = 300
    opts.tol = 0.0001

    # status information in the end.
    # If 2, print print out status information every round.
    opts.verbose = 0
    opts.recordTimes = True

    # i.e. norm(abs(A*x-b0))/norm(b0) at each iteration
    opts.recordMeasurementErrors = False
    # each iteration
    opts.recordReconErrors = False
    # iteration
    opts.recordResiduals = True
    # legend of a plot.  This is used when plotting results of benchmarks.
    # The algorithm name is used by default if no label is specified.
    opts.label = []
    # The True signal. If it is provided, reconstruction error will be
    # computed and used for stopping condition
    opts.xt = []
    opts.customAlgorithm = []
    opts.customx0 = []

    # between the True signal and the initializer
    opts.initAngle = []

    if 'custom' == algorithm.lower():
        # No extra options
        pass
    else:
        if 'amplitudeflow' == algorithm.lower():
            # Specifies how search direction for line search is chosen upon
            # each iteration
            opts.searchMethod = 'steepestDescent'
            #  method is NCG)
            opts.betaChoice = []
        else:
            if 'coordinatedescent' == algorithm.lower():
                opts.indexChoice = 'greedy'
                # choose from ['cyclic','random' # ,'greedy'].
            else:
                if 'fienup' == algorithm.lower():
                    opts.FienupTuning = 0.5
                    # Gerchberg-Saxton algorithm.
                    # It influences the update of the
                    # fourier domain value at each
                    # iteration.
                    opts.maxInnerIters = 10
                    # the inner-loop solver
                    # will have.
                else:
                    if 'gerchbergsaxton' == algorithm.lower():
                        opts.maxInnerIters = 10
                        # the inner-loop solver
                        # will have.
                    else:
                        if 'kaczmarz' == algorithm.lower():
                            opts.indexChoice = 'cyclic'
                            # choose from ['cyclic','random'].
                        else:
                            if 'phasemax' == algorithm.lower():
                                pass
                            else:
                                if 'phaselamp' == algorithm.lower():
                                    pass
                                else:
                                    if 'phaselift' == algorithm.lower():
                                        # This controls the weight of trace(X), where X=xx'
                                        # in the objective function (see phaseLift paper for details)
                                        opts.regularizationPara = 0.1
                                    else:
                                        if 'raf' == algorithm.lower():
                                            # The maximum number of iterations that are allowed to occurr
                                            # between reweights of objective function
                                            opts.reweightPeriod = 20
                                            # each iteration
                                            opts.searchMethod = 'steepestDescent'
                                            # method is NCG)
                                            opts.betaChoice = 'HS'
                                        else:
                                            if 'rwf' == algorithm.lower():
                                                # Constant used to reweight objective function (see RWF paper
                                                # for details)
                                                opts.eta = 0.9
                                                # between reweights of objective function
                                                opts.reweightPeriod = 20
                                                # each iteration
                                                opts.searchMethod = 'steepestDescent'
                                                # method is NCG)
                                                opts.betaChoice = 'HS'
                                            else:
                                                if 'sketchycgm' == algorithm.lower():
                                                    opts.rank = 1
                                                    # Algorithm1 in the sketchyCGM
                                          # paper.
                                                    opts.eta = 1
                                                else:
                                                    if 'taf' == algorithm.lower():
                                                        # Constant used to truncate objective function (see paper for
                                                        # details)
                                                        opts.gamma = 0.7
                                                        # between truncations of objective function
                                                        opts.truncationPeriod = 20
                                                        # each iteration
                                                        opts.searchMethod = 'steepestDescent'
                                                        # method is NCG)
                                                        opts.betaChoice = 'HS'
                                                    else:
                                                        if 'twf' == algorithm.lower():
                                                            # The maximum number of iterations that are allowed to occur
                                                            # between truncations of objective function
                                                            opts.truncationPeriod = 20
                                                            # each iteration
                                                            opts.searchMethod = 'steepestDescent'
                                                            # method is NCG)
                                                            opts.betaChoice = 'HS'
                                                            # Truncation parameters. These default values are defined as
            # in the proposing paper for the case where line search is
            # used.
                                                            opts.alpha_lb = 0.1
                                                            opts.alpha_ub = 5
                                                            opts.alpha_h = 6
                                                        else:
                                                            if 'wirtflow' == algorithm.lower():
                                                                # Specifies how search direction for line search is chosen upon each
                                                                # iteration
                                                                opts.searchMethod = 'steepestDescent'
                                                                # is NCG)
                                                                opts.betaChoice = 'HS'
                                                            else:
                                                                # error('Invalid algorithm "%s" provided', algorithm)
                                                                print(
                                                                    "error", 'Invalid algorithm "{0}" provided'.format(algorithm))

    return opts
