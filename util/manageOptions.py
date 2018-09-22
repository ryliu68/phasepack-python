# Generated with SMOP  0.41
from libsmop import *
# .\manageOptions.m

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
    
@function
def manageOptions(opts=None,*args,**kwargs):
    varargin = manageOptions.varargin
    nargin = manageOptions.nargin

    # Set default algorithm if not specified
    if logical_not(isfield(opts,'algorithm')):
        opts.algorithm = copy('gerchbergsaxton')
# .\manageOptions.m:25
    
    
    # Obtain and apply default options that relevant to the specified
    # algorithm
    defaults=getDefaultOpts(opts.algorithm)
# .\manageOptions.m:30
    extras=getExtraOpts(opts,defaults)
# .\manageOptions.m:31
    opts=applyOpts(opts,defaults,false)
# .\manageOptions.m:32
    
    if logical_not(strcmp(opts.algorithm,'custom')):
        warnExtraOpts(extras)
    
    return opts
    
if __name__ == '__main__':
    pass
    
    # This function returns a struct containing the default options for 
# the specified solver algorithm.
    
@function
def getDefaultOpts(algorithm=None,*args,**kwargs):
    varargin = getDefaultOpts.varargin
    nargin = getDefaultOpts.nargin

    opts=copy(struct)
# .\manageOptions.m:43
    
    
    opts.algorithm = copy(algorithm)
# .\manageOptions.m:47
    
    opts.initMethod = copy('optimal')
# .\manageOptions.m:49
    
    opts.isComplex = copy(true)
# .\manageOptions.m:51
    
    opts.isNonNegativeOnly = copy(false)
# .\manageOptions.m:53
    
    opts.maxIters = copy(10000)
# .\manageOptions.m:55
    
    # Note: since elapsed time will be checked at the end of each iteration,
    # the real time the solver takes is the time of the iteration it
    # goes beyond this maxTime.
    opts.maxTime = copy(300)
# .\manageOptions.m:60
    
    opts.tol = copy(0.0001)
# .\manageOptions.m:62
    
    # status information in the end.
    # If 2, print print out status information every round.
    opts.verbose = copy(0)
# .\manageOptions.m:66
    
    opts.recordTimes = copy(true)
# .\manageOptions.m:68
    
    # i.e. norm(abs(A*x-b0))/norm(b0) at each iteration
    opts.recordMeasurementErrors = copy(false)
# .\manageOptions.m:71
    
    # each iteration
    opts.recordReconErrors = copy(false)
# .\manageOptions.m:74
    
    # iteration
    opts.recordResiduals = copy(true)
# .\manageOptions.m:77
    
    # legend of a plot.  This is used when plotting results of benchmarks.
    # The algorithm name is used by default if no label is specified.
    opts.label = copy([])
# .\manageOptions.m:81
    
    # The true signal. If it is provided, reconstruction error will be
    # computed and used for stopping condition
    opts.xt = copy([])
# .\manageOptions.m:86
    
    opts.customAlgorithm = copy([])
# .\manageOptions.m:88
    
    opts.customx0 = copy([])
# .\manageOptions.m:90
    
    # between the true signal and the initializer
    opts.initAngle = copy([])
# .\manageOptions.m:93
    
    if 'custom' == lower(algorithm):
        # No extra options
        pass
    else:
        if 'amplitudeflow' == lower(algorithm):
            # Specifies how search direction for line search is chosen upon
            # each iteration
            opts.searchMethod = copy('steepestDescent')
# .\manageOptions.m:102
            #  method is NCG)
            opts.betaChoice = copy([])
# .\manageOptions.m:105
        else:
            if 'coordinatedescent' == lower(algorithm):
                opts.indexChoice = copy('greedy')
# .\manageOptions.m:107
                # choose from ['cyclic','random'
                                          # ,'greedy'].
            else:
                if 'fienup' == lower(algorithm):
                    opts.FienupTuning = copy(0.5)
# .\manageOptions.m:111
                    # Gerchberg-Saxton algorithm.
                                          # It influences the update of the 
                                          # fourier domain value at each
                                          # iteration.
                    opts.maxInnerIters = copy(10)
# .\manageOptions.m:116
                    # the inner-loop solver 
                                          # will have.
                else:
                    if 'gerchbergsaxton' == lower(algorithm):
                        opts.maxInnerIters = copy(10)
# .\manageOptions.m:120
                        # the inner-loop solver
                                          # will have.
                    else:
                        if 'kaczmarz' == lower(algorithm):
                            opts.indexChoice = copy('cyclic')
# .\manageOptions.m:124
                            # choose from ['cyclic','random'].
                        else:
                            if 'phasemax' == lower(algorithm):
                                pass
                            else:
                                if 'phaselamp' == lower(algorithm):
                                    pass
                                else:
                                    if 'phaselift' == lower(algorithm):
                                        # This controls the weight of trace(X), where X=xx'  
            # in the objective function (see phaseLift paper for details)
                                        opts.regularizationPara = copy(0.1)
# .\manageOptions.m:133
                                    else:
                                        if 'raf' == lower(algorithm):
                                            # The maximum number of iterations that are allowed to occurr
            # between reweights of objective function
                                            opts.reweightPeriod = copy(20)
# .\manageOptions.m:137
                                            # each iteration
                                            opts.searchMethod = copy('steepestDescent')
# .\manageOptions.m:140
                                            # method is NCG)
                                            opts.betaChoice = copy('HS')
# .\manageOptions.m:143
                                        else:
                                            if 'rwf' == lower(algorithm):
                                                # Constant used to reweight objective function (see RWF paper 
            # for details)
                                                opts.eta = copy(0.9)
# .\manageOptions.m:147
                                                # between reweights of objective function
                                                opts.reweightPeriod = copy(20)
# .\manageOptions.m:150
                                                # each iteration
                                                opts.searchMethod = copy('steepestDescent')
# .\manageOptions.m:153
                                                # method is NCG)
                                                opts.betaChoice = copy('HS')
# .\manageOptions.m:156
                                            else:
                                                if 'sketchycgm' == lower(algorithm):
                                                    opts.rank = copy(1)
# .\manageOptions.m:158
                                                    # Algorithm1 in the sketchyCGM
                                          # paper.
                                                    opts.eta = copy(1)
# .\manageOptions.m:161
                                                else:
                                                    if 'taf' == lower(algorithm):
                                                        # Constant used to truncate objective function (see paper for
            # details)
                                                        opts.gamma = copy(0.7)
# .\manageOptions.m:165
                                                        # between truncations of objective function
                                                        opts.truncationPeriod = copy(20)
# .\manageOptions.m:168
                                                        # each iteration
                                                        opts.searchMethod = copy('steepestDescent')
# .\manageOptions.m:171
                                                        # method is NCG)
                                                        opts.betaChoice = copy('HS')
# .\manageOptions.m:174
                                                    else:
                                                        if 'twf' == lower(algorithm):
                                                            # The maximum number of iterations that are allowed to occur
            # between truncations of objective function
                                                            opts.truncationPeriod = copy(20)
# .\manageOptions.m:178
                                                            # each iteration
                                                            opts.searchMethod = copy('steepestDescent')
# .\manageOptions.m:181
                                                            # method is NCG)
                                                            opts.betaChoice = copy('HS')
# .\manageOptions.m:184
                                                            # Truncation parameters. These default values are defined as 
            # in the proposing paper for the case where line search is
            # used.
                                                            opts.alpha_lb = copy(0.1)
# .\manageOptions.m:188
                                                            opts.alpha_ub = copy(5)
# .\manageOptions.m:189
                                                            opts.alpha_h = copy(6)
# .\manageOptions.m:190
                                                        else:
                                                            if 'wirtflow' == lower(algorithm):
                                                                # Specifies how search direction for line search is chosen upon each
            # iteration
                                                                opts.searchMethod = copy('steepestDescent')
# .\manageOptions.m:194
                                                                # is NCG)
                                                                opts.betaChoice = copy('HS')
# .\manageOptions.m:197
                                                            else:
                                                                error('Invalid algorithm "%s" provided',algorithm)
    
    return opts
    
if __name__ == '__main__':
    pass
    