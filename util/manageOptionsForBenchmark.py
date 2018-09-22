# Generated with SMOP  0.41
from libsmop import *
# .\manageOptionsForBenchmark.m

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
    
    # PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    # This function integrates the user specified values and default values for 
# fields in opts and raises warning for those unnecessary but user-provided
# fields. 
# params is as defined in benchmarkPR.m. See its header or User Guide for
# details.
    
@function
def manageOptionsForBenchmark(dataSet=None,params=None,*args,**kwargs):
    varargin = manageOptionsForBenchmark.varargin
    nargin = manageOptionsForBenchmark.nargin

    
    # Obtain and apply default options recognized by a dataSet
    defaults=getDefaultOptsForBenchmack(dataSet)
# .\manageOptionsForBenchmark.m:25
    extras=getExtraOpts(params,defaults)
# .\manageOptionsForBenchmark.m:26
    params=applyOpts(params,defaults,false)
# .\manageOptionsForBenchmark.m:27
    
    if logical_not(strcmp(dataSet,'custom')):
        warnExtraOpts(extras)
    
    return params
    
if __name__ == '__main__':
    pass
    
    # Return a struct that consists of all the special field(initialized to default values)
# for the dataSet
    
@function
def getDefaultOptsForBenchmack(dataSet=None,*args,**kwargs):
    varargin = getDefaultOptsForBenchmack.varargin
    nargin = getDefaultOptsForBenchmack.nargin

    # General options
    params.verbose = copy(true)
# .\manageOptionsForBenchmark.m:39
    
    params.plotType = copy('auto')
# .\manageOptionsForBenchmark.m:40
    
    # currently supports ['semilogy', 'linear','auto'].
    params.numTrials = copy(1)
# .\manageOptionsForBenchmark.m:42
    
    # combination will run.
    params.policy = copy('median')
# .\manageOptionsForBenchmark.m:44
    
    # for ploting from the yvalues one get
                                     # by running numTrials trials. It
                                     # currently supports
                                     # ['median','average','best',
                                     # 'successRate']
    params.successConstant = copy(1e-05)
# .\manageOptionsForBenchmark.m:50
    
    # is less than it, 
                                     # the trial will be counted as a
                                     # success. This parameter will only be
                                     # used when policy='successRate'.
    params.maxTime = copy(300)
# .\manageOptionsForBenchmark.m:55
    
    # algorithm/dataset trial.
    
    params.recordSignals = copy(false)
# .\manageOptionsForBenchmark.m:58
    
    # at each trial.
    
    # Specific options
    if '1dgaussian' == lower(dataSet):
        params.n = copy(10)
# .\manageOptionsForBenchmark.m:64
        # signal
        params.m = copy(80)
# .\manageOptionsForBenchmark.m:66
        params.isComplex = copy(true)
# .\manageOptionsForBenchmark.m:67
        params.isNonNegativeOnly = copy(false)
# .\manageOptionsForBenchmark.m:68
        # and non-negative
        params.SNR = copy(inf)
# .\manageOptionsForBenchmark.m:70
    else:
        if '2dimage' == lower(dataSet):
            params.imagePath = copy('data/shapes.png')
# .\manageOptionsForBenchmark.m:72
            params.L = copy(12)
# .\manageOptionsForBenchmark.m:73
            # created
        else:
            if 'transmissionmatrix' == lower(dataSet):
                params.n = copy(256)
# .\manageOptionsForBenchmark.m:76
                params.m = copy(dot(256,20))
# .\manageOptionsForBenchmark.m:77
                params.isComplex = copy(true)
# .\manageOptionsForBenchmark.m:78
                params.isNonNegativeOnly = copy(false)
# .\manageOptionsForBenchmark.m:79
                # and non-negative
            else:
                error('unknown dataset: %s\n',dataSet)
    
    return params
    
if __name__ == '__main__':
    pass
    