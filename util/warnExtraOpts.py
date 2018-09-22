# Generated with SMOP  0.41
from libsmop import *
# .\warnExtraOpts.m

    #                               getExtraOpts.m
# 
# This function raises a warning for all the fields in extras struct 
# since they won't be used by the solver in estimating the unknown signal.
# It is used as a helper function in manageOptions.m and
# manageOptionsForBenchmark.m.
# 
# Inputs:
#         extras(struct)   :  consists of extra, unnecessary options 
#                             provided by the user.
# 
# 
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    
@function
def warnExtraOpts(extras=None,*args,**kwargs):
    varargin = warnExtraOpts.varargin
    nargin = warnExtraOpts.nargin

    optNames=fieldnames(extras)
# .\warnExtraOpts.m:18
    for i in arange(1,length(optNames)).reshape(-1):
        optName=optNames[i]
# .\warnExtraOpts.m:21
        warning('Provided option "%s" is invalid and will be ignored',optName)
    
    return
    
if __name__ == '__main__':
    pass
    