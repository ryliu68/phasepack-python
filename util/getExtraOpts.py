# Generated with SMOP  0.41
from libsmop import *
# .\getExtraOpts.m

    #                               getExtraOpts.m
# 
# This function creates and outputs a struct that consists of all the
# options in opts but not in otherOpts.
# It is used as a helper function in manageOptions.m and
# manageOptionsForBenchmark.m
    
    # Inputs:
#         opts(struct)       :  consists of options.
#         otherOpts(struct)  :  consists of options.
# Outputs:
#         extras(struct)     :  consists of extral options appearing in
#                               opts but not in otherOpts.
    
    # 
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    
@function
def getExtraOpts(opts=None,otherOpts=None,*args,**kwargs):
    varargin = getExtraOpts.varargin
    nargin = getExtraOpts.nargin

    extras=copy(struct)
# .\getExtraOpts.m:21
    optNames=fieldnames(opts)
# .\getExtraOpts.m:22
    for i in arange(1,length(optNames)).reshape(-1):
        optName=optNames[i]
# .\getExtraOpts.m:25
        if logical_not(isfield(otherOpts,optName)):
            setattr(extras,optName,getattr(opts,(optName)))
# .\getExtraOpts.m:27
    
    return extras
    
if __name__ == '__main__':
    pass
    