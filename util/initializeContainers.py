# Generated with SMOP  0.41
from libsmop import *
# .\initializeContainers.m

    #                               initializeContainers.m
# 
# This function initializes and outputs containers for convergence info
# according to user's choice. It is invoked in solve*.m.
# 
# Inputs:
#         opts(struct)              :  consists of options.
# Outputs:
#         solveTimes(struct)        :  empty [] or initialized with
#                                      opts.maxIters x 1 zeros if
#                                      recordTimes. 
#         measurementErrors(struct) :  empty [] or initialized with 
#                                      opts.maxIters x 1 zeros if
#                                      recordMeasurementErrors.
#         reconErrors(struct)       :  empty [] or initialized with 
#                                      opts.maxIters x 1 zeros if
#                                      recordReconErrors.
#         residuals(struct)         :  empty [] or initialized with
#                                      opts.maxIters x 1 zeros if
#                                      recordResiduals.
# 
# 
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    
@function
def initializeContainers(opts=None,*args,**kwargs):
    varargin = initializeContainers.varargin
    nargin = initializeContainers.nargin

    solveTimes=[]
# .\initializeContainers.m:28
    measurementErrors=[]
# .\initializeContainers.m:29
    reconErrors=[]
# .\initializeContainers.m:30
    residuals=[]
# .\initializeContainers.m:31
    if opts.recordTimes:
        solveTimes=zeros(opts.maxIters,1)
# .\initializeContainers.m:33
    
    if opts.recordMeasurementErrors:
        measurementErrors=zeros(opts.maxIters,1)
# .\initializeContainers.m:36
    
    if opts.recordReconErrors:
        reconErrors=zeros(opts.maxIters,1)
# .\initializeContainers.m:39
    
    if opts.recordResiduals:
        residuals=zeros(opts.maxIters,1)
# .\initializeContainers.m:42
    
    return solveTimes,measurementErrors,reconErrors,residuals
    
if __name__ == '__main__':
    pass
    