# Generated with SMOP  0.41
from libsmop import *
# .\generateOutputs.m

    #                               generateOutput.m
    
    # Generate output struct according to the convergence info recorded.
# 
# Inputs:
#    opts(struct)                 : consists of options by default of
#                                   chosen by user.
#                                   For details see the User Guide.
#    iter(integer)                : The total iterations a solver runs.
#    solveTimes(vector)           : consists of elapsed time at each
#                                   iteration. 
#    measurementErrors(vector)    : consists of measurement
#                                   error at each iteration. A single
#                                   measurement error at a certain
#                                   iteration is equal to
#                                   norm(abs(Ax)-b0)/norm(b0), where A is
#                                   the m x n measurement matrix or
#                                   function handle x is the n x 1
#                                   estimated signal at that iteration and
#                                   b0 is the m x 1 measurements.
#    reconErrors(vector)          : consists of relative reconstruction
#                                   error at each iteration.
#                                   A single reconstruction error at a
#                                   certain iteration is equal to
#                                   norm(xt-x)/norm(xt), where xt is the m
#                                   x 1 true signal, x is the n x 1
#                                   estimated signal at that iteration.
#    residuals(vector)            : consists of residuals at each
#                                   iteration.
#                                   Definition of a single residual depends
#                                   on the specific algorithm used see the
#                                   specific algorithm's file's header for
#                                   details.
# Outputs:
#    outs : A struct with convergence information
#    iterationCount(integer) : the number of
#                              iteration the algorithm runs. 
#    solveTimes(vector) : consists of elapsed (exist when
#                         recordTimes==true) time at each iteration.
#                         
#    measurementErrors(vector) : consists of the errors (exist when
#                                recordMeasurementErrors==true)   i.e.
#           norm(abs(A*x-b0))/norm(b0) at each iteration.
#           
#    reconErrors(vector): consists of the reconstruction (exist when
#                      recordReconErrors==true) errors i.e.
#                      norm(xt-x)/norm(xt) at each iteration.
#          
#    residuals(vector): consists of values that (exist when
#                       recordResiduals==true) will be compared with
#                       opts.tol for stopping condition checking.
#                       Definition varies across solvers.
    
    
    # PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    ## -----------------------------START----------------------------------
    
    
@function
def generateOutputs(opts=None,iter=None,solveTimes=None,measurementErrors=None,reconErrors=None,residuals=None,*args,**kwargs):
    varargin = generateOutputs.varargin
    nargin = generateOutputs.nargin

    outs=copy(struct)
# .\generateOutputs.m:64
    if logical_not(isempty(solveTimes)):
        outs.solveTimes = copy(solveTimes(arange(1,iter)))
# .\generateOutputs.m:66
    
    if logical_not(isempty(measurementErrors)):
        outs.measurementErrors = copy(measurementErrors(arange(1,iter)))
# .\generateOutputs.m:69
    
    if logical_not(isempty(reconErrors)):
        outs.reconErrors = copy(reconErrors(arange(1,iter)))
# .\generateOutputs.m:72
    
    if logical_not(isempty(residuals)):
        outs.residuals = copy(residuals(arange(1,iter)))
# .\generateOutputs.m:75
    
    outs.iterationCount = copy(iter)
# .\generateOutputs.m:77
    return outs
    
if __name__ == '__main__':
    pass
    