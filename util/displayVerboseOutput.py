# Generated with SMOP  0.41
from libsmop import *
# .\displayVerboseOutput.m

    #                               displayVerboseOutput.m
    
    # This function prints out the convergence information at the current
# iteration. It will be invoked inside solve*.m if opts.verbose is set 
# to be >=1.
    
    # Inputs:
#   iter(integer)                        : Current iteration number.
#   currentTime(real number)             : Elapsed time so far(clock starts
#                                          when the algorithm main loop
#                                          started).
#   currentResid(real number)            : Definition depends on the 
#                                          specific algorithm used see the
#                                          specific algorithm's file's
#                                          header for details.
#   currentReconError(real number)       : relative reconstruction error.
#                                          norm(xt-x)/norm(xt), where xt
#                                          is the m x 1 true signal, x is
#                                          the n x 1 estimated signal.
#                                                
#   currentMeasurementError(real number) : norm(abs(Ax)-b0)/norm(b0), where
#                                          A is the m x n measurement
#                                          matrix or function handle
#                                          x is the n x 1 estimated signal
#                                          and b0 is the m x 1 
#                                          measurements.
    
    # 
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    
@function
def displayVerboseOutput(iter=None,currentTime=None,currentResid=None,currentReconError=None,currentMeasurementError=None,*args,**kwargs):
    varargin = displayVerboseOutput.varargin
    nargin = displayVerboseOutput.nargin

    fprintf('Iteration = %d',iter)
    fprintf(' | IterationTime = %f',currentTime)
    if logical_not(isempty(currentResid)):
        fprintf(' | Residual = %d',currentResid)
    
    if logical_not(isempty(currentReconError)):
        fprintf(' | currentReconError = %d',currentReconError)
    
    if logical_not(isempty(currentMeasurementError)):
        fprintf(' | MeasurementError = %d',currentMeasurementError)
    
    fprintf('\n')
    return
    
if __name__ == '__main__':
    pass
    