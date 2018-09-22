# Generated with SMOP  0.41
from libsmop import *
# .\runSignalReconstructionDemo.m

    ##                   runSignalReconstructionDemo.m
    
    # This script will create phaseless measurements from a 1d test signal, and 
# then recover the image using phase retrieval methods.  We now describe 
# the details of the simple recovery problem that this script implements.
# 
#                         Recovery Problem
# This script creates a complex-valued random Gaussian signal. Measurements
# of the signal are then obtained by applying a linear operator to the
# signal, and computing the magnitude (i.e., removing the phase) of 
# the results.
    
    #                       Measurement Operator
# Measurement are obtained using a linear operator, called 'A', that 
# contains random Gaussian entries.
    
    #                      The Recovery Algorithm
# The image is recovered by calling the method 'solvePhaseRetrieval', and
# handing the measurement operator and linear measurements in as arguments.
# A struct containing options is also handed to 'solvePhaseRetrieval'.
# The entries in this struct specify which recovery algorithm is used.
    
    # For more details, see the Phasepack user guide.
    
    # PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    
@function
def runSignalReconstructionDemo(*args,**kwargs):
    varargin = runSignalReconstructionDemo.varargin
    nargin = runSignalReconstructionDemo.nargin

    ## Specify the signal length, and number of measurements
    n=100
# .\runSignalReconstructionDemo.m:32
    
    m=dot(8,n)
# .\runSignalReconstructionDemo.m:33
    
    ## Build the target signal
    x_true=randn(n,1) + dot(1j,randn(n,1))
# .\runSignalReconstructionDemo.m:36
    ## Create the measurement operator
# Note: we use a dense matrix in this example, but PhasePack also supports
# function handles.  See the more complex 'runImageReconstructionDemo.m'
# script for an example using the fast Fourier transform.
    A=randn(m,n) + dot(1j,randn(m,n))
# .\runSignalReconstructionDemo.m:42
    ## Compute phaseless measurements
    b=abs(dot(A,x_true))
# .\runSignalReconstructionDemo.m:45
    ## Set options for PhasePack - this is where we choose the recovery algorithm
    opts=copy(struct)
# .\runSignalReconstructionDemo.m:48
    
    opts.algorithm = copy('PhaseMax')
# .\runSignalReconstructionDemo.m:49
    
    opts.initMethod = copy('optimal')
# .\runSignalReconstructionDemo.m:50
    
    opts.tol = copy(0.001)
# .\runSignalReconstructionDemo.m:51
    
    opts.verbose = copy(2)
# .\runSignalReconstructionDemo.m:52
    
    ## Run the Phase retrieval Algorithm
    fprintf('Running %s algorithm\n',opts.algorithm)
    # Call the solver using the measurement operator 'A', the
# measurements 'b', the length of the signal to be recovered, and the
# options.  Note, the measurement operator can be either a function handle
# or a matrix.   Here, we use a matrix.  In this case, we have omitted the 
# second argument. If 'A' had been a function handle, we would have 
# handed the transpose of 'A' in as the second argument.
    x,outs=solvePhaseRetrieval(A,[],b,n,opts,nargout=2)
# .\runSignalReconstructionDemo.m:62
    # Note: 'outs' is a struct containing convergene information.
    
    ## Remove phase ambiguity
# Phase retrieval can only recover images up to a phase ambiguity. 
# Let's apply a phase rotation to align the recovered signal with the 
# original so they look the same when we plot them.
    rotation=sign(dot(x.T,ravel(x_true)))
# .\runSignalReconstructionDemo.m:69
    x=dot(x,rotation)
# .\runSignalReconstructionDemo.m:70
    # Print some useful info to the console
    fprintf('Signal recovery required %d iterations (%f secs)\n',outs.iterationCount,outs.solveTimes(end()))
    ## Plot results
    figure
    # Plot the true vs recovered signal.  Ideally, this scatter plot should be
# clustered around the 45-degree line.
    subplot(1,2,1)
    scatter(real(x_true),real(x))
    xlabel('Original signal value')
    ylabel('Recovered signal value')
    title('Original vs recovered signal')
    # Plot a convergence curve
    subplot(1,3,3)
    convergedCurve=semilogy(outs.solveTimes,outs.residuals)
# .\runSignalReconstructionDemo.m:88
    set(convergedCurve,'linewidth',1.75)
    grid('on')
    xlabel('Time (sec)')
    ylabel('Error')
    title('Convergence Curve')
    set(gcf,'units','points','position',concat([0,0,1200,300]))
    return
    
if __name__ == '__main__':
    pass
    