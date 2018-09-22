# Generated with SMOP  0.41
from libsmop import *
# experimentGaussian1D.m

    #                                   experimentGaussian1D.m
    
    # Produce a random signal reconstruction problem with a random Gaussian
# signal and a random Gaussian measurement matrix.
    
    # Inputs:
#  n                : number of unknowns.
#  m                : number of measurements.
#  isComplex        : If the signal is complex.
#  isNonNegativeOnly: If the signal is non-negative.
#  SNR              : Signal-Noise ratio.
    
    # Outputs:
#  A     : A function handle: nx1 -> mx1. It returns 
#          A*x.
#  At    : The transpose of A
#  b0    : A mx1 real, non-negative vector consists of the 
#          measurements abs(A*x).
#  xt    : The true signal.
    
    
    
    # PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    
@function
def experimentGaussian1D(n=None,m=None,isComplex=None,isNonNegativeOnly=None,*args,**kwargs):
    varargin = experimentGaussian1D.varargin
    nargin = experimentGaussian1D.nargin

    
    # Build the test problem
    xt=randn(n,1) + dot(dot(isComplex,randn(n,1)),1j)
# experimentGaussian1D.m:31
    
    if isNonNegativeOnly:
        xt=abs(xt)
# experimentGaussian1D.m:34
    
    
    A=randn(m,n) + dot(dot(isComplex,randn(m,n)),1j)
# experimentGaussian1D.m:37
    
    b0=abs(dot(A,xt))
# experimentGaussian1D.m:38
    
    
    # Build function handles for the problem
    At=lambda x=None: dot(A.T,x)
# experimentGaussian1D.m:41
    A=lambda x=None: dot(A,x)
# experimentGaussian1D.m:42
    plotter=lambda x=None: plot(x,xt)
# experimentGaussian1D.m:45
    return A,At,b0,xt,plotter
    
if __name__ == '__main__':
    pass
    
    
@function
def plot(x=None,xt=None,*args,**kwargs):
    varargin = plot.varargin
    nargin = plot.nargin

    scatter(abs(xt),abs(x))
    xlabel('true signal','fontsize',14)
    ylabel('recovered signal','fontsize',14)
    return
    
if __name__ == '__main__':
    pass
    