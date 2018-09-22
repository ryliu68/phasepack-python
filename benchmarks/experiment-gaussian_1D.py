#                                   experimentGaussian1D.m
#
# Produce a random signal reconstruction problem with a random Gaussian
# signal and a random Gaussian measurement matrix.
#
# Inputs:
#  n                : number of unknowns.
#  m                : number of measurements.
#  isComplex        : If the signal is complex.
#  isNonNegativeOnly: If the signal is non-negative.
#  SNR              : Signal-Noise ratio.
#
# Outputs:
#  A     : A function handle: nx1 -> mx1. It returns 
#          A*x.
#  At    : The transpose of A
#  b0    : A mx1 real, non-negative vector consists of the 
#          measurements abs(A*x).
#  xt    : The true signal.
#
#
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017

'''
function [A, At, b0, xt, plotter] = experimentGaussian1D(n, m, isComplex, isNonNegativeOnly)
    
    # Build the test problem
    xt = randn(n, 1)+isComplex*randn(n, 1)*1i# # true solution

    if isNonNegativeOnly                       
        xt = abs(xt)#
    end
    
    A = randn(m, n)+isComplex*randn(m, n)*1i# # matrix
    b0 = abs(A*xt)#                           # measurements
    
    # Build function handles for the problem
    At = @(x) A'*x#
    A = @(x) A*x#
   
    
    plotter = @(x) plot(x,xt)#

end
'''

import matplotlib.pyplot as plt
import numpy as np
def experimentGaussian1D(n=None,m=None,isComplex=None,isNonNegativeOnly=None,*args,**kwargs):
    # Build the test problem
    xt=np.random.randn(n,1) + np.dot(np.dot(isComplex,np.random.randn(n,1)),1j)
    
    if isNonNegativeOnly:
        xt=abs(xt)
    
    
    A=np.random.randn(m,n) + np.dot(np.dot(isComplex,np.random.randn(m,n)),1j)
    
    b0=abs(np.dot(A,xt))
    
    
    # Build function handles for the problem
    At=lambda x=None: np.dot(A.T,x)
    A=lambda x=None: np.dot(A,x)
    plotter=lambda x=None: plot(x,xt)
    return A,At,b0,xt,plotter

'''

function plot(x,xt)
    scatter(abs(xt),abs(x))#
    xlabel('true signal','fontsize',14)#
    ylabel('recovered signal','fontsize',14)#
end
'''
def plot(x=None,xt=None,*args,**kwargs):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(abs(xt),abs(x))
    ax.set_xlabel('true signal','fontsize',14)
    ax.set_ylabel('recovered signal','fontsize',14)
    plt.show()
    return