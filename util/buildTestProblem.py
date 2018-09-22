# Generated with SMOP  0.41
from libsmop import *
# .\buildTestProblem.m

    #                           buildTestProblem.m
    
    # This function creates and outputs random generated data and measurements
# according to user's choice. It is invoked in test*.m in
# order to build a test problem.
    
    # Inputs:
#   m(integer): number of measurements.
#   n(integer): length of the unknown signal.
#   isComplex(boolean, default=true): whether the signal and measurement
#     matrix is complex. isNonNegativeOnly(boolean, default=false): whether
#     the signal is real and non-negative.
#   dataType(string, default='gaussian'): it currently supports
#     ['gaussian', 'fourier'].
    
    # Outputs:
#   A: m x n measurement matrix/function handle.
#   xt: n x 1 vector, true signal.
#   b0: m x 1 vector, measurements.
#   At: A n x m matrix/function handle that is the transpose of A.
    
    
    # PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017
    
    
@function
def buildTestProblem(m=None,n=None,isComplex=None,isNonNegativeOnly=None,dataType=None,*args,**kwargs):
    varargin = buildTestProblem.varargin
    nargin = buildTestProblem.nargin

    if logical_or(logical_not(exist('isComplex','var')),isempty(isComplex)):
        isComplex=copy(true)
# .\buildTestProblem.m:29
    
    if logical_or(logical_not(exist('isNonNegativeOnly','var')),isempty(isNonNegativeOnly)):
        isNonNegativeOnly=copy(false)
# .\buildTestProblem.m:32
    
    if logical_not(exist('dataType','var')):
        dataType='Gaussian'
# .\buildTestProblem.m:35
    
    if 'gaussian' == lower(dataType):
        A=(mvnrnd(zeros(1,n),eye(n) / 2,m) + dot(dot(isComplex,1j),mvnrnd(zeros(1,n),eye(n) / 2,m)))
# .\buildTestProblem.m:40
        At=A.T
# .\buildTestProblem.m:42
        xt=(mvnrnd(zeros(1,n),eye(n) / 2) + dot(dot(isComplex,1j),mvnrnd(zeros(1,n),eye(n) / 2))).T
# .\buildTestProblem.m:43
        b0=abs(dot(A,xt))
# .\buildTestProblem.m:45
    else:
        if 'fourier' == lower(dataType):
            #  Define the Fourier measurement operator.
        #  The operator 'A' maps an n-vector into an m-vector, then
        #  computes the fft on that m-vector to produce m measurements.
            # rips first 'length' entries from a vector
            rip=lambda x=None,length=None: x(arange(1,length))
# .\buildTestProblem.m:52
            A=lambda x=None: fft(concat([[x],[zeros(m - n,1)]]))
# .\buildTestProblem.m:53
            At=lambda x=None: rip(dot(m,ifft(x)),n)
# .\buildTestProblem.m:54
            xt=(mvnrnd(zeros(1,n),eye(n) / 2) + dot(dot(isComplex,1j),mvnrnd(zeros(1,n),eye(n) / 2))).T
# .\buildTestProblem.m:55
            b0=abs(A(xt))
# .\buildTestProblem.m:57
        else:
            error('invalid dataType: %s',dataType)
    
    return A,xt,b0,At
    
if __name__ == '__main__':
    pass
    