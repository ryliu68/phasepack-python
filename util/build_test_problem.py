#                           buildTestProblem.m
#
# This function creates and outputs random generated data and measurements
# according to user's choice. It is invoked in test*.m in
# order to build a test problem.
#
# Inputs:
#   m(integer): number of measurements.
#   n(integer): length of the unknown signal.
#   isComplex(boolean, default=true): whether the signal and measurement
#     matrix is complex. isNonNegativeOnly(boolean, default=false): whether
#     the signal is real and non-negative.
#   dataType(string, default='gaussian'): it currently supports
#     ['gaussian', 'fourier'].
#
# Outputs:
#   A: m x n measurement matrix/function handle.
#   xt: n x 1 vector, true signal.
#   b0: m x 1 vector, measurements.
#   At: A n x m matrix/function handle that is the transpose of A.
#
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017


import numpy as np
from numpy.linalg import norm
from numpy import dot
from numpy import eye as eye
from numpy.fft import fft as fft
from numpy.fft import ifft as ifft
import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt


def buildTestProblem(m=None, n=None, isComplex=None, isNonNegativeOnly=None, dataType=None, *args, **kwargs):
    if (not isComplex) | (not isComplex):
        isComplex = True

    if (not isNonNegativeOnly) | (not isNonNegativeOnly):
        isNonNegativeOnly = False

    if not dataType:
        dataType = 'Gaussian'

    if 'gaussian' == dataType.lower():
        # A = (mvnrnd(zeros(1, n), eye(n)/2, m) + isComplex * 1i * mvnrnd(zeros(1, n), eye(n)/2, m))
        real_part_A = np.dot(np.random.randn(m, n), cholesky(eye(n)/2)) + np.zeros([1, n])
        A = real_part_A + np.dot(1j, real_part_A) * isComplex
        At = A.T
        #xt = (mvnrnd(zeros(1, n), eye(n)/2) + isComplex * 1i * mvnrnd(zeros(1, n), eye(n)/2))'     
        real_part_xt = np.dot(np.random.randn(n), cholesky(eye(n)/2)) + np.zeros([1, n])
        xt = (real_part_xt + np.dot(1j, real_part_xt) * isComplex).T
        b0 = abs(dot(A, xt))
    elif 'fourier' == dataType.lower():
        #  Define the Fourier measurement operator.
        #  The operator 'A' maps an n-vector into an m-vector, then
        #  computes the fft on that m-vector to produce m measurements.
        # rips first 'length' entries from a vector
        rip = lambda x=None, length=None: x(range(1, length))
        A = lambda x=None: fft(concat([[x], [np.zeros(m - n, 1)]]))
        At = lambda x=None: rip(dot(m, ifft(x)), n)
        xt = (np.array(np.zeros(1, n), eye(n) / 2) +
                dot(dot(isComplex, 1j), np.array(np.zeros(1, n), eye(n) / 2))).T
        b0 = abs(A(xt))
    else:
        # error('invalid dataType: %s',dataType)
        print("error", 'invalid dataType: {0}'.format(dataType))

    return A, xt, b0