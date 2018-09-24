#                           solveTWF.m

#  Implementation of the truncated Wirtinger Flow (TWF) algorithm.
#  The code below is adapted from implementation of the
#  Wirtinger Flow algorithm designed and implemented by E. Candes, X. Li,
#  and M. Soltanolkotabi.

# I/O
#  Inputs:
#     A:    m x n matrix or a function handle to a method that
#           returns A*x.
#     At:   The adjoint (transpose) of 'A'. If 'A' is a function handle,
#           'At' must be provided.
#     b0:   m x 1 real,non-negative vector consists of all the measurements.
#     x0:   n x 1 vector. It is the initial guess of the unknown signal x.
#     opts: A struct consists of the options for the algorithm. For details,
#           see header in solvePhaseRetrieval.m or the User Guide.

#     Note: When a function handle is used, the
#     value of 'At' (a function handle for the adjoint of 'A') must be
#     supplied.
#
# Outputs:
#     sol:  n x 1 vector. It is the estimated signal.
#     outs: A struct containing convergence info. For details,
#           see header in solvePhaseRetrieval.m or the User Guide.
#
#
#  See the script 'testTWF.m' for an example of proper usage of this
#  function.

# Notations
#  x is the estimation of the signal. y is the vector of measurements such
#  that yi = |<ai,x>|^2 for i = 1,...,m Most of our notations are
#  consistent with the notations used in the TWF paper referenced below.

# Algorithm Description
#  Similar to WF, TWF successively refines the estimate via a gradient
#  descent scheme.  The loss function is the negative log of the Poisson
#  likelihood.
#
#  Unlike WF, TWF regularizes the gradient flow in a data-dependent fashion
#  by operating only upon some iteration-varying index subsets that
#  correspond to those data yi whose resulting gradient components are in
#  some sense not excessively large.
#
#  This gives us a more stable search directions and avoids the overshoot
#  problem of the Wirtinger Flow Algorithm.

#  We also add a feature: when opts.isComplex==false and
#  opts.isNonNegativeOnly==true i.e. when the signal is real and
#  non_negative signal, then at each iteration, negative values in the
#  latest solution vector will be set to 0. This helps to speed up the
#  convergence of errors.
#
#  For a detailed explanation, see the TWF paper referenced below.

# References
#  Paper Title:   Solving Random Quadratic Systems of Equations Is Nearly
#                 as Easy as Solving Linear Systems
#  Place:         Algorithm 1
#  Authors:       Yuxin Chen, Emmanuel J. Candes
#  arXiv Address: https://arxiv.org/abs/1505.05114
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

# -----------------------------START-----------------------------------
import math
import struct
import sys
import time

import numpy as np
from numpy import dot
from numpy.linalg import norm
from numpy.random import randn

from check_if_number import checkIfNumber
from display_verbose_output import displayVerboseOutput
from generate_outputs import generateOutputs
from stop_now import stopNow

sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/solvers')
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/initializers')
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/util')


def solveTWF(A=None, At=None, b0=None, x0=None, opts=None, *args, **kwargs):

    #     addpath('solvers/linesearch');

    innerOpts = struct
    innerOpts.maxIters = opts.maxIters
    innerOpts.maxTime = opts.maxTime
    innerOpts.tol = opts.tol
    innerOpts.verbose = opts.verbose
    innerOpts.recordTimes = opts.recordTimes
    innerOpts.recordResiduals = opts.recordResiduals
    innerOpts.recordMeasurementErrors = opts.recordMeasurementErrors
    innerOpts.recordReconErrors = opts.recordReconErrors
    innerOpts.xt = opts.xt
    innerOpts.updateObjectivePeriod = opts.truncationPeriod
    innerOpts.searchMethod = opts.searchMethod
    innerOpts.betaChoice = opts.betaChoice
    sol, outs = gradientDescentSolver(
        A, At, x0, b0, updateObjective, innerOpts, nargout=2)


    def updateObjective(x=None, Ax=None, *args, **kwargs):
        y = b0 ** 2

        m = numel(y)

        Kt = dot(1 / m, norm(ravel(y) - abs(ravel(Ax)) ** 2, 1))

        # Truncation rules
        # Unlike what specified in the TWF paper Algorithm1, the
        # term sqrt(n)/abs(x) does not appear in the following equations
        Eub = abs(Ax) / norm(x) < opts.alpha_ub

        Elb = abs(Ax) / norm(x) > opts.alpha_lb

        Eh = abs(y - abs(Ax) ** 2) < dot(dot(opts.alpha_h, Kt), abs(Ax)) / norm(x)

        mask = np.multiply(np.multiply(Eub, Elb), Eh)
        s = sum(mask)
        f = lambda z=None: dot(
            (0.5 / s), sum(np.multiply(mask, (abs(z) ** 2 - np.multiply(y, log(abs(z) ** 2))))))
        gradf = lambda z=None: np.multiply(
            dot((1.0 / s), mask), (abs(z) ** 2 - y)) / conj(z)
        return f, gradf
    return sol, outs