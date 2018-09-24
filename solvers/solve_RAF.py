## ------------------------solveRAF.m--------------------------------------
    
    # Solver for Reweighted Amplitude Flow as given in Algorithm 1
# of the Reweighted Amplitude Flow (TAF) paper. Refer to the userguide for
# a detailed usage of the package.
    
    # PAPER TITLE:
#              Solving Almost all Systems of Random Quadratic Equations
    
    # ARXIV LINK:
#              https://arxiv.org/pdf/1705.10407.pdf
    
    # INPUTS:
#         A:   Function handle/numerical matrix for data matrix A. The rows
#              of this matrix are the measurement vectors that produce
#              amplitude measurements '\psi'.
#         At:  Function handle/numerical matrix for A transpose.
#         b0:  Observed data vector consisting of amplitude measurements
#              generated from b0 = |A*x|. We assign it to 'psi' to be
#              consistent with the notation in the paper.
#         x0:  The initial vector to be used by any solver. 
#        opts: struct consists of the options for the algorithm. For
#              details,see header in solvePhaseRetrieval.m or the User
#              Guide.
    
    # OUPTUT :
#         sol: n x 1 vector. It is the estimated signal.
#        outs: A struct consists of the convergence info. For details,
#              see header in solvePhaseRetrieval.m or the User Guide.
    
    # Note:        When a function handle is used, the value of 'n' (the len
#              of the unknown signal) and 'At' (a function handle for the
#              adjoint of 'A') must be supplied. When 'A' is numeric, the
#              values of 'At' and 'n' are ignored and inferred from the
#              arguments
    
    # DESCRIPTION:
#             The spectral method proposed by Candes et al. in their
#             seminal Wirtinger Flow paper suffers from fat tails that can
#             distort the true solution. A possible way to overcome this is
#             to truncate the outliers. In this paper, the authors propose
#             a weighted truncated iterative descent method that removes
#             those measurement vector a_m that are not correlated to the
#             intial guess. This truncation and weighting step is done at
#             each iteration of the gradient descent method.
    
    # METHOD:
#         1) Our implementation uses FASTA, A Forward Backward Splitting
#             Package.
    
    #         2) Set the objectve f = @(z) 1/2 * sum(weights .* (abs(z) -
#             b0).^2). The weights are computed according to equation (14)
#             in algorithm 1 of the paper
# 
#         3) The gradient grad = @(z) weights .* (z - b0 .* sign(z)).
#             There is no non smooth term hence g = 0. Conseqently prox =
#             @(z) z
#
    
    # PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    ## -----------------------------START-----------------------------------
    
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

def solveRAF(A=None,At=None,b0=None,x0=None,opts=None,*args,**kwargs):

    #     addpath('solvers/linesearch');
    
    m=len(b0)
    b0New=max(b0,dot(1e-30,np.ones(m,1)))
    if opts.isComplex:
        beta=dot(5,np.ones(m,1))
    else:
        beta=dot(10,np.ones(m,1))
    
    
    innerOpts=struct
    innerOpts.maxIters = opts.maxIters
    innerOpts.maxTime = opts.maxTime
    innerOpts.tol = opts.tol
    innerOpts.verbose = opts.verbose
    innerOpts.recordTimes = opts.recordTimes
    innerOpts.recordResiduals = opts.recordResiduals
    innerOpts.recordMeasurementErrors = opts.recordMeasurementErrors
    innerOpts.recordReconErrors = opts.recordReconErrors
    innerOpts.xt = opts.xt
    innerOpts.updateObjectivePeriod = opts.reweightPeriod
    innerOpts.searchMethod = opts.searchMethod
    innerOpts.betaChoice = opts.betaChoice
    sol,outs=gradientDescentSolver(A,At,x0,b0,updateObjective,innerOpts,nargout=2)
    
    def updateObjective(__=None,Ax=None,*args,**kwargs):

        tmp=abs(Ax) / b0New
        weights=tmp / (tmp + beta)
        s=sum(weights)
        f=lambda z=None: dot((0.5 / s),sum(np.multiply(weights,(abs(z) - b0) ** 2)))
        gradf=lambda z=None: np.multiply(dot((1.0 / s),weights),(z - np.multiply(b0,np.sign(z))))
        return f,gradf
    return sol, outs