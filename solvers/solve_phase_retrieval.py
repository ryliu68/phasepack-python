#                               solvePhaseRetrieval.m
#   This method solves the problem:

#                        Find x given b0 = |Ax+epsilon|
#
#  Where A is a m by n complex matrix, x is a n by 1 complex vector, b0 is
#   a m by 1 real,non-negative vector and epsilon is a m by 1 vector. The
#   user supplies function handles A, At and measurement b0. Note: The
#   unknown signal to be recovered must be 1D for our interface. Inputs:
#    A     : A m x n matrix (or optionally a function handle to a method)
#            that returns A*x
#    At    : The adjoint (transpose) of 'A.' It can be a n x m matrix or a
#            function handle.
#    b0    : A m x 1 real,non-negative vector consists of  all the
#            measurements.
#    n     : The size of the unknown signal. It must be provided if A is a
#            function handle.
#    opts  : An optional struct with options.  The commonly used fields
#            of 'opts' are:
#               initMethod              : (string,
#               default='truncatedSpectral') The method used
#                                         to generate the initial guess x0.
#                                         User can use a customized initial
#                                         guess x0 by providing value to
#                                         the field customx0.
#               algorithm               : (string, default='altmin') The
#                                         algorithm used
#                                         to solve the phase retrieval
#                                         algorithm. User can use a
#                                         customized algorithm by providing
#                                         a function [A,At,b0,x0,opts]
#                                         ->[x,outs,opts] to the field
#                                         customAlgorithm.
#               maxIters                : (integer, default=1000) The
#                                         maximum number of
#                                         iterations allowed before
#                                         termination.
#               maxTime                 : (positive real number,
#                                         default=120, unit=second)
#                                         The maximum seconds allowed
#                                         before termination.
#               tol                     : (double, default=1e-6) The
#                                         stopping tolerance.
#                                         It will be compared with
#                                         reconerror if xt is provided.
#                                         Otherwise, it will be compared
#                                         with residual. A smaller value of
#                                         'tol' usually results in more
#                                         iterations.
#               verbose                 : ([0,1,2], default=0)  If ==1,
#                                         print out
#                                         convergence information in the
#                                         end. If ==2, print out
#                                         convergence information at every
#                                         iteration.
#               recordMeasurementErrors : (boolean, default=False) Whether
#                                         to compute and record
#                                         error(i.e.
#                                         norm(|Ax|-b0)/norm(b0)) at each
#                                         iteration.
#               recordResiduals         : (boolean, default=True) If it's
#                                         True, residual will be
#                                         computed and recorded at each
#                                         iteration. If it's False,
#                                         residual won't be recorded.
#                                         Residual also won't be computed
#                                         if xt is provided. Note: The
#                                         error metric of residual varies
#                                         across
#               recordReconErrors       : (boolean, default=False) Whether
#                                         to record
#                                         reconstruction error. If it's
#                                         True, opts.xt must be provided.
#                                         If xt is provided reconstruction
#                                         error will be computed regardless
#                                         of this flag and used for
#                                         stopping condition.
#               recordTimes             : (boolean, default=True) Whether
#                                         to record
#                                         time at each iteration. Time will
#                                         be measured regardless of this
#                                         flag.
#               xt                      : A n x 1 vector. It is the True
#                                         signal. Its purpose is
#                                         to compute reconerror.

#            There are other more algorithms specific options not listed
#            here. To use these options, set the corresponding field in
#            'opts'. For example:
#                      >> opts.tol=1e-8; >> opts.maxIters = 100;

#   Outputs:
#    sol               : The approximate solution outs : A struct with
#                        convergence information
#    iterationCount    : An integer that is  the number of iterations the
#                        algorithm runs.
#    solveTimes        : A vector consists  of elapsed (exist when
#                        recordTimes==True) time at each iteration.
#    measurementErrors : A vector consists of the errors (exist when
#                        recordMeasurementErrors==True)   i.e.
#                        norm(abs(A*x-b0))/norm(b0) at each iteration.
#    reconErrors       : A vector consists of the reconstruction (exist
#                        when recordReconErrors==True) errors
#                        i.e. norm(xt-x)/norm(xt) at each iteration.
#    residuals         : A vector consists of values that (exist when
#                        recordResiduals==True)  will be compared with
#                        opts.tol for stopping condition  checking.
#                        Definition varies across
#    opts              : A struct that contains fields used by the solver.
#                        Its possible fields are the same as the input
#                        parameter opts.
#
# For more details and more options in opts, see the Phasepack user
#   guide.

# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein Copyright (c) University of Maryland,
# 2017

# -----------------------------START-----------------------------------
import sys
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/solvers')
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/initializers')
sys.path.append(u'C:/Users/MrLiu/Desktop/phasepack-python/util')
import struct

from inspect import isfunction

import numpy as np
from numpy import dot as dot

from init_amplitude import initAmplitude
from init_angle import initAngle
from init_optimal_spectral import initOptimalSpectral
from init_orthogonal import initOrthogonal
from init_spectral import initSpectral
from init_weighted import initWeighted
from manage_options import manageOptions
from solve_amplitude_flow import solveAmplitudeFlow
from solve_coordinate_dcescent import solveCoordinateDescent
# from import optsCustomAlgorithm
from solve_fienup import solveFienup
from solve_gerchberg_saxton import solveGerchbergSaxton
# from solve_phase_retrieval import solvePhaseRetrieval
from solve_phase_lamp import solvePhaseLamp
from solve_phase_lift import solvePhaseLift
from solve_phase_max import solvePhaseMax
# from import solveKaczmarzSimple
# from import solve_phase_lift
from solve_RAF import solveRAF
from solve_RWF import solveRWF
from solve_sketchy_CGM import solveSketchyCGM
from solve_TAF import solveTAF
from solve_TWF import solveTWF
from solve_wirt_flow import solveWirtFlow
from solve_kaczmarz_simple import solveKaczmarzSimple



# from  solveCoordinateDescent     solveRWF   solveTAF


def solvePhaseRetrieval(A=None, At=None, b0=None, n=None, opts=None, *args, **kwargs):

    # Add path to helper functions
    # addpath('util');
    # addpath('initializers');

    # If opts is not provided, create it
    if not opts:
        opts = struct

    # If A is a matrix, infer n and At from A
    # print(np.isnan(A[0][0]))
    # print(A[0][0])
    # if isnumeric(A) & ~isempty(A)
    if (not np.isnan(A[0][0])) & (A[0][0] != None):
        # n = size(A, 2)
        n = A.shape[2-1]
        print(A.shape)
        print(n)
        # Transform matrix into function form
        def get_At(x=None):
            return dot(A.T, x)
        def get_A(x=None):
            return dot(A, x)
        # At = lambda x=None: dot(A.T, x)
        # A = lambda x=None: dot(A, x)
    # print(type(A()))

    # Apply default options and validate user-defined options
    opts = manageOptions(opts)
    # Check that inputs are of valid datatypes and sizes
    validateInput(A, At, b0, n, opts)
    # Check that At is the adjoint/transpose of A
    checkAdjoint(A, At, b0)
    x0 = initX(A, At, b0, n, opts)

    # Truncate imaginary components of x0 if working with real values
    if not (opts.isComplex):
        x0 = x0.real()
    else:
        if opts.isNonNegativeOnly:
            # warning('opts.isNonNegativeOnly will not be used when the signal is complex.')
            print('opts.isNonNegativeOnly will not be used when the signal is complex.')

    sol, outs = solveX(A, At, b0, x0, opts)

    return [sol, outs, opts]


# Helper functions
# Initialize x0 using the specified initialization method
def initX(A=None, At=None, b0=None, n=None, opts=None):
    if (opts.initMethod).lower() in ['truncatedspectral', 'truncated']:
        x0 = initSpectral(A, At, b0, n, True, True, opts.verbose)
    elif 'spectral' == (opts.initMethod).lower():
        x0 = initSpectral(A, At, b0, n, False, True, opts.verbose)
    elif (opts.initMethod).lower() in ['amplitudespectral', 'amplitude']:
        x0 = initAmplitude(A, At, b0, n, opts.verbose)
    elif (opts.initMethod).lower() in ['weightedspectral', 'weighted']:
        x0 = initWeighted(A, At, b0, n, opts.verbose)
    elif (opts.initMethod).lower() in ['orthogonalspectral', 'orthogonal']:
        x0 = initOrthogonal(A, At, b0, n, opts.verbose)
    elif (opts.initMethod).lower() in ['optimal', 'optimalspectral']:
        x0 = initOptimalSpectral(A, At, b0, n, True, opts.verbose)
    elif 'angle' == (opts.initMethod).lower():
        assert(hasattr(opts, 'xt'),
               'The True solution, opts.xt, must be specified to use the angle initializer.')
        assert(hasattr(opts, 'initAngle'),
               'An angle, opts.initAngle, must be specified (in radians) to use the angle initializer.')
        x0 = initAngle(opts.xt, opts.initAngle)
    elif 'custom' == (opts.initMethod).lower():
        x0 = opts.customx0
    else:
        print("error", 'Unknown initialization method "%s"', opts.initMethod)

    return x0

# Estimate x0 using the specified algorithm


def solveX(A=None, At=None, b0=None, x0=None, opts=None, *args, **kwargs):
    # print((opts.algorithm).lower())

    if 'custom' == (opts.algorithm).lower():
        sol, outs = optsCustomAlgorithm(A, At, b0, x0, opts)
    elif 'amplitudeflow' == (opts.algorithm).lower():
        sol, outs = solveAmplitudeFlow(A, At, b0, x0, opts)
    elif 'coordinatedescent' == (opts.algorithm).lower():
        sol, outs = solveCoordinateDescent(A, At, b0, x0, opts)
    elif 'fienup' == (opts.algorithm).lower():
        sol, outs = solveFienup(A, At, b0, x0, opts)
    elif 'gerchbergsaxton' == (opts.algorithm).lower():
        sol, outs = solveGerchbergSaxton(A, At, b0, x0, opts)
    elif 'kaczmarz' == (opts.algorithm).lower():
        sol, outs = solveKaczmarzSimple(A, At, b0, x0, opts)
    elif 'phasemax' == (opts.algorithm).lower():
        sol, outs = solvePhaseMax(A, At, b0, x0, opts)
    elif 'phaselamp' == (opts.algorithm).lower():
        sol, outs = solvePhaseLamp(A, At, b0, x0, opts)
    elif 'phaselift' == (opts.algorithm).lower():
        sol, outs = solvePhaseLift(A, At, b0, x0, opts)
    elif 'raf' == (opts.algorithm).lower():
        sol, outs = solveRAF(A, At, b0, x0, opts)
    elif 'rwf' == (opts.algorithm).lower():
        sol, outs = solveRWF(A, At, b0, x0, opts)
    elif 'sketchycgm' == (opts.algorithm).lower():
        sol, outs = solveSketchyCGM(A, At, b0, x0, opts)
    elif 'taf' == (opts.algorithm).lower():
        sol, outs = solveTAF(A, At, b0, x0, opts)
    elif 'twf' == (opts.algorithm).lower():
        sol, outs = solveTWF(A, At, b0, x0, opts)
    elif 'wirtflow' == (opts.algorithm).lower():
        sol, outs = solveWirtFlow(A, At, b0, x0, opts)
    else:
        error('Unknown algorithm "%s"', opts.algorithm)

    return sol, outs


# Check validity of input


def validateInput(A=None, At=None, b0=None, n=None, opts=None, *args, **kwargs):
    # print(888888888888)
    # print(np.isnan(A[0][0]))
    # print(888888888888)

    # if ~isnumeric(A) & (isempty(At)|isempty(n))
    '''
    if (not np.isnan(A[0][0])) & ((At == None) | (n == None)):
        # error('If A is a function handle, then At and n must be provided')
        print('If A is a function handle, then At and n must be provided')
    '''
    assert(n > 0, 'n must be positive')
    assert(abs(b0) == b0, 'b must be real-valued and non-negative')

    # if ~isnumeric(A) & isnumeric(At)
    '''
    if ( np.isnan(A[0][0])) & (not np.isnan(A[0][0])):
        # error('If A is a function handle, then At must also be a function handle');
        print('If A is a function handle, then At must also be a function handle')
    '''

    if not(opts.customx0 == None):
        # assert(isequal(size(opts.customx0), [n 1]), 'customx0 must be a column vector of length n');
        # print(np.shape(opts.customx0)==(n,1))
        assert(np.shape(opts.customx0) == (n, 1),
               'customx0 must be a column vector of length n')

    return


# Check that A and At are indeed ajoints of one another
def checkAdjoint(A=None, At=None, b=None, *args, **kwargs):
    # print(np.shape(b))
    y = np.random.randn(np.shape(b)[0])
    # Aty = At(y)
    Aty = dot(A.T, y)
    # print("Aty:",Aty)
    x = np.random.randn(np.shape(Aty)[0])
    # Ax = A(x)
    Ax = dot(A, x)
    # A.reshape(6,order='F')
    # innerProduct1 = Ax(:)'*y(:);
    # innerProduct2 = x(:)'*Aty(:);
    innerProduct1 = np.dot(Ax.flatten('F'), y)
    # print(innerProduct1)
    innerProduct2 = np.dot(x, Aty.flatten('F'))
    # print(innerProduct2)
    error = abs(innerProduct1 - innerProduct2) / abs(innerProduct1)
    if error >= 0.001:
        print('Invalid measurement operator:  At is not the adjoint of A.  Error = {0}'.format(str(error)))
    # assert(error < 0.001, concat(
    #     ['Invalid measurement operator:  At is not the adjoint of A.  Error = ', str(error)]))
    return