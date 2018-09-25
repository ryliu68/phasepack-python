from numpy.linalg import norm
from numpy import dot
from numpy import eye  
import numpy as np
from numpy.linalg import cholesky


def buildTestProblem(m=None, n=None, isComplex=None, isNonNegativeOnly=None, dataType=None):
    if isComplex == None:
        isComplex = True

    if isNonNegativeOnly == None:
        isNonNegativeOnly = False

    if dataType == None:
        dataType = 'Gaussian'

    if 'gaussian' == dataType.lower():
        real_part_A = np.dot(np.random.randn(
            m, n), cholesky(eye(n)/2)) + np.zeros([1, n])
        A = real_part_A + np.dot(1j, real_part_A) * isComplex
        # At = A.T
        real_part_xt = np.dot(np.random.randn(
            n), cholesky(eye(n)/2)) + np.zeros([1, n])
        xt = (real_part_xt + np.dot(1j, real_part_xt) * isComplex).T
        b0 = abs(dot(A, xt))
    else:
        print("error", 'invalid dataType: {0}'.format(dataType))

    return A, xt, b0
