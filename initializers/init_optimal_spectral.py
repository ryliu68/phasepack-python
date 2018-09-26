import numpy as np
import math
import struct
from numpy import dot
import scipy
from scipy.sparse.linalg import cg


def initOptimalSpectral(A=None, At=None, b0=None, n=None, isScaled=None, verbose=None):
    # If A is a matrix, infer n and At from A. Then, transform matrix into
    # a function handle.
    # if A.isnumeric():
    #     n = np.size(A, 2)
    #     At = lambda x=None: np.dot(A.T, x)
    #     A = lambda x=None: np.dot(A, x)

    m = np.size(b0)
    if verbose == None or verbose:
        print(['Estimating signal of length {0} using an orthogonal '.format(
            n)+'initializer with {0} measurements...\n'.format(m)])

    # Measurements as defined in the paper
    y = b0 ** 2
    delta = m / n

    # Normalize the measurements
    ymean = np.mean(y)
    y = y / ymean

    # Apply pre-processing function
    yplus = []
    for i in range(len(y)):
        if y[i] < 0:
            yplus.append(0)
        else:
            yplus.append(y[i])
    yplus = np.array(yplus)
    T = (yplus - 1) / (yplus + math.sqrt(delta) - 1)

    # Un-normalize the measurements
    T = T*ymean
    # Build the function handle associated to the matrix Y
    # Yfunc = lambda x=None: (1/m)*At(np.multiply(T, np.dot(A, x)))

    # Our implemention uses Matlab's built-in function eigs() to get the leading
    # eigenvector because of greater efficiency.
    # Create opts struct for eigs
    opts = struct
    opts.isreal = False
    '''
    # Get the eigenvector that corresponds to the largest eigenvalue of the associated matrix of Yfunc.
    [x0,~] = eigs(Yfunc, n, 1, 'lr', opts);
    '''
    id = np.eye(256)
    _, x0 = scipy.sparse.linalg.eigs(id, k=1, which="LR")

    # This part does not appear in the Null paper. We add it for better performance. Rescale the solution to have approximately the correct magnitude
    if isScaled:
        b = b0
        Ax = abs(np.dot(A, x0))
        u = Ax * b
        l = Ax * Ax
        s = math.sqrt(np.dot(np.ravel(u), np.ravel(u))) / \
            math.sqrt(np.dot(np.ravel(l), np.ravel(l)))
        x0 = np.dot(x0, s)
    if verbose == None or verbose:
        print('Initialization finished.\n')

    return x0
