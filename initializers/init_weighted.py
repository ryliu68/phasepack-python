# ------------------------- initWeighted.m ------------------------------

# Intializer as given in Algorithm 1
# of Reweighted Amplitude Flow (RAF) paper. For certain definitions and
# descriptions, user may need to refer to equations (13) and Algorithm
# box 1 for details.

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
#         n:   Length of unknown signal to be recovered by phase retrieval.

# OUPTUT :
#         x0:  The initial vector to be used by any solver.

# DESCRIPTION:
#              This method uses truncation in order to remove the
#              limitations of the spectral initializer which suffers from
#              heavy tailed distributions resulting from large 4th order
#              moment generating functions. The method truncates |I|
#              largest elements of \psi and uses the corresponding
#              measurement vectors. The method additionally introduces
#              weights on the selected elements of \psi that further
#              refines the process.


# METHOD:
#         1.) Find the set 'I'. I is set of indices of |I| largest elements
#             of \psi, where |I| is the cardinality  of I. (see paragraph
#             before equation (7) on page 6 in paper).

#         2.) Using a mask R, form the matrix Y which is described in
#             detail in equation (13) of algorithm 1 of the paper.

#         3.) Compute weights W = \psi .^ gamma for the data psi whose
#             indices are in the set I. Gamma is a predetermined paramter
#             chosen by the authors in Step 1 of Algorithm 1 in the paper.

#         4.) Compute the leading eigenvector of Y (computed in step 2) and
#             scale it according to the norm of x as described in Step 3,
#             Algorithm 1.

#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

# -----------------------------START----------------------------------
import numpy as np
import struct
import math


def initWeighted(A=None, At=None, b0=None, n=None, verbose=None, *args, **kwargs):
    psi = (b0)

    # If A is a matrix, infer n and At from A
    if A.isnumeric():
        n = A.shape(2-1)
        At = lambda x=None: np.dot(A.T, x)
        A = lambda x=None: np.dot(A, x)

    # Number of measurements. Also the number of rows of A.
    m = len(psi)
    if not(verbose) or verbose:
        print(['Estimating signal of length {0} using an orthogonal '.format(
            n)+'initializer with {0} measurements...\n'.format(m)])

    # Each amplitude measurement is weighted by parameter gamma to refine
# the distribution of the measurement vectors a_m. Details given in
# algorithm box 1 of referenced paper.
    gamma = 0.5
    # Cardinality of I. I is the set that contains the indices of the
# truncated vectors. Namely, those vectors whose corresponding
# amplitude measurements is in the top 'card_S' elements of the sorted
# data vetor.
    card_I = math.floor((np.dot(3, m)) / 13)
    # STEP 1: Construct the set I of indices
    __, index_array = sort(psi, 'descend', nargout=2)

    ind = index_array(range(1, card_I))

    R = np.zeros(m, 1)
    R[ind] = 1

    W = (np.multiply(R, psi)) ** gamma

    Y = lambda x=None: At(np.multiply(W, A(x)))

    # to equation (13) in algorithm 1.

    # STEP 3: Use eigs to compute leading eigenvector of Y (Y is computed
# in previous step)
    opts = struct
    opts.isreal = False

    V, __ = eigs(Y, n, 1, 'lr', opts, nargout=2)
    alpha = lambda x=None: (np.dot(abs(A(x)).T, psi)) / \
        (np.dot(abs(A(x)).T, abs(A(x))))

    x0 = np.multiply(V, alpha(V))
    if not(verbose) or verbose:
        print('Initialization finished.\n')

    return x0
