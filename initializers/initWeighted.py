# Generated with SMOP  0.41
from libsmop import *
# initWeighted.m

    ## ------------------------- initWeighted.m ------------------------------
    
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
    
    
    # PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    ## -----------------------------START----------------------------------
    
    
@function
def initWeighted(A=None,At=None,b0=None,n=None,verbose=None,*args,**kwargs):
    varargin = initWeighted.varargin
    nargin = initWeighted.nargin

    psi=copy(b0)
# initWeighted.m:64
    
    # If A is a matrix, infer n and At from A
    if isnumeric(A):
        n=size(A,2)
# initWeighted.m:68
        At=lambda x=None: dot(A.T,x)
# initWeighted.m:70
        A=lambda x=None: dot(A,x)
# initWeighted.m:71
    
    # Number of measurements. Also the number of rows of A.
    m=length(psi)
# initWeighted.m:75
    if logical_not(exist('verbose','var')) or verbose:
        fprintf(concat(['Estimating signal of length %d using a weighted ','initializer with %d measurements...\n']),n,m)
    
    # Each amplitude measurement is weighted by parameter gamma to refine
# the distribution of the measurement vectors a_m. Details given in
# algorithm box 1 of referenced paper.
    gamma=0.5
# initWeighted.m:85
    # Cardinality of I. I is the set that contains the indices of the
# truncated vectors. Namely, those vectors whose corresponding
# amplitude measurements is in the top 'card_S' elements of the sorted
# data vetor.
    card_I=floor((dot(3,m)) / 13)
# initWeighted.m:92
    # STEP 1: Construct the set I of indices
    __,index_array=sort(psi,'descend',nargout=2)
# initWeighted.m:97
    
    ind=index_array(arange(1,card_I))
# initWeighted.m:98
    
    # STEP 2: Form Y
    R=zeros(m,1)
# initWeighted.m:103
    R[ind]=1
# initWeighted.m:104
    
    W=(multiply(R,psi)) ** gamma
# initWeighted.m:105
    
    Y=lambda x=None: At(multiply(W,A(x)))
# initWeighted.m:106
    
    # to equation (13) in algorithm 1.
    
    # STEP 3: Use eigs to compute leading eigenvector of Y (Y is computed
# in previous step)
    opts=copy(struct)
# initWeighted.m:112
    opts.isreal = copy(false)
# initWeighted.m:113
    
    V,__=eigs(Y,n,1,'lr',opts,nargout=2)
# initWeighted.m:114
    alpha=lambda x=None: (dot(abs(A(x)).T,psi)) / (dot(abs(A(x)).T,abs(A(x))))
# initWeighted.m:115
    
    x0=multiply(V,alpha(V))
# initWeighted.m:116
    if logical_not(exist('verbose','var')) or verbose:
        fprintf('Initialization finished.\n')
    
    return x0
    
if __name__ == '__main__':
    pass
    