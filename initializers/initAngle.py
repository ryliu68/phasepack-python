# Generated with SMOP  0.41
from libsmop import *
# initAngle.m

    ## ----------------------------initAngle.m-----------------------------
    
    # Given the true solution of a phase retrieval problem, this method
# produces a random initialization that makes the specified angle with that
# solution.  This routine is meant to be used for benchmarking only; it
# enables the user to investigate the sensitivity of different methods on
# the accuracy of the initializer.
#  
## I/O
#  Inputs:
#     Xtrue:  a vector
#     theta: an angle, specified in radians.
## returns
#     x0: a random vector oriented with the specified angle relative to
#         Xtrue.
#
    
    # PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    ## -----------------------------START----------------------------------
    
    
@function
def initAngle(xt=None,theta=None,*args,**kwargs):
    varargin = initAngle.varargin
    nargin = initAngle.nargin

    # To get the correct angle, we add a perturbation to Xtrue.  
# Start by producing a random direction.
    d=randn(size(xt))
# initAngle.m:29
    # Orthogonalize the random direction
    d=d - dot((dot(d.T,xt)) / norm(xt) ** 2,xt)
# initAngle.m:32
    # Re-scale to have same norm as the signal
    d=dot(d / norm(d),norm(xt))
# initAngle.m:35
    # Add just enough perturbation to get the correct angle
    x0=xt + dot(d,tan(theta))
# initAngle.m:38
    # Normalize
    x0=dot(x0 / norm(x0),norm(xt))
# initAngle.m:41
    return x0
    
if __name__ == '__main__':
    pass
    