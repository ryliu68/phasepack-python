# Generated with SMOP  0.41
from libsmop import *
# .\plotRecoveredVSOriginal.m

    #                       plotRecoveredVSOriginal.m
    
    # This function plots the real part of the recovered signal against
# the real part of the original signal.
# It is used in all the test*.m scripts.
    
    # Inputs:
#       x:  a n x 1 vector. Recovered signal.
#       xt: a n x 1 vector. Original signal.
    
    # 
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    
@function
def plotRecoverdVSOriginal(x=None,xt=None,*args,**kwargs):
    varargin = plotRecoverdVSOriginal.varargin
    nargin = plotRecoverdVSOriginal.nargin

    figure
    scatter(real(x),real(xt))
    hold('on')
    plot(concat([- 3,3]),concat([- 3,3]))
    title('Visual Correlation of Recovered signal with True Signal')
    xlabel('Recovered Signal')
    ylabel('True Signal')
    return
    
if __name__ == '__main__':
    pass
    