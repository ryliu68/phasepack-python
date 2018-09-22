# Generated with SMOP  0.41
from libsmop import *
# .\checkIfNumber.m

    #                               checkIfNumber.m
    
    # This function checks if the input parameter num is a 
# number. If it is not, an error will be raised. It is used to check 
# the validity of an numerical input
# (e.g. It is used to check opts.maxInnerIters in the function 
# validateOptions in solveGerchbergSaxton.m).
    
    # Inputs:
#         name(string)       :  The name of the variable to be checked.
#         num(integer)       :  value of the variable to be checked.
#  
# 
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    
@function
def checkIfNumber(name=None,num=None,*args,**kwargs):
    varargin = checkIfNumber.varargin
    nargin = checkIfNumber.nargin

    if logical_not((logical_and(isnumeric(num),numel(num)) == 1)):
        error('%s must be a number!!\n',name)
    
    return
    
if __name__ == '__main__':
    pass
    