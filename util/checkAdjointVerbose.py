# Generated with SMOP  0.41
from libsmop import *
# .\checkAdjointVerbose.m

    
@function
def checkAdjointVerbose(A=None,At=None,n=None,*args,**kwargs):
    varargin = checkAdjointVerbose.varargin
    nargin = checkAdjointVerbose.nargin

    if isnumeric(A):
        x=randn(n,1)
# .\checkAdjointVerbose.m:4
        Ax=dot(A,x)
# .\checkAdjointVerbose.m:5
        y=randn(numel(Ax),1)
# .\checkAdjointVerbose.m:6
        Aty=dot(At,y)
# .\checkAdjointVerbose.m:7
        inner1=dot(Ax.T,y)
# .\checkAdjointVerbose.m:9
        inner2=dot(x.T,Aty)
# .\checkAdjointVerbose.m:10
        fprintf('inner1 = %f, inner2 = %f, error = %f\n',inner1,inner2,abs(inner1 - inner2) / abs(inner1))
    else:
        x=randn(n,1)
# .\checkAdjointVerbose.m:14
        Ax=A(x)
# .\checkAdjointVerbose.m:15
        y=randn(numel(Ax),1)
# .\checkAdjointVerbose.m:16
        Aty=At(y)
# .\checkAdjointVerbose.m:17
        inner1=dot(Ax.T,y)
# .\checkAdjointVerbose.m:19
        inner2=dot(x.T,Aty)
# .\checkAdjointVerbose.m:20
        fprintf('inner1 = %f, inner2 = %f, error = %f\n',inner1,inner2,abs(inner1 - inner2) / abs(inner1))
    
    return output_args
    
if __name__ == '__main__':
    pass
    