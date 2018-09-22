# Generated with SMOP  0.41
from libsmop import *
# .\autoplot.m

    #                             autoplot.m
    
    # This function generates a line plot from x-axis and y-axis data.  It
# automatically selects either a linear or log axis, and it chooses colors
# and markers for the lines to make them easily distinguisable.
    
    # inputs:
#       xvals:  a column vector of horizontal axis data
#       yvals:  a matrix, each column of which contains the data to
#                   generate a line of the plot
#       curveNames:  a cell array of string containing the name of each
#               curve to appear in the legend.
    
    
@function
def autoplot(xvals=None,yvals=None,curveNames=None,*args,**kwargs):
    varargin = autoplot.varargin
    nargin = autoplot.nargin

    ## figure out what kind of axis to use
# Measure correlation between x/y values and a counter.  Also measure
# correlation with the log-values.
    count=(arange(1,numel(xvals))).T
# .\autoplot.m:20
    xLinCorr=corrcoef(concat([count,ravel(xvals)]))
# .\autoplot.m:21
    xLinCorr=min(abs(ravel(xLinCorr)))
# .\autoplot.m:21
    xLogCorr=corrcoef(concat([count,log(ravel(xvals))]))
# .\autoplot.m:22
    xLogCorr=min(abs(ravel(xLogCorr)))
# .\autoplot.m:22
    yLinCorr=corrcoef(concat([count,yvals]))
# .\autoplot.m:23
    yLinCorr=min(abs(ravel(yLinCorr)))
# .\autoplot.m:23
    yLogCorr=corrcoef(concat([count,log(yvals)]))
# .\autoplot.m:24
    yLogCorr=min(abs(ravel(yLogCorr)))
# .\autoplot.m:24
    # Pick the type of axis that produces the largest correlation
    useLogY=min(ravel(yvals)) > 0 and yLogCorr > yLinCorr
# .\autoplot.m:26
    useLogX=min(ravel(xvals)) > 0 and xLogCorr > xLinCorr
# .\autoplot.m:27
    ## Choose plotting function with appropriate axes
    if useLogY and useLogX:
        myplot=lambda x=None,y=None,optName=None,optVal=None: loglog(x,y,optName,optVal)
# .\autoplot.m:31
    else:
        if useLogY and logical_not(useLogX):
            myplot=lambda x=None,y=None,optName=None,optVal=None: semilogy(x,y,optName,optVal)
# .\autoplot.m:33
        else:
            if logical_not(useLogY) and useLogX:
                myplot=lambda x=None,y=None,optName=None,optVal=None: semilogx(x,y,optName,optVal)
# .\autoplot.m:35
            else:
                myplot=lambda x=None,y=None,optName=None,optVal=None: plot(x,y,optName,optVal)
# .\autoplot.m:37
    
    ## Plot stuff
    markers=cellarray(['.','o','x','*','s','d','v','p','h','+'])
# .\autoplot.m:41
    lineTypes=cellarray(['-','-.','--'])
# .\autoplot.m:42
    nm=numel(markers)
# .\autoplot.m:44
    nl=numel(lineTypes)
# .\autoplot.m:45
    figure
    grid('on')
    for k in arange(1,length(curveNames)).reshape(-1):
        h=myplot(xvals,yvals(arange(),k),'DisplayName',curveNames[k])
# .\autoplot.m:49
        set(h,'linewidth',1.75,'LineStyle',lineTypes[mod(k - 1,nl) + 1],'Marker',markers[mod(k - 1,nm) + 1])
        hold('on')
        grid('on')
    
    return
    
if __name__ == '__main__':
    pass
    