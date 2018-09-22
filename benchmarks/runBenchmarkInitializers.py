# Generated with SMOP  0.41
from libsmop import *
# runBenchmarkInitializers.m

    #                   runBenchmarkInitializers.m
    
    #   This function runs an assortment of initialization algorithms on test
#   data and plots the accuracy of each method as a function of the number
#   of samples used.
    
    #   Note: for small values of m (the number of samples), the orthogonal
#   initializer might produce warnings because the spectral matrix
#   low-rank, and the smallest eigenvalue is not unique.
    
    ## User-defined parameters.
 # The dimension of the signal to reconstruct. This must be 256, 1600, or 
 # 4096 when using the transmission matrix dataset.  It can be any positive
 # integer when using Gaussian data.
    n=256
# runBenchmarkInitializers.m:17
    # A list containing the numbers of samples for which each algorithm is run
    m=dot(n,concat([0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]))
# runBenchmarkInitializers.m:19
    numTrials=10
# runBenchmarkInitializers.m:20
    
    # Select the measurement matrix to use
# Valid options: {'transmissionMatrix','gaussian'}
#measurementOperator = 'transmissionMatrix';
    measurementOperator='gaussian'
# runBenchmarkInitializers.m:25
    ## Allocate space for te results
    results=zeros(numel(m),6)
# runBenchmarkInitializers.m:28
    
    verbose=copy(false)
# runBenchmarkInitializers.m:29
    
    A=[]
# runBenchmarkInitializers.m:30
    
    ## Loop over the values of m, and store the performance of each method
    fprintf('running trials...\n')
    for m_index in arange(1,numel(m)).reshape(-1):
        fprintf('    m = %d\n',m(m_index))
        for trial in arange(1,numTrials).reshape(-1):
            # generate a random signal reconstruction problem
            if cellarray(['gaussian','synthetic']) == lower(measurementOperator):
                isComplex=copy(true)
# runBenchmarkInitializers.m:40
                isNonNegativeOnly=copy(false)
# runBenchmarkInitializers.m:41
                A,At,b0,xt,__=experimentGaussian1D(n,m(m_index),isComplex,isNonNegativeOnly,nargout=5)
# runBenchmarkInitializers.m:42
            else:
                if cellarray(['tm','transmissionmatrix']) == lower(measurementOperator):
                    A,b0,xt,plotter=experimentTransMatrixWithSynthSignal(n,m(m_index),A,nargout=4)
# runBenchmarkInitializers.m:44
                    At=[]
# runBenchmarkInitializers.m:45
                else:
                    error(concat(['Invalid dataset choice (',datatype,'): valid choices are "synthetic" and "transmissionMatrix"']))
            # original spectral
            x=initSpectral(A,At,b0,n,false,true,verbose)
# runBenchmarkInitializers.m:51
            results[m_index,1]=results(m_index,1) + abs(corr(x,xt))
# runBenchmarkInitializers.m:52
            x=initSpectral(A,At,b0,n,true,true,verbose)
# runBenchmarkInitializers.m:55
            results[m_index,2]=results(m_index,2) + abs(corr(x,xt))
# runBenchmarkInitializers.m:56
            x=initAmplitude(A,At,b0,n,verbose)
# runBenchmarkInitializers.m:59
            results[m_index,3]=results(m_index,3) + abs(corr(x,xt))
# runBenchmarkInitializers.m:60
            x=initWeighted(A,At,b0,n,verbose)
# runBenchmarkInitializers.m:63
            results[m_index,4]=results(m_index,4) + abs(corr(x,xt))
# runBenchmarkInitializers.m:64
            x=initOptimalSpectral(A,At,b0,n,true,verbose)
# runBenchmarkInitializers.m:67
            results[m_index,5]=results(m_index,5) + abs(corr(x,xt))
# runBenchmarkInitializers.m:68
            x=initOrthogonal(A,At,b0,n,verbose)
# runBenchmarkInitializers.m:71
            results[m_index,6]=results(m_index,6) + abs(corr(x,xt))
# runBenchmarkInitializers.m:72
    
    results=results / numTrials
# runBenchmarkInitializers.m:77
    names=cellarray(['spectral','truncated','amplitude','weighted','optimal','orthogonal'])
# runBenchmarkInitializers.m:78
    autoplot(m,results / numTrials,names)
    xlabel('number of samples','fontsize',16)
    ylabel('cosine similarity','fontsize',16)
    title(concat(['initializer accuracy vs number of sample: n=',num2str(n)]))
    l=legend('show','Location','northeastoutside')
# runBenchmarkInitializers.m:83
    
    set(l,'fontsize',16)