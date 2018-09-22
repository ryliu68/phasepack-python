# Generated with SMOP  0.41
from libsmop import *
# runBenchmarkTransMatMN.m

    #                        runBenchmarkTransMatMN.m
# 
# This example benchmarks algorithms based on their ability to reconstruct
# a synthetic signal (random Gaussian) using measurements acquired from a
# real data matrix (a transmission matrix).  The benchmark shows how the
# different methods behave as the number of rows sampled from the
# transmission matrix (i.e., the number 'm' of measurements) increases.
    
    # This benchmark looks at the behavior of the methods using a *real*
# empirical measurement matrix, but using synthetic signals so that
# reconstruction accuracy can be measured directly.
    
    # Note:  You will need to download the empirical datasets before this
# benchmark can be run.  See the user guide.
# 
# Expected time to run: 5mins
    
    # The script does the following:
# 
# 1. Set up parameters and create a list of algorithm structs.
    
    # 2. Invoke the general benchmark function benchmarkPR. A graph of errors
# (under specified error metrics) of different algorithms will be shown as
# the number of measurements varies.
    
    
    # PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    ## -----------------------------START----------------------------------
    
    clc
    clear
    close_('all')
    ## 1.Set up parameters
# Choose x label (values shown on the x axis of the benchmark plot) and 
# y label (values shown on the y-axis). The value on the x axis is the m/n
# ratio. The value on the y axis is 'reconerror', which is the relative
# 2-norm difference between the true and recovered signal.
    xitem='m/n'
# runBenchmarkTransMatMN.m:43
    
    xvalues=concat([1,1.5,2,2.5,3,3.5,4,4.5,5,7.5,10,12.5,15])
# runBenchmarkTransMatMN.m:44
    
    yitem='reconerror'
# runBenchmarkTransMatMN.m:45
    
    # Choose dataset: in this case we use the empirical transmission matrix
    dataSet='transmissionMatrix'
# runBenchmarkTransMatMN.m:48
    # Set up general parameters
    params.verbose = copy(false)
# runBenchmarkTransMatMN.m:51
    params.numTrials = copy(10)
# runBenchmarkTransMatMN.m:52
    
    params.n = copy(256)
# runBenchmarkTransMatMN.m:53
    
    # transmissionMatrix dataset, this can be one of {246,1600,4096}
    params.policy = copy('median')
# runBenchmarkTransMatMN.m:55
    
    # Create a list of algorithms structs
# You can specify the algorithm name, the initializer, and other options
# for each method in the struct.
    wf=struct('initMethod','spectral','algorithm','wirtflow')
# runBenchmarkTransMatMN.m:61
    twf=struct('algorithm','twf')
# runBenchmarkTransMatMN.m:62
    rwf=struct('algorithm','rwf')
# runBenchmarkTransMatMN.m:63
    ampflow=struct('algorithm','amplitudeflow')
# runBenchmarkTransMatMN.m:64
    taf=struct('initMethod','orthogonal','algorithm','taf')
# runBenchmarkTransMatMN.m:65
    raf=struct('initMethod','weighted','algorithm','raf')
# runBenchmarkTransMatMN.m:66
    fienup=struct('algorithm','fienup')
# runBenchmarkTransMatMN.m:67
    gs=struct('algorithm','gerchbergsaxton')
# runBenchmarkTransMatMN.m:68
    cd=struct('algorithm','coordinatedescent','maxIters',3000)
# runBenchmarkTransMatMN.m:69
    kac=struct('algorithm','kaczmarz','maxIters',1000)
# runBenchmarkTransMatMN.m:70
    pmax=struct('algorithm','phasemax','maxIters',1000)
# runBenchmarkTransMatMN.m:71
    plamp=struct('algorithm','phaselamp')
# runBenchmarkTransMatMN.m:72
    scgm=struct('algorithm','sketchycgm')
# runBenchmarkTransMatMN.m:73
    plift=struct('algorithm','phaselift','maxIters',1000)
# runBenchmarkTransMatMN.m:74
    # Grab your pick of algorithms to benchmark.
    algorithms=cellarray([wf,raf,fienup,gs,pmax,plamp,plift])
# runBenchmarkTransMatMN.m:78
    # Run benchmark
    benchmarkSynthetic(xitem,xvalues,yitem,algorithms,dataSet,params)