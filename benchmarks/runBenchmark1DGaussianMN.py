# Generated with SMOP  0.41
from libsmop import *
# runBenchmark1DGaussianMN.m

    #                           runBenchmark1DGaussianMN.m
# 
# This example benchmarks algorithms based on their ability to reconstruct
# a synthetic signal (random Gaussian) using synthetic measurements 
# (random Gaussian).  The benchmark shows how the
# different methods behave as the number of measurements increases.
# 
# This script does the following:
# 
# 1. Set up parameters and create a list of algorithm structs.
    
    # 2. Invoke the general benchmark function benchmarkPR. A graph of errors
# (under specified error metrics) of different algorithms will be shown.
    
    # The benchmark program compares the performance of specified algorithms on
# 1D gaussian measurements that have different m/n ratio(i.e. the ratio
# between the number of measurements and the number of unknowns).
    
    
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
# runBenchmark1DGaussianMN.m:36
    xvalues=concat([1,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,5,6])
# runBenchmark1DGaussianMN.m:37
    yitem='reconerror'
# runBenchmark1DGaussianMN.m:38
    # Choose Dataset and set up dataSet '1DGaussian' specific parameters
    dataSet='1DGaussian'
# runBenchmark1DGaussianMN.m:41
    # Set up general parameters
    params.verbose = copy(false)
# runBenchmark1DGaussianMN.m:44
    params.numTrials = copy(5)
# runBenchmark1DGaussianMN.m:45
    
    params.n = copy(100)
# runBenchmark1DGaussianMN.m:46
    
    params.isComplex = copy(true)
# runBenchmark1DGaussianMN.m:47
    
    params.policy = copy('median')
# runBenchmark1DGaussianMN.m:48
    
    # Create a list of algorithms structs
    wf=struct('algorithm','wirtflow','initMethod','spectral')
# runBenchmark1DGaussianMN.m:52
    twf=struct('algorithm','twf','initMethod','truncated')
# runBenchmark1DGaussianMN.m:53
    rwf=struct('algorithm','rwf','initMethod','weighted')
# runBenchmark1DGaussianMN.m:54
    ampflow=struct('algorithm','amplitudeflow')
# runBenchmark1DGaussianMN.m:55
    taf=struct('algorithm','taf','initMethod','truncated')
# runBenchmark1DGaussianMN.m:56
    raf=struct('initMethod','weighted','algorithm','raf')
# runBenchmark1DGaussianMN.m:57
    fienup=struct('algorithm','fienup')
# runBenchmark1DGaussianMN.m:58
    gs=struct('algorithm','gerchbergsaxton')
# runBenchmark1DGaussianMN.m:59
    cd=struct('algorithm','coordinatedescent','maxIters',3000)
# runBenchmark1DGaussianMN.m:60
    kac=struct('algorithm','kaczmarz','maxIters',1000)
# runBenchmark1DGaussianMN.m:61
    pmax=struct('algorithm','phasemax','maxIters',1000)
# runBenchmark1DGaussianMN.m:62
    plamp=struct('algorithm','phaselamp')
# runBenchmark1DGaussianMN.m:63
    scgm=struct('algorithm','sketchycgm')
# runBenchmark1DGaussianMN.m:64
    plift=struct('algorithm','phaselift','maxIters',1000)
# runBenchmark1DGaussianMN.m:65
    # Grab your pick of algorithms.
    algorithms=cellarray([wf,taf,rwf])
# runBenchmark1DGaussianMN.m:69
    # Run benchmark
    benchmarkSynthetic(xitem,xvalues,yitem,algorithms,dataSet,params)