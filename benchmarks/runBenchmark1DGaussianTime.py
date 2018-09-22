# Generated with SMOP  0.41
from libsmop import *
# runBenchmark1DGaussianTime.m

    #                           runBenchmark1DGaussianMN.m
# 
# This example benchmarks algorithms based on their ability to reconstruct
# a synthetic signal (random Gaussian) using synthetic measurements 
# (random Gaussian).  The benchmark shows how the
# different methods behave as the runtime increases.
    
    # This script does the following:
# 
# 1. Set up parameters and create a list of algorithm structs.
    
    # 2. Invoke the general benchmark function benchmarkPR. A graph of errors
# (under specified error metrics) of different algorithms at each level
# of allowed runtime will be shown.
    
    
    # PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    ## -----------------------------START----------------------------------
    
    clc
    clear
    close_('all')
    ## 1.Set up parameters
# Choose x label (values shown on the x axis of the benchmark plot) and 
# y label (values shown on the y-axis). The value on the x axis is the 
# runtime. The value on the y axis is 'reconerror', which is the relative
# 2-norm difference between the true and recovered signal.
    xitem='time'
# runBenchmark1DGaussianTime.m:33
    xvalues=concat([0.1,1])
# runBenchmark1DGaussianTime.m:34
    
    yitem='reconerror'
# runBenchmark1DGaussianTime.m:35
    # Choose Dataset and set up dataSet '1DGaussian' specific parameters
    dataSet='1DGaussian'
# runBenchmark1DGaussianTime.m:39
    # Set up general parameters
    params.verbose = copy(false)
# runBenchmark1DGaussianTime.m:42
    params.numTrials = copy(2)
# runBenchmark1DGaussianTime.m:43
    
    params.n = copy(500)
# runBenchmark1DGaussianTime.m:44
    
    params.m = copy(dot(4,params.n))
# runBenchmark1DGaussianTime.m:45
    
    params.isComplex = copy(true)
# runBenchmark1DGaussianTime.m:46
    
    params.policy = copy('median')
# runBenchmark1DGaussianTime.m:47
    # Create a list of algorithms structs
    wf=struct('initMethod','spectral','algorithm','wirtflow')
# runBenchmark1DGaussianTime.m:51
    twf=struct('algorithm','twf')
# runBenchmark1DGaussianTime.m:52
    rwf=struct('algorithm','rwf')
# runBenchmark1DGaussianTime.m:53
    ampflow=struct('algorithm','amplitudeflow')
# runBenchmark1DGaussianTime.m:54
    taf=struct('initMethod','orthogonal','algorithm','taf')
# runBenchmark1DGaussianTime.m:55
    raf=struct('initMethod','weighted','algorithm','raf')
# runBenchmark1DGaussianTime.m:56
    fienup=struct('algorithm','fienup')
# runBenchmark1DGaussianTime.m:57
    gs=struct('algorithm','gerchbergsaxton')
# runBenchmark1DGaussianTime.m:58
    cd=struct('algorithm','coordinatedescent','maxIters',dot(dot(300,2),params.n))
# runBenchmark1DGaussianTime.m:59
    kac=struct('algorithm','kaczmarz','maxIters',1000)
# runBenchmark1DGaussianTime.m:60
    pmax=struct('algorithm','phasemax','maxIters',1000)
# runBenchmark1DGaussianTime.m:61
    plamp=struct('algorithm','phaselamp')
# runBenchmark1DGaussianTime.m:62
    scgm=struct('algorithm','sketchycgm')
# runBenchmark1DGaussianTime.m:63
    plift=struct('algorithm','phaselift','maxIters',1000)
# runBenchmark1DGaussianTime.m:64
    # Grab your pick of algorithms.
    algorithms=cellarray([raf,fienup,ampflow,plift,pmax,plamp])
# runBenchmark1DGaussianTime.m:68
    # Run benchmark
    benchmarkSynthetic(xitem,xvalues,yitem,algorithms,dataSet,params)