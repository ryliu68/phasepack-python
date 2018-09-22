# Generated with SMOP  0.41
from libsmop import *
# runBenchmarkAngle.m

    #                           runBenchmarkAngle.m
# 
# This example benchmarks algorithms based on their ability to reconstruct
# a synthetic signal (random Gaussian) using synthetic measurements 
# (random Gaussian).  The benchmark shows how the
# different methods behave as the angle between the initalizer and the
# true signal changes.
# 
# This script does the following:
# 
# 1. Set up parameters and create a list of algorithm structs.
    
    # 2. Invoke the general benchmark function benchmarkPR. A graph of errors
# (under specified error metrics) of different algorithms will be shown.
    
    
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
    xitem='angle'
# runBenchmarkAngle.m:33
    xvalues=arange(0.001,pi / 2,pi / 10)
# runBenchmarkAngle.m:34
    yitem='reconerror'
# runBenchmarkAngle.m:35
    # Choose Dataset and set up dataSet '1DGaussian' specific parameters
    dataSet='1DGaussian'
# runBenchmarkAngle.m:39
    # Set up general parameters
    params.verbose = copy(false)
# runBenchmarkAngle.m:42
    params.numTrials = copy(10)
# runBenchmarkAngle.m:43
    
    params.n = copy(100)
# runBenchmarkAngle.m:44
    
    params.m = copy(dot(5,params.n))
# runBenchmarkAngle.m:45
    
    params.isComplex = copy(true)
# runBenchmarkAngle.m:46
    
    params.policy = copy('median')
# runBenchmarkAngle.m:47
    
    # Create a list of algorithms structs
    wf=struct('algorithm','wirtflow')
# runBenchmarkAngle.m:51
    twf=struct('algorithm','twf')
# runBenchmarkAngle.m:52
    rwf=struct('algorithm','rwf')
# runBenchmarkAngle.m:53
    ampflow=struct('algorithm','amplitudeflow')
# runBenchmarkAngle.m:54
    taf=struct('algorithm','taf')
# runBenchmarkAngle.m:55
    raf=struct('algorithm','raf')
# runBenchmarkAngle.m:56
    fienup=struct('algorithm','fienup')
# runBenchmarkAngle.m:57
    gs=struct('algorithm','gerchbergsaxton')
# runBenchmarkAngle.m:58
    cd=struct('algorithm','coordinatedescent','maxIters',3000)
# runBenchmarkAngle.m:59
    kac=struct('algorithm','kaczmarz','maxIters',1000)
# runBenchmarkAngle.m:60
    pmax=struct('algorithm','phasemax','maxIters',1000)
# runBenchmarkAngle.m:61
    plamp=struct('algorithm','phaselamp')
# runBenchmarkAngle.m:62
    scgm=struct('algorithm','sketchycgm')
# runBenchmarkAngle.m:63
    plift=struct('algorithm','phaselift','maxIters',1000)
# runBenchmarkAngle.m:64
    # Grab your pick of algorithms.
    algorithms=cellarray([rwf,pmax,plamp])
# runBenchmarkAngle.m:67
    # Run benchmark
    benchmarkSynthetic(xitem,xvalues,yitem,algorithms,dataSet,params)