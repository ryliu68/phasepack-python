# Generated with SMOP  0.41
from libsmop import *
# runBenchmark2DImageRecovery.m

    #                           runBenchmark2DImageRecovery.m
# 
# This example benchmarks algorithms based on their ability to reconstruct
# an image using synthetic measurements (random Gaussian).  The benchmark 
# shows how the different methods behave as the number of Fourier masks
# increases.
# 
# 
# This script does the following: 
# 1. Set up parameters and create a list of
# algorithm structs. 
# 
# 2. Invoke the general benchmark function benchmarkPR.  This function will
# reconstruct the image using different numbers of masks, and produce a
# curve for each algorithm.
    
    
    # PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    ## -----------------------------START----------------------------------
    
    clc
    clear
    close_('all')
    ## 1.Set up parameters
# Choose x label, values shown on the x axis.  For 2D images, we use masks
# to acquire Fourier measurements.  More masks means more measurements.
    xitem='masks'
# runBenchmark2DImageRecovery.m:32
    
    xvalues=concat([2,3,4,5,6])
# runBenchmark2DImageRecovery.m:33
    
    yitem='reconerror'
# runBenchmark2DImageRecovery.m:34
    
    # Tell the benchmark function to use a 2D image
    dataSet='2DImage'
# runBenchmark2DImageRecovery.m:38
    # Set up general parameters
    params.verbose = copy(false)
# runBenchmark2DImageRecovery.m:41
    params.imagePath = copy('data/shapes.png')
# runBenchmark2DImageRecovery.m:42
    
    params.numTrials = copy(5)
# runBenchmark2DImageRecovery.m:43
    
    # Create a list of algorithms structs
    wf=struct('initMethod','spectral','algorithm','wirtflow')
# runBenchmark2DImageRecovery.m:46
    twf=struct('algorithm','twf')
# runBenchmark2DImageRecovery.m:47
    rwf=struct('algorithm','rwf')
# runBenchmark2DImageRecovery.m:48
    ampflow=struct('algorithm','amplitudeflow')
# runBenchmark2DImageRecovery.m:49
    taf=struct('initMethod','orthogonal','algorithm','taf')
# runBenchmark2DImageRecovery.m:50
    raf=struct('initMethod','weighted','algorithm','raf')
# runBenchmark2DImageRecovery.m:51
    fienup=struct('algorithm','fienup')
# runBenchmark2DImageRecovery.m:52
    gs=struct('algorithm','gerchbergsaxton')
# runBenchmark2DImageRecovery.m:53
    cd=struct('algorithm','coordinatedescent','maxIters',10000000.0)
# runBenchmark2DImageRecovery.m:54
    kac=struct('algorithm','kaczmarz','maxIters',1000)
# runBenchmark2DImageRecovery.m:55
    pmax=struct('algorithm','phasemax','maxIters',1000)
# runBenchmark2DImageRecovery.m:56
    plamp=struct('algorithm','phaselamp')
# runBenchmark2DImageRecovery.m:57
    scgm=struct('algorithm','sketchycgm')
# runBenchmark2DImageRecovery.m:58
    plift=struct('algorithm','phaselift','maxIters',1000)
# runBenchmark2DImageRecovery.m:59
    # Grab your pick of algorithms.
    algorithms=cellarray([twf,taf])
# runBenchmark2DImageRecovery.m:63
    # Run benchmark
    benchmarkSynthetic(xitem,xvalues,yitem,algorithms,dataSet,params)