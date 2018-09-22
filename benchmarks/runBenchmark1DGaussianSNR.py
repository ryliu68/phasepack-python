# Generated with SMOP  0.41
from libsmop import *
# runBenchmark1DGaussianSNR.m

    #                         runBenchmark1DGaussianSNR.m
    
    # This example benchmarks algorithms based on their ability to reconstruct
# a synthetic signal (random Gaussian) using synthetic measurements 
# (random Gaussian).  The benchmark shows how the
# different methods behave as the signal-to-noise ratio of the 
# measurements increases.
    
    # The script does the following:
# 
# 1. Set up parameters and create a list of algorithm structs.
    
    # 2. Invoke the general benchmark function benchmarkPR. A graph of errors
# (under specified error metrics) of different algorithms at each level of
# SNR will be shown.
    
    
    # PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    ## -----------------------------START----------------------------------
    
    clc
    clear
    close_('all')
    ## 1.Set up parameters
# Choose x label (values shown on the x axis of the benchmark plot) and 
# y label (values shown on the y-axis). The value on the x axis is the SNR
# of the measurements. The value on the y axis is 'reconerror', which is 
# the relative 2-norm difference between the true and recovered signal.
    
    xitem='SNR'
# runBenchmark1DGaussianSNR.m:36
    xvalues=concat([0.1,1,10,100])
# runBenchmark1DGaussianSNR.m:37
    yitem='reconerror'
# runBenchmark1DGaussianSNR.m:38
    # Choose Dataset and set up dataSet '1DGaussian' specific parameters
    dataSet='1DGaussian'
# runBenchmark1DGaussianSNR.m:42
    # Set up general parameters
    params.verbose = copy(false)
# runBenchmark1DGaussianSNR.m:45
    params.numTrials = copy(5)
# runBenchmark1DGaussianSNR.m:46
    
    params.n = copy(50)
# runBenchmark1DGaussianSNR.m:47
    
    params.m = copy(500)
# runBenchmark1DGaussianSNR.m:48
    
    params.isComplex = copy(true)
# runBenchmark1DGaussianSNR.m:49
    
    params.policy = copy('median')
# runBenchmark1DGaussianSNR.m:50
    # Create a list of algorithms structs
    wf=struct('initMethod','spectral','algorithm','wirtflow')
# runBenchmark1DGaussianSNR.m:54
    twf=struct('algorithm','twf')
# runBenchmark1DGaussianSNR.m:55
    rwf=struct('algorithm','rwf')
# runBenchmark1DGaussianSNR.m:56
    ampflow=struct('algorithm','amplitudeflow')
# runBenchmark1DGaussianSNR.m:57
    taf=struct('initMethod','orthogonal','algorithm','taf')
# runBenchmark1DGaussianSNR.m:58
    raf=struct('initMethod','weighted','algorithm','raf')
# runBenchmark1DGaussianSNR.m:59
    fienup=struct('algorithm','fienup')
# runBenchmark1DGaussianSNR.m:60
    gs=struct('algorithm','gerchbergsaxton')
# runBenchmark1DGaussianSNR.m:61
    cd=struct('algorithm','coordinatedescent','maxIters',dot(dot(300,2),params.n))
# runBenchmark1DGaussianSNR.m:62
    kac=struct('algorithm','kaczmarz','maxIters',1000)
# runBenchmark1DGaussianSNR.m:63
    pmax=struct('algorithm','phasemax','maxIters',1000)
# runBenchmark1DGaussianSNR.m:64
    plamp=struct('algorithm','phaselamp')
# runBenchmark1DGaussianSNR.m:65
    scgm=struct('algorithm','sketchycgm')
# runBenchmark1DGaussianSNR.m:66
    plift=struct('algorithm','phaselift','maxIters',1000)
# runBenchmark1DGaussianSNR.m:67
    # Grab your pick of algorithms.
    algorithms=cellarray([gs,wf,plift,pmax,plamp])
# runBenchmark1DGaussianSNR.m:71
    # Run benchmark
    benchmarkSynthetic(xitem,xvalues,yitem,algorithms,dataSet,params)