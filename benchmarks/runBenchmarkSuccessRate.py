# Generated with SMOP  0.41
from libsmop import *
# runBenchmarkSuccessRate.m

    #                           runBenchmarkSuccessRate.m
# 
# This example benchmarks algorithms based on their ability to reconstruct
# a synthetic signal (random Gaussian) using synthetic measurements 
# (random Gaussian).  The benchmark shows how the
# different methods behave as the number of measurements increases.  The
# y-axis plots the success rate (rate of exact signal recovery).  This is
# done by setting params.policy='successrate'.
    
    # The algorithms currently used in this implementation are all instances of
# PhaseMax, but with different levels of accuracy in the initializer.  To
# control the level of initialization accuracy, we set the initializer to
# "angle", and specify the angle between the true signal, and the initial
# guess.
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
# runBenchmarkSuccessRate.m:44
    xvalues=arange(2,12,0.5)
# runBenchmarkSuccessRate.m:45
    yitem='reconerror'
# runBenchmarkSuccessRate.m:46
    # Choose Dataset and set up dataSet '1DGaussian' specific parameters
    dataSet='1DGaussian'
# runBenchmarkSuccessRate.m:49
    # Set up general parameters
    params.verbose = copy(false)
# runBenchmarkSuccessRate.m:52
    params.numTrials = copy(20)
# runBenchmarkSuccessRate.m:53
    
    params.n = copy(500)
# runBenchmarkSuccessRate.m:54
    
    params.isComplex = copy(true)
# runBenchmarkSuccessRate.m:55
    
    params.policy = copy('successrate')
# runBenchmarkSuccessRate.m:56
    
    params.successConstant = copy(0.0001)
# runBenchmarkSuccessRate.m:57
    #  Each of these algorithm is an instance of PhaseMax.  However, they each 
#  are initialized with starting points of different accuracies.  The 
#  "angle" initializer grabs the "initAngle" entry from the options, and
#  produces an initializer that makes this angle with the true signal.
    pmax25=struct('algorithm','phasemax','initMethod','angle')
# runBenchmarkSuccessRate.m:63
    pmax25.tol = copy(1e-06)
# runBenchmarkSuccessRate.m:64
    pmax25.initAngle = copy(dot(dot(25 / 360,2),pi))
# runBenchmarkSuccessRate.m:65
    pmax36=struct('algorithm','phasemax','initMethod','angle')
# runBenchmarkSuccessRate.m:67
    pmax36.tol = copy(1e-06)
# runBenchmarkSuccessRate.m:68
    pmax36.initAngle = copy(dot(dot(36 / 360,2),pi))
# runBenchmarkSuccessRate.m:69
    pmax45=struct('algorithm','phasemax','initMethod','angle')
# runBenchmarkSuccessRate.m:71
    pmax45.tol = copy(1e-06)
# runBenchmarkSuccessRate.m:72
    pmax45.initAngle = copy(dot(dot(45 / 360,2),pi))
# runBenchmarkSuccessRate.m:73
    # Grab your pick of algorithms.
    algorithms=cellarray([pmax25,pmax36,pmax45])
# runBenchmarkSuccessRate.m:76
    # Run benchmark
    results=benchmarkSynthetic(xitem,xvalues,yitem,algorithms,dataSet,params)
# runBenchmarkSuccessRate.m:80