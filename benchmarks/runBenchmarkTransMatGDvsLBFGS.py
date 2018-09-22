# Generated with SMOP  0.41
from libsmop import *
# runBenchmarkTransMatGDvsLBFGS.m

    #                  runBenchmarkTransMatGDvsLBFGS.m
# 
# This example will compare the convergence of truncated Wirtinger flow
# with regular gradient descent to trunated Wirtinger flow with L-BFGS
# acceleration.  The measurement matrix used for the comparison is the
# empirical transmission matrix.
    
    # Note: you will need to download the transmission matrix datasets to run
# this script.  See the user guide for instructions.
# 
# This script does the following:
# 
# 1. Set up parameters and create a list of algorithm structs.
    
    # 2. Invoke the general benchmark function benchmarkPR. A graph of errors
# (under specified error metrics) of different algorithms at each number of
# iterations will be shown.
    
    
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
# number of iterations.
    xitem='iterations'
# runBenchmarkTransMatGDvsLBFGS.m:35
    xvalues=concat([10,50,100,500,1000])
# runBenchmarkTransMatGDvsLBFGS.m:36
    
    yitem='reconerror'
# runBenchmarkTransMatGDvsLBFGS.m:37
    # Choose Dataset
    dataSet='transmissionMatrix'
# runBenchmarkTransMatGDvsLBFGS.m:41
    # Set up general parameters
    params.verbose = copy(false)
# runBenchmarkTransMatGDvsLBFGS.m:44
    params.numTrials = copy(20)
# runBenchmarkTransMatGDvsLBFGS.m:45
    
    params.n = copy(256)
# runBenchmarkTransMatGDvsLBFGS.m:46
    
    params.m = copy(dot(20,params.n))
# runBenchmarkTransMatGDvsLBFGS.m:47
    
    params.isComplex = copy(true)
# runBenchmarkTransMatGDvsLBFGS.m:48
    
    params.policy = copy('median')
# runBenchmarkTransMatGDvsLBFGS.m:49
    
    # Create two different versions of truncated wirtinger flow.  One using a
# steepest descent optimizer, and one using L-BFGS.  We also specify a
# 'label' for each algorithm, which is used to produce the legend of the
# plot.
    twf_sd=struct('algorithm','twf','searchMethod','steepestDescent','label','TWF-SD')
# runBenchmarkTransMatGDvsLBFGS.m:56
    twf_lbfgs=struct('algorithm','twf','searchMethod','LBFGS','label','TWF-LBFGS')
# runBenchmarkTransMatGDvsLBFGS.m:57
    # Grab your pick of algorithms.
    algorithms=cellarray([twf_sd,twf_lbfgs])
# runBenchmarkTransMatGDvsLBFGS.m:61
    # Run benchmark
    benchmarkSynthetic(xitem,xvalues,yitem,algorithms,dataSet,params)