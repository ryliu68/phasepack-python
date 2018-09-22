# Generated with SMOP  0.41
from libsmop import *
# runBenchmarkTransmissionMatrix.m

    #                    runBenchmarkTransmissionMatrix.m
# 
# Benchmark algorthms using real/empirical data.  The measurement matrix is
# a transmission matrix, and the measurements were acquired using an
# optical aparatus. The data acquisition is described in "Coherent
# Inverse Scattering via Transmission Matrices: Efficient Phase Retrieval
# Algorithms and a Public Dataset." 
#    This script will reconstruct images using various algorithms, and
# report the image quality of the results.
    
    # See the header of transmissionMatrixExperiment.m, or the paper cited
# above for more details.
    
    # This script does the following: 
# 1. Set up parameters and create a list of
# algorithm structs. 
# 2. Invoke the general benchmark function benchmarkTransmissionMatrix.
    
    # The benchmark program compares the performance of specified algorithms on
# reconstructing an image.  By default, this script reconstructs a 16x16
# image.  However, it can also reconstruct a 40x40 or 64x64 image by
# changing the options below.  Note, the runtime and memory requirements 
# are very high when reconstructing larger images.
#
    
    # PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    ## -----------------------------START----------------------------------
    
    #clc
#clear
#close all
    
    ## 1.Set up parameters
    imageSize=64
# runBenchmarkTransmissionMatrix.m:38
    
    datasetSelection=5
# runBenchmarkTransmissionMatrix.m:39
    
    residualConstant=0.3
# runBenchmarkTransmissionMatrix.m:40
    # Create a list of algorithms structs
    wf=struct('initMethod','spectral','algorithm','wirtflow')
# runBenchmarkTransmissionMatrix.m:44
    twf=struct('algorithm','twf')
# runBenchmarkTransmissionMatrix.m:45
    rwf=struct('algorithm','rwf')
# runBenchmarkTransmissionMatrix.m:46
    ampflow=struct('algorithm','amplitudeflow')
# runBenchmarkTransmissionMatrix.m:47
    taf=struct('initMethod','orthogonal','algorithm','taf')
# runBenchmarkTransmissionMatrix.m:48
    raf=struct('initMethod','weighted','algorithm','raf')
# runBenchmarkTransmissionMatrix.m:49
    fienup=struct('algorithm','fienup')
# runBenchmarkTransmissionMatrix.m:50
    gs=struct('algorithm','gerchbergsaxton')
# runBenchmarkTransmissionMatrix.m:51
    cd=struct('algorithm','coordinatedescent','maxIters',50000)
# runBenchmarkTransmissionMatrix.m:52
    kac=struct('algorithm','kaczmarz','maxIters',1000)
# runBenchmarkTransmissionMatrix.m:53
    pmax=struct('algorithm','phasemax','maxIters',1000)
# runBenchmarkTransmissionMatrix.m:54
    plamp=struct('algorithm','phaselamp','maxIters',1000)
# runBenchmarkTransmissionMatrix.m:55
    scgm=struct('algorithm','sketchycgm')
# runBenchmarkTransmissionMatrix.m:56
    plift=struct('algorithm','phaselift','initMethod','orthogonal')
# runBenchmarkTransmissionMatrix.m:57
    # Grab your pick of algorithms.
    algorithms=cellarray([plift])
# runBenchmarkTransmissionMatrix.m:60
    ## 2. Run benchmark
    benchmarkTransmissionMatrix(imageSize,datasetSelection,residualConstant,algorithms)