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
#
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

# -----------------------------START----------------------------------

import struct
# 1.Set up parameters
# Choose x label, values shown on the x axis.  For 2D images, we use masks
# to acquire Fourier measurements.  More masks means more measurements.
xitem = 'masks'

xvalues = concat([2, 3, 4, 5, 6])

yitem = 'reconerror'

# Tell the benchmark function to use a 2D image
dataSet = '2DImage'
# Set up general parameters
params = struct
params.verbose = False
params.imagePath = 'data/shapes.png'

params.numTrials = 5

# Create a list of algorithms structs
wf = struct('initMethod', 'spectral', 'algorithm', 'wirtflow')
twf = struct('algorithm', 'twf')
rwf = struct('algorithm', 'rwf')
ampflow = struct('algorithm', 'amplitudeflow')
taf = struct('initMethod', 'orthogonal', 'algorithm', 'taf')
raf = struct('initMethod', 'weighted', 'algorithm', 'raf')
fienup = struct('algorithm', 'fienup')
gs = struct('algorithm', 'gerchbergsaxton')
cd = struct('algorithm', 'coordinatedescent', 'maxIters', 10000000.0)
kac = struct('algorithm', 'kaczmarz', 'maxIters', 1000)
pmax = struct('algorithm', 'phasemax', 'maxIters', 1000)
plamp = struct('algorithm', 'phaselamp')
scgm = struct('algorithm', 'sketchycgm')
plift = struct('algorithm', 'phaselift', 'maxIters', 1000)
# Grab your pick of algorithms.
algorithms = cellarray([twf, taf])
# Run benchmark
benchmarkSynthetic(xitem, xvalues, yitem, algorithms, dataSet, params)
