#                         runBenchmark1DGaussianSNR.m
#
# This example benchmarks algorithms based on their ability to reconstruct
# a synthetic signal (random Gaussian) using synthetic measurements
# (random Gaussian).  The benchmark shows how the
# different methods behave as the signal-to-noise ratio of the
# measurements increases.
#
# The script does the following:
#
# 1. Set up parameters and create a list of algorithm structs.

# 2. Invoke the general benchmark function benchmarkPR. A graph of errors
# (under specified error metrics) of different algorithms at each level of
# SNR will be shown.
#
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017


# -----------------------------START----------------------------------

import struct
import numpy as np
# 1.Set up parameters
# Choose x label (values shown on the x axis of the benchmark plot) and
# y label (values shown on the y-axis). The value on the x axis is the SNR
# of the measurements. The value on the y axis is 'reconerror', which is
# the relative 2-norm difference between the true and recovered signal.

xitem = 'SNR'
xvalues = concat([0.1, 1, 10, 100])
yitem = 'reconerror'
# Choose Dataset and set up dataSet '1DGaussian' specific parameters
dataSet = '1DGaussian'
# Set up general parameters
params = struct
params.verbose = False
params.numTrials = 5

params.n = 50

params.m = 500

params.isComplex = True

params.policy = 'median'
# Create a list of algorithms structs
wf = struct('initMethod', 'spectral', 'algorithm', 'wirtflow')
twf = struct('algorithm', 'twf')
rwf = struct('algorithm', 'rwf')
ampflow = struct('algorithm', 'amplitudeflow')
taf = struct('initMethod', 'orthogonal', 'algorithm', 'taf')
raf = struct('initMethod', 'weighted', 'algorithm', 'raf')
fienup = struct('algorithm', 'fienup')
gs = struct('algorithm', 'gerchbergsaxton')
cd = struct('algorithm', 'coordinatedescent',
            'maxIters', np.dot(np.dot(300, 2), params.n))
kac = struct('algorithm', 'kaczmarz', 'maxIters', 1000)
pmax = struct('algorithm', 'phasemax', 'maxIters', 1000)
plamp = struct('algorithm', 'phaselamp')
scgm = struct('algorithm', 'sketchycgm')
plift = struct('algorithm', 'phaselift', 'maxIters', 1000)
# Grab your pick of algorithms.
algorithms = cellarray([gs, wf, plift, pmax, plamp])
# Run benchmark
benchmarkSynthetic(xitem, xvalues, yitem, algorithms, dataSet, params)
