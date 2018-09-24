#                           runBenchmarkSuccessRate.m
#
# This example benchmarks algorithms based on their ability to reconstruct
# a synthetic signal (random Gaussian) using synthetic measurements
# (random Gaussian).  The benchmark shows how the
# different methods behave as the number of measurements increases.  The
# y-axis plots the success rate (rate of exact signal recovery).  This is
# done by setting params.policy='successrate'.
#
# The algorithms currently used in this implementation are all instances of
# PhaseMax, but with different levels of accuracy in the initializer.  To
# control the level of initialization accuracy, we set the initializer to
# "angle", and specify the angle between the true signal, and the initial
# guess.
#
# This script does the following:
#
# 1. Set up parameters and create a list of algorithm structs.
#
# 2. Invoke the general benchmark function benchmarkPR. A graph of errors
# (under specified error metrics) of different algorithms will be shown.
#
# The benchmark program compares the performance of specified algorithms on
# 1D gaussian measurements that have different m/n ratio(i.e. the ratio
# between the number of measurements and the number of unknowns).
#
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

# -----------------------------START----------------------------------

import struct
import numpy as np
import math
from benchmar_synthetic import benchmarkSynthetic
# 1.Set up parameters
# Choose x label (values shown on the x axis of the benchmark plot) and
# y label (values shown on the y-axis). The value on the x axis is the m/n
# ratio. The value on the y axis is 'reconerror', which is the relative
# 2-norm difference between the true and recovered signal.
xitem = 'm/n'
xvalues = np.arange(2, 12, 0.5)
yitem = 'reconerror'
# Choose Dataset and set up dataSet '1DGaussian' specific parameters
dataSet = '1DGaussian'
# Set up general parameters
params = struct
params.verbose = False
params.numTrials = 20
params.n = 500
params.isComplex = True
params.policy = 'successrate'
params.successConstant = 0.0001
# runBenchmarkSuccessRate.m:57
#  Each of these algorithm is an instance of PhaseMax.  However, they each
#  are initialized with starting points of different accuracies.  The
#  "angle" initializer grabs the "initAngle" entry from the options, and
#  produces an initializer that makes this angle with the true signal.
pmax25 = struct('algorithm', 'phasemax', 'initMethod', 'angle')
pmax25.tol = 1e-06
pmax25.initAngle = np.dot(np.dot(25 / 360, 2), math.pi)
pmax36 = struct('algorithm', 'phasemax', 'initMethod', 'angle')
pmax36.tol = 1e-06
pmax36.initAngle = np.dot(np.dot(36 / 360, 2), math.pi)
pmax45 = struct('algorithm', 'phasemax', 'initMethod', 'angle')
pmax45.tol = 1e-06
pmax45.initAngle = np.dot(np.dot(45 / 360, 2), math.pi)
# Grab your math.pick of algorithms.
algorithms = cellarray([pmax25, pmax36, pmax45])
# Run benchmark
results = benchmarkSynthetic(
    xitem, xvalues, yitem, algorithms, dataSet, params)
