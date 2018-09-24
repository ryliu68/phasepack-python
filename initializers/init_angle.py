# ----------------------------initAngle.m-----------------------------

# Given the true solution of a phase retrieval problem, this method
# produces a random initialization that makes the specified angle with that
# solution.  This routine is meant to be used for benchmarking only; it
# enables the user to investigate the sensitivity of different methods on
# the accuracy of the initializer.
#
# I/O
#  Inputs:
#     Xtrue:  a vector
#     theta: an angle, specified in radians.
# returns
#     x0: a random vector oriented with the specified angle relative to
#         Xtrue.
#
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

# -----------------------------START----------------------------------

import numpy as np
import math


def initAngle(xt=None, theta=None, *args, **kwargs):
    # To get the correct angle, we add a perturbation to Xtrue.
    # Start by producing a random direction.
    d = np.random.randn(np.size(xt))
    # Orthogonalize the random direction
    d = d - np.dot((np.dot(d.T, xt)) / math.sqrt(np.dot(xt, xt)) ** 2, xt)
    # Re-scale to have same norm as the signal
    d = np.dot(d / math.sqrt(np.dot(d, d)), math.sqrt(np.dot(xt, xt)))
    # Add just enough perturbation to get the correct angle
    x0 = xt + np.dot(d, math.tan(theta))
    # Normalize
    x0 = np.dot(x0 / math.sqrt(np.dot(x0, x0)), math.sqrt(np.dot(xt, xt)))

    return x0
