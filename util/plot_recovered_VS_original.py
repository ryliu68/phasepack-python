#                       plotRecoveredVSOriginal.m
#
# This function plots the real part of the recovered signal against
# the real part of the original signal.
# It is used in all the test*.m scripts.
#
# Inputs:
#       x:  a n x 1 vector. Recovered signal.
#       xt: a n x 1 vector. Original signal.

#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

import matplotlib.pyplot as plt
import numpy as np


def plotRecoveredVSOriginal(x=None, xt=None,):
    fig = plt.figure()
    ax_1 = fig.add_subplot(111)
    ax_1.scatter(np.real(x), np.real(xt), c='b', marker='o')
    plt.plot([-3, 3], [-3, 3])
    ax_1.set_title('Visual Correlation of Recovered signal with True Signal')
    ax_1.set_xlabel('Recovered Signal')
    ax_1.set_ylabel('True Signal')
    plt.show()
    return
