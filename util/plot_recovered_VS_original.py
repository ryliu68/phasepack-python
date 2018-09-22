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

'''
function plotRecoverdVSOriginal(x,xt)
    figure
    scatter(real(x),real(xt)) hold on
    plot([-3 3], [-3 3])
    title('Visual Correlation of Recovered signal with True Signal')
    xlabel('Recovered Signal')
    ylabel('True Signal')
end
'''

import matplotlib.pyplot as plt


def plotRecoverdVSOriginal(x=None, xt=None, *args, **kwargs):
    fig = plt.figure()
    ax_1 = fig.add_subplot(111)
    ax_1.scatter(x.real, xt.real, c='r', marker='o')
    plt.plot([-3, 3], [-3, 3])
    ax_1.set_title('Visual Correlation of Recovered signal with True Signal')
    ax_1.set_xlabel('Recovered Signal')
    ax_1.set_ylabel('True Signal')
    plt.show()
    return

# if __name__ == '__main__':
#     plotRecoverdVSOriginal(3+2j, 4+5j)
