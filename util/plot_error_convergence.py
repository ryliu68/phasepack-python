#                       plotErrorConvergence.m
#
# This function plots some convergence curve according to the values of
# options in opts specified by user. It is used in all the test*.m scripts.
# Specifically,
# If opts.recordReconErrors is True, it plots the convergence curve of
# reconstruction error versus the number of iterations.
# If opts.recordResiduals is True, it plots the convergence curve of
# residuals versus the number of iterations.
# The definition of residuals is algorithm specific. For details, see the
# specific algorithm's solve*.m file.
# If opts.recordMeasurementErrors is True, it plots the convergence curve
# of measurement errors.
#
# Inputs are as defined in the header of solvePhaseRetrieval.m.
# See it for details.
#
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

'''
function plotErrorConvergence(outs, opts)
    # Plot the error convergence curve
    if opts.recordReconErrors==True
        figure
        semilogy(outs.reconErrors)
        xlabel('Iterations')
        ylabel('ReconErrors')
        title(strcat('Convergence curve:',{' '},opts.algorithm))
    end
    if opts.recordResiduals==True
        figure
        semilogy(outs.residuals)
        xlabel('Iterations')
        ylabel('Residuals')
        title(strcat('Convergence curve:',{' '},opts.algorithm))
    end
    if opts.recordMeasurementErrors==True
        figure
        semilogy(outs.measurementErrors)
        xlabel('Iterations')
        ylabel('MeasurementErros')
        title(strcat('Convergence curve:',{' '},opts.algorithm))
    end
end
'''

import matplotlib.pyplot as plt


def plotErrorConvergence(outs=None, opts=None, *args, **kwargs):
     # Plot the error convergence curve
    if opts.recordReconErrors == True:
        fig = plt.figure()
        ax_1 = fig.add_subplot(111)
        ax_1.semilogy(outs.reconErrors)
        ax_1.set_xlabel('Iterations')
        ax_1.set_ylabel('ReconErrors')
        ax_1.set_title('Convergence curve:'+str(' ')+opts.algorithm)
        plt.show()

    if opts.recordResiduals == True:
        fig = plt.figure()
        ax_1 = fig.add_subplot(111)
        ax_1.semilogy(outs.residuals)
        ax_1.set_xlabel('Iterations')
        ax_1.set_ylabel('Residuals')
        ax_1.set_title('Convergence curve:'+str(' ')+opts.algorithm)
        plt.show()

    if opts.recordMeasurementErrors == True:
        fig = plt.figure()
        ax_1 = fig.add_subplot(111)
        ax_1.semilogy(outs.measurementErrors)
        ax_1.set_xlabel('Iterations')
        ax_1.set_ylabel('MeasurementErros')
        ax_1.set_title('Convergence curve:'+str(' ')+opts.algorithm)
        plt.show()

    return
