#                               initializeContainers.m
#
# This function initializes and outputs containers for convergence info
# according to user's choice. It is invoked in solve*.m.
#
# Inputs:
#         opts(struct)              :  consists of options.
# Outputs:
#         solveTimes(struct)        :  empty [] or initialized with
#                                      opts.maxIters x 1 np.zeros if
#                                      recordTimes.
#         measurementErrors(struct) :  empty [] or initialized with
#                                      opts.maxIters x 1 np.zeros if
#                                      recordMeasurementErrors.
#         reconErrors(struct)       :  empty [] or initialized with
#                                      opts.maxIters x 1 np.zeros if
#                                      recordReconErrors.
#         residuals(struct)         :  empty [] or initialized with
#                                      opts.maxIters x 1 np.zeros if
#                                      recordResiduals.
#
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

'''
function [solveTimes,measurementErrors,reconErrors,residuals] = initializeContainers(opts)
    solveTimes = []
    measurementErrors = []
    reconErrors = []
    residuals = []
    if opts.recordTimes
        solveTimes = zeros(opts.maxIters,1)
    end
    if opts.recordMeasurementErrors
        measurementErrors = zeros(opts.maxIters,1)
    end
    if opts.recordReconErrors
        reconErrors = zeros(opts.maxIters,1)
    end
    if opts.recordResiduals
        residuals = zeros(opts.maxIters,1)
    end
end
'''
import numpy as np


def initializeContainers(opts=None, *args, **kwargs):
    solveTimes = []
    measurementErrors = []
    reconErrors = []
    residuals = []
    if opts.recordTimes:
        solveTimes = np.zeros(opts.maxIters, 1)

    if opts.recordMeasurementErrors:
        measurementErrors = np.zeros(opts.maxIters, 1)

    if opts.recordReconErrors:
        reconErrors = np.zeros(opts.maxIters, 1)

    if opts.recordResiduals:
        residuals = np.zeros(opts.maxIters, 1)

    return solveTimes, measurementErrors, reconErrors, residuals

# if __name__ == '__main__':
#     pass
