import numpy as np


def initializeContainers(opts=None, *args, **kwargs):
    solveTimes = []
    measurementErrors = []
    reconErrors = []
    residuals = []

    if opts.recordTimes:
        solveTimes = np.zeros((opts.maxIters, 1))

    if opts.recordMeasurementErrors:
        measurementErrors = np.zeros(opts.maxIters, 1)

    if opts.recordReconErrors:
        reconErrors = np.zeros(opts.maxIters, 1)

    if opts.recordResiduals:
        residuals = np.zeros((opts.maxIters, 1))

    return solveTimes, measurementErrors, reconErrors, residuals
