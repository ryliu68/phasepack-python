#                     benchmarkTransmissionMatrix.m
#
# Code adapted from Sudarshan Nagesh for the reconstruction of images from
# a transmission matrix.
#
# This script will reconstruct images from phaseless measurements that were
# acquired when imaging through a diffusive media.  The quality of the
# reconstruction will be computed by comparing to a known solution, and
# different algorithms are compared.
#   Three datasets are available with resolutions 16x16, 40x40, and 64x64.
# At each resolution, there are 5 sets of measurements, each corresponding
# to a different image.  The user must select which resolution and dataset
# they want to run on.
#   As the script runs, all reconstructed images will be saved inside the
# 'benchmarkResults' folder.
#
# Note: The empirical dataset must be downloaded and installed into the
# 'data' directory for this to work. See the user guide.
#
#
# I/O
#  Inputs:
#  imageSize            : Size of image to reconstruct.  Must be {16,40,64}
#  datasetSelection     : Choose which of the sets of measurements to use.
#                           Must be in {1,2,3,4,5}.
#  residualConstant     : Only use rows of the transmission matrix that
#                           had residuals less than this cutoff. Must be
#                           between 0 and 1.  Recommend 0.4.
#  algorithms           : A cell-array of structs that describe algorithms.
#
#
# References
#  Paper Title:   Coherent inverse scattering via transmission matrices:
#  Efficient phase retrieval algorithms and a public dataset.
#  Authors:       Christopher A. Metzler,  Manoj
#  K. Sharma,  Sudarshan Nagesh,  Richard G. Baraniuk,
#                 Oliver Cossairt,  Ashok Veeraraghavan
#
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

# -----------------------------START----------------------------------


def benchmarkTransmissionMatrix(imageSize=None, datasetSelection=None, residualConstant=None, algorithms=None, *args, **kwargs):
    # Load the transmission matrix, and phaseless measurements
    A, Xt, b = loadData(residualConstant, datasetSelection,
                        imageSize, nargout=3)

    # printed after all calculations are done.
    # Results will also be printed incrementally as the benchmark runs.
    resultsTable = '\nAlgorithm  |  Measurement Error \n'

    for k in range(1, len(algorithms)):
        opts = algorithms[k]
        print('Running Algorithm: %s\n', opts.algorithm)
        reconError, measurementError = reconstructSignal(
            A, Xt, b, imageSize, datasetSelection, opts, nargout=2)
        resultsTable = concat([resultsTable, pad(
            opts.algorithm, 16), sprintf('%0.5f', measurementError), '\n'])

    # print the table
    print(resultsTable)
    return


#######################################################################
###                        Utility functions                        ###
#######################################################################

# Load the measurement operator and measurements that correspond to the
# image size and dataset chosen by the user.  This method returns the
# measurement operator 'A' (a dense matrix), a ground truth solution
# 'Xt', and a set of measurements 'Y'.

import math
import os
import numpy as np


def loadData(residualConstant=None, datasetSelection=None, imageSize=None, *args, **kwargs):

    # Create some strings for dealing with different filenames according to the dataset chosen
    if cellarray([16]) == imageSize:
        measurementType = 'AmpSLM_16x16'
    else:
        if cellarray([40]) == imageSize:
            measurementType = 'PhaseSLM_40x40'
        else:
            if cellarray([64]) == imageSize:
                measurementType = 'AmpSLM_64x64'
            else:
                error(
                    'illegal imageSize: %d. It should be chosen from {16, 40, 64}', imageSize)

    # Make sure the dataset selection is valid
    if imageSize == 16 or imageSize == 40:
        assert(datasetSelection < 5)
    else:
        if imageSize == 64:
            assert(datasetSelection < 6)

    # Load the transmission matrix, ground truth image, and measurements
    dataRoot = getFolderPath('data')

    try:
        print(concat(
            ['Loading transmission matrix (A_prVAMP.mat), this may take several minutes']))
        load(strcat(dataRoot, 'TransmissionMatrices/Coherent_Data/',
                    measurementType, '/A_prVAMP.mat'))
        print(concat(['Loading test image (XH_test.mat)']))
        load(strcat(dataRoot, '/TransmissionMatrices/Coherent_Data/',
                    measurementType, '/XH_test.mat'))
        print(concat(['Loading measurements (YH_squared_test.mat)']))
        load(strcat(dataRoot, 'TransmissionMatrices/Coherent_Data/',
                    measurementType, '/YH_squared_test.mat'))
    finally:
        pass

    # Only use the most accurate rows of the transmission matrix.  This is
    # determined by checking that the residuals found during row calcuation
    # are less than the specificed "residualConstant".
    findPixels = find(residual_vector < residualConstant)
    YH_squared_test = YH_squared_test(arange(), findPixels)
    A = A(findPixels, arange())

    Y = double(YH_squared_test(datasetSelection, arange()))
    b = math.sqrt(Y)

    # Note:  this image is real-valued
    Xt = double(XH_test(datasetSelection, arange())).T
    Xt = Xt / max(abs(Xt))

    originalImage = imresize(reshape(real(Xt), imageSize, imageSize), 4)
    originalImageName = concat(
        ['TM', str(imageSize), '-', str(datasetSelection), '-original'])
    imshow(originalImage, [])
    title(originalImageName, 'fontsize', 16)
    drawnow

    saveBenchmarkImage(originalImage, concat(
        ['TM', str(imageSize)]), originalImageName)
    return A, Xt, b


# Reconstruct a signal from a given measurement operator and set of
# measurements.  Compare the results to the ground-truth Xt, and report
# error.


def reconstructSignal(A=None, Xt=None, Y=None, imageSize=None, datasetSelection=None, opts=None, *args, **kwargs):
    print('    Reconstructing image...')

    # Convenient variables
    n = imageSize ** 2

    imageSizeStr = str(imageSize)

    ind = datasetSelection

    # Solve the PR problem
    X, outs, opts = solvePhaseRetrieval(A, [], ravel(Y), n, opts, nargout=3)

    # transform that maps the recovered solution onto the true solution
    Xrec = concat([ravel(X), ones(numel(X), 1)])
    coeffs = np.linalg.solve(Xrec, ravel(Xt))
    Xrec = np.dot(Xrec, coeffs)

    realIm = imresize(reshape(real(Xrec), imageSize, imageSize), 4)
    recoveredImageNameAbs = strcat('TM', imageSizeStr, '-', str(
        ind), '-', opts.algorithm, str(outs.iterationCount), '-real')
    imshow(realIm, [])
    title(recoveredImageNameAbs, 'fontsize', 16)
    drawnow
    realIm = realIm - min(ravel(realIm))
    realIm = realIm / max(ravel(realIm))
    saveBenchmarkImage(realIm, concat(
        ['TM', imageSizeStr]), recoveredImageNameAbs)
    reconError = norm(ravel(Xt) - ravel(Xrec)) / norm(ravel(Xt))
    measurementError = norm(abs(np.dot(A, X)) - ravel(Y)) / norm(ravel(Y))

    print('    Relative measurement error = %s\n', measurementError)
    return reconError, measurementError

# Find the path for the specified folder by searching the current and
# parent directories


def getFolderPath(fname=None, *args, **kwargs):
    dir = os.listdir(os.getcwd())
    d = dir

    if any(strcmp(fname, cellarray([d(concat([d.isdir])).name]))):
        path = concat([fname, '/'])
    else:
        # Look for folder fname in parent directory
        d = dir('../')
        if any(strcmp(fname, cellarray([d(concat([d.isdir])).name]))):
            path = concat(['../', fname, '/'])
        else:
            error(concat(['Unable to find path of folder: ', fname,
                          '.  Make sure your current directory is the PhasePack root.']))

    return path

# Save an image to a sub-folder inside 'benchmarkResults'


def saveBenchmarkImage(image=None, folder=None, fname=None, *args, **kwargs):
    bmRoot = getFolderPath('benchmarkResults')
    if not(exist(concat([bmRoot, folder]), 'dir')):
        mkdir(concat([bmRoot, folder]))

    fullName = strcat(bmRoot, folder, '/', fname, '.png')
    print('    Saving reconstructed image: %s\n', fullName)
    try:
        imwrite(image, fullName)
    finally:
        pass

    return
