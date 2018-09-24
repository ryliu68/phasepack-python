#                        experimentImage2D.m
#
# Create a Fourier measurement operator and measurements for reconstructing
# an image.
#
# Inputs:
# numMasks: number of random octanary Fourier masks.
# imagePath: The path of the image to be recovered
#
# Outputs:
#  A     : A function handle: n1*n2 x 1 -> n1*n2*numMasks x 1. It returns
#          A*x.
#  At    : The transpose of A
#  b0    : A n1*n2*numMasks x 1 real, non-negative vector consists of the
#          measurements abs(A*x).
#  Xt    : The true signal - a vectorization of the image
#
#
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

# -----------------------------START----------------------------------

import numpy as np
import math


def experimentImage2D(numMasks=None, imagePath=None, *args, **kwargs):
    # Read Image
    # Below Xt is n1 x n2 x 3; i.e. we have three n1 x n2 images, one for each of the 3 color channels
    image = rgb2gray(imread(concat([imagePath])))
    image = double(image)
    dims = np.shape(image)
    L = numMasks

    # Each mask has iid entries following the octanary pattern.
    # Entries have the form b = b1*b2 , where b1 is sampled
    # from {-1, 1, -i, i} with equal probability 1/4, and b2 from
    # { sqrt(1/2), sqrt(3)} with probability 4/5 and 1/5 respectively.
    b1 = concat([[- 1], [1], [- 1j], [1j]])
    b2 = concat([[repmat(sqrt(0.5), 4, 1)], [math.sqrt(3)]])

    masks = np.multiply(b1(randi(4, concat([dims, L]))), b2(
        randi(5, concat([dims, L]))))

    # Make linear operators that act on a vectorized image
    A = lambda x=None: fourierMeasurementOperator(x, masks, dims)
    At = lambda y=None: transposeOperator(y, masks, dims)
    Xt = ravel(image)
    b0 = abs(A(ravel(Xt)))

    subplot(1, 2, 1)
    imagesc(image)
    title('original')
    subplot(1, 2, 2)
    plotter = lambda x=None: imagesc(reshape(abs(x), dims))
    return A, At, b0, Xt, plotter


def fourierMeasurementOperator(x=None, masks=None, dims=None, *args, **kwargs):
    x = np.reshape(x, dims)

    n1, n2 = np.shape(x, nargout=2)
    L = np.shape(masks, 3)

    # Compute measurements
    copies = repmat(x, concat([1, 1, L]))
    y = fft2(np.multiply(masks, copies))
    y = ravel(y)
    return y


def transposeOperator(y=None, masks=None, dims=None, *args, **kwargs):
    n1 = dims(1)
    n2 = dims(2)
    L = np.shape(masks, 3)

    y = reshape(y, concat([n1, n2, L]))

    x = np.multiply(dot(dot(n1, n2), ifft2(y)), conj(masks))
    x = sum(x, 3)
    x = ravel(x)
    return x
