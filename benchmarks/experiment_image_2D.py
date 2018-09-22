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

'''
function [A, At, b0, Xt, plotter] = experimentImage2D(numMasks, imagePath)
    ##  Read Image 
    # Below Xt is n1 x n2 x 3 i.e. we have three n1 x n2 images, one for each of the 3 color channels  
    image = rgb2gray(imread([imagePath]))
    image = double(image)
    dims = size(image)
    L = numMasks

    ## Make octanary masks and linear sampling operators  
    # Each mask has iid entries following the octanary pattern.
    # Entries have the form b = b1*b2 , where b1 is sampled
    # from {-1, 1, -i, i} with equal probability 1/4, and b2 from
    # { sqrt(1/2), sqrt(3)} with probability 4/5 and 1/5 respectively.
    b1 = [-11-1i1i]
    b2 = [repmat(sqrt(0.5),4,1) sqrt(3)]
    # Masks has size [n1,n2,L]
    masks = b1(randi(4,[dims,L])) .* b2(randi(5,[dims,L])) # Storage for L masks, each of dim n1 x n2
    
    ## Make linear operators that act on a vectorized image
    A = @(x) fourierMeasurementOperator(x, masks, dims)
    At = @(y) transposeOperator(y, masks, dims)
    
  
    Xt = image(:)
    b0 = abs(A(Xt(:)))
    
    # Set up plotting
    subplot(1,2,1)
    imagesc(image)
    title('original')
    subplot(1,2,2)
    
    plotter = @(x) imagesc(reshape(abs(x),dims))

end
'''
import numpy as np


def experimentImage2D(numMasks=None, imagePath=None, *args, **kwargs):
    # Read Image
    # Below Xt is n1 x n2 x 3; i.e. we have three n1 x n2 images, one for each of the 3 color channels
    image = rgb2gray(imread(concat([imagePath])))
# experimentImage2D.m:30
    image = double(image)
# experimentImage2D.m:31
    dims = np.shape(image)
# experimentImage2D.m:32
    L = numMasks
# experimentImage2D.m:33

    # Each mask has iid entries following the octanary pattern.
    # Entries have the form b = b1*b2 , where b1 is sampled
    # from {-1, 1, -i, i} with equal probability 1/4, and b2 from
    # { sqrt(1/2), sqrt(3)} with probability 4/5 and 1/5 respectively.
    b1 = concat([[- 1], [1], [- 1j], [1j]])
# experimentImage2D.m:40
    b2 = concat([[repmat(sqrt(0.5), 4, 1)], [sqrt(3)]])
# experimentImage2D.m:41

    masks = np.multiply(b1(randi(4, concat([dims, L]))), b2(
        randi(5, concat([dims, L]))))
# experimentImage2D.m:43

    # Make linear operators that act on a vectorized image
    A = lambda x=None: fourierMeasurementOperator(x, masks, dims)
# experimentImage2D.m:46
    At = lambda y=None: transposeOperator(y, masks, dims)
# experimentImage2D.m:47
    Xt = ravel(image)
# experimentImage2D.m:50
    b0 = abs(A(ravel(Xt)))
# experimentImage2D.m:51

    subplot(1, 2, 1)
    imagesc(image)
    title('original')
    subplot(1, 2, 2)
    plotter = lambda x=None: imagesc(reshape(abs(x), dims))
# experimentImage2D.m:59
    return A, At, b0, Xt, plotter


'''
function y = fourierMeasurementOperator(x, masks, dims)
    x = reshape(x, dims)   # image comes in as a vector.  Reshape to rectangle
    [n1,n2] = size(x)
    L = size(masks,3)              # get number of masks

    # Compute measurements
    copies = repmat(x,[1,1,L])
    y = fft2(masks.*copies)
    y = y(:)
end
'''


def fourierMeasurementOperator(x=None, masks=None, dims=None, *args, **kwargs):
    x = np.reshape(x, dims)
# experimentImage2D.m:64

    n1, n2 = np.shape(x, nargout=2)
# experimentImage2D.m:65
    L = np.shape(masks, 3)
# experimentImage2D.m:66

    # Compute measurements
    copies = repmat(x, concat([1, 1, L]))
# experimentImage2D.m:69
    y = fft2(np.multiply(masks, copies))
# experimentImage2D.m:70
    y = ravel(y)
# experimentImage2D.m:71
    return y

    '''

function x = transposeOperator(y, masks, dims)
    n1 = dims(1)
    n2 = dims(2)
    L = size(masks,3)              # get number of masks
    y = reshape(y, [n1,n2,L])   # image comes in as a vector.  Reshape to rectangle
    
    x = n1*n2*ifft2(y).*conj(masks)
    x = sum(x,3)
    x = x(:)
end

'''


def transposeOperator(y=None, masks=None, dims=None, *args, **kwargs):
    n1 = dims(1)
# experimentImage2D.m:76
    n2 = dims(2)
# experimentImage2D.m:77
    L = np.shape(masks, 3)
# experimentImage2D.m:78

    y = reshape(y, concat([n1, n2, L]))
# experimentImage2D.m:79

    x = np.multiply(dot(dot(n1, n2), ifft2(y)), conj(masks))
# experimentImage2D.m:81
    x = sum(x, 3)
# experimentImage2D.m:82
    x = ravel(x)
# experimentImage2D.m:83
    return x
