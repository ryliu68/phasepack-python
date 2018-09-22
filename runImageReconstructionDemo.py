# Generated with SMOP  0.41
from libsmop import *
# .\runImageReconstructionDemo.m

    ##                   runImageReconstructionDemo.m
    
    # This script will create phaseless measurements from a test image, and 
# then recover the image using phase retrieval methods.  We now describe 
# the details of the simple recovery problem that this script implements.
# 
#                         Recovery Problem
# This script loads a test image, and converts it to grayscale.
# Measurements of the image are then obtained by applying a linear operator
# to the image, and computing the magnitude (i.e., removing the phase) of 
# the linear measurements.
    
    #                       Measurement Operator
# Measurement are obtained using a linear operator, called 'A', that 
# obtains masked Fourier measurements from an image.  Measurements are 
# created by multiplying the image (coordinate-wise) by a 'mask,' and then
# computing the Fourier transform of the product.  There are 8 masks,
# each of which is an array of binary (+1/-1) variables. The output of
# the linear measurement operator contains the Fourier modes produced by 
# all 8 masks.  The measurement operator, 'A', is defined as a separate 
# function near the end of the file.  The adjoint/transpose of the
# measurement operator is also defined, and is called 'At'.
    
    #                         Data structures
# PhasePack assumes that unknowns take the form of vectors (rather than 2d
# images), and so we will represent our unknowns and measurements as a 
# 1D vector rather than 2D images.
    
    #                      The Recovery Algorithm
# The image is recovered by calling the method 'solvePhaseRetrieval', and
# handing the measurement operator and linear measurements in as arguments.
# A struct containing options is also handed to 'solvePhaseRetrieval'.
# The entries in this struct specify which recovery algorithm is used.
    
    # For more details, see the Phasepack user guide.
    
    # PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    
@function
def runImageReconstructionDemo(*args,**kwargs):
    varargin = runImageReconstructionDemo.varargin
    nargin = runImageReconstructionDemo.nargin

    ## Specify the target image and number of measurements/masks
    image=imread('data/logo.jpg')
# .\runImageReconstructionDemo.m:44
    
    image=double(rgb2gray(image))
# .\runImageReconstructionDemo.m:45
    
    num_fourier_masks=8
# .\runImageReconstructionDemo.m:46
    
    ## Create 'num_fourier_masks' random binary masks. Store them in a 3d array.
    numrows,numcols=size(image,nargout=2)
# .\runImageReconstructionDemo.m:50
    
    random_vars=rand(num_fourier_masks,numrows,numcols)
# .\runImageReconstructionDemo.m:51
    
    masks=dot((random_vars < 0.5),2) - 1
# .\runImageReconstructionDemo.m:52
    
    ## Compute phaseless measurements
# Note, the measurement operator 'A', and it's adjoint 'At', are defined
# below as separate functions
    x=ravel(image)
# .\runImageReconstructionDemo.m:57
    
    b=abs(A(x))
# .\runImageReconstructionDemo.m:58
    
    ## Set options for PhasePack - this is where we choose the recovery algorithm
    opts=copy(struct)
# .\runImageReconstructionDemo.m:61
    
    opts.algorithm = copy('PhaseMax')
# .\runImageReconstructionDemo.m:62
    
    opts.initMethod = copy('optimal')
# .\runImageReconstructionDemo.m:63
    
    opts.tol = copy(0.001)
# .\runImageReconstructionDemo.m:64
    
    opts.verbose = copy(2)
# .\runImageReconstructionDemo.m:65
    
    ## Run the Phase retrieval Algorithm
    fprintf('Running %s algorithm\n',opts.algorithm)
    # Call the solver using the measurement operator 'A', its adjoint 'At', the
# measurements 'b', the length of the signal to be recovered, and the
# options.  Note, this method can accept either function handles or
# matrices as measurement operators.   Here, we use function handles
# because we rely on the FFT to do things fast.
    x,outs,opts=solvePhaseRetrieval(A,At,b,numel(x),opts,nargout=3)
# .\runImageReconstructionDemo.m:74
    # Convert the vector output back into a 2D image
    recovered_image=reshape(x,numrows,numcols)
# .\runImageReconstructionDemo.m:77
    # Phase retrieval can only recover images up to a phase ambiguity. 
# Let's apply a phase rotation to align the recovered image with the 
# original so it looks nice when we display it.
    rotation=sign(dot(ravel(recovered_image).T,ravel(image)))
# .\runImageReconstructionDemo.m:82
    recovered_image=real(dot(rotation,recovered_image))
# .\runImageReconstructionDemo.m:83
    # Print some useful info to the console
    fprintf('Image recovery required %d iterations (%f secs)\n',outs.iterationCount,outs.solveTimes(end()))
    ## Plot results
    figure
    # Plot the original image
    subplot(1,3,1)
    imagesc(image)
    title('Original Image')
    # Plot the recovered image
    subplot(1,3,2)
    imagesc(real(recovered_image))
    title('Recovered Image')
    # Plot a convergence curve
    subplot(1,3,3)
    convergedCurve=semilogy(outs.solveTimes,outs.residuals)
# .\runImageReconstructionDemo.m:101
    set(convergedCurve,'linewidth',1.75)
    grid('on')
    xlabel('Time (sec)')
    ylabel('Error')
    title('Convergence Curve')
    set(gcf,'units','points','position',concat([0,0,1200,300]))
    ###########################################################################
####               Measurement Operator Defined Below                  ####
###########################################################################
    
    # Create a measurement operator that maps a vector of pixels into Fourier
# measurements using the random binary masks defined above.
    
@function
def A(pixels=None,*args,**kwargs):
    varargin = A.varargin
    nargin = A.nargin

    # The reconstruction method stores iterates as vectors, so this 
    # function needs to accept a vector as input.  Let's convert the vector
    # back to a 2D image.
    im=reshape(pixels,concat([numrows,numcols]))
# .\runImageReconstructionDemo.m:120
    
    measurements=zeros(num_fourier_masks,numrows,numcols)
# .\runImageReconstructionDemo.m:122
    
    for m in arange(1,num_fourier_masks).reshape(-1):
        this_mask=squeeze(masks(m,arange(),arange()))
# .\runImageReconstructionDemo.m:125
        measurements[m,arange(),arange()]=fft2(multiply(im,this_mask))
# .\runImageReconstructionDemo.m:126
    
    # Convert results into vector format
    measurements=ravel(measurements)
# .\runImageReconstructionDemo.m:129
    return measurements
    
if __name__ == '__main__':
    pass
    
    # The adjoint/transpose of the measurement operator
    
@function
def At(measurements=None,*args,**kwargs):
    varargin = At.varargin
    nargin = At.nargin

    # The reconstruction method stores measurements as vectors, so we need 
    # to accept a vector input, and convert it back into a 3D array of 
    # Fourier measurements.
    measurements=reshape(measurements,concat([num_fourier_masks,numrows,numcols]))
# .\runImageReconstructionDemo.m:137
    
    im=zeros(numrows,numcols)
# .\runImageReconstructionDemo.m:139
    for m in arange(1,num_fourier_masks).reshape(-1):
        this_mask=squeeze(masks(m,arange(),arange()))
# .\runImageReconstructionDemo.m:141
        this_measurements=squeeze(measurements(m,arange(),arange()))
# .\runImageReconstructionDemo.m:142
        im=im + dot(dot(multiply(this_mask,ifft2(this_measurements)),numrows),numcols)
# .\runImageReconstructionDemo.m:143
    
    # Vectorize the results before handing them back to the reconstruction
    # method
    pixels=ravel(im)
# .\runImageReconstructionDemo.m:147
    return pixels
    
if __name__ == '__main__':
    pass
    
    return pixels
    
if __name__ == '__main__':
    pass
    