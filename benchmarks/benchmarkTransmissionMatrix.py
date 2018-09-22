# Generated with SMOP  0.41
from libsmop import *
# benchmarkTransmissionMatrix.m

    #                     benchmarkTransmissionMatrix.m
    
    # Code adapted from Sudarshan Nagesh for the reconstruction of images from
# a transmission matrix.
    
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
    
    # Note: The empirical dataset must be downloaded and installed into the
# 'data' directory for this to work. See the user guide.
    
    
    ##I/O
#  Inputs:
#  imageSize            : Size of image to reconstruct.  Must be {16,40,64}
#  datasetSelection     : Choose which of the sets of measurements to use.
#                           Must be in {1,2,3,4,5}.
#  residualConstant     : Only use rows of the transmission matrix that
#                           had residuals less than this cutoff. Must be 
#                           between 0 and 1.  Recommend 0.4.
#  algorithms           : A cell-array of structs that describe algorithms.
    
    
    ##References
#  Paper Title:   Coherent inverse scattering via transmission matrices:
#  Efficient phase retrieval algorithms and a public dataset.
#  Authors:       Christopher A. Metzler,  Manoj
#  K. Sharma,  Sudarshan Nagesh,  Richard G. Baraniuk,
#                 Oliver Cossairt,  Ashok Veeraraghavan
    
    
    # PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein 
# Copyright (c) University of Maryland, 2017
    
    ## -----------------------------START----------------------------------
    
    
@function
def benchmarkTransmissionMatrix(imageSize=None,datasetSelection=None,residualConstant=None,algorithms=None,*args,**kwargs):
    varargin = benchmarkTransmissionMatrix.varargin
    nargin = benchmarkTransmissionMatrix.nargin

    
    ## Load the transmission matrix, and phaseless measurements
    A,Xt,b=loadData(residualConstant,datasetSelection,imageSize,nargout=3)
# benchmarkTransmissionMatrix.m:50
    
    # printed after all calculations are done.
    # Results will also be printed incrementally as the benchmark runs.
    resultsTable='\nAlgorithm  |  Measurement Error \n'
# benchmarkTransmissionMatrix.m:55
    
    for k in arange(1,length(algorithms)).reshape(-1):
        opts=algorithms[k]
# benchmarkTransmissionMatrix.m:59
        fprintf('Running Algorithm: %s\n',opts.algorithm)
        reconError,measurementError=reconstructSignal(A,Xt,b,imageSize,datasetSelection,opts,nargout=2)
# benchmarkTransmissionMatrix.m:63
        resultsTable=concat([resultsTable,pad(opts.algorithm,16),sprintf('%0.5f',measurementError),'\n'])
# benchmarkTransmissionMatrix.m:66
    
    
    ## print the table
    fprintf(resultsTable)
    return
    
if __name__ == '__main__':
    pass
    
    #######################################################################
###                        Utility functions                        ###
#######################################################################
    
    # Load the measurement operator and measurements that correspond to the
# image size and dataset chosen by the user.  This method returns the
# measurement operator 'A' (a dense matrix), a ground truth solution
# 'Xt', and a set of measurements 'Y'.
    
@function
def loadData(residualConstant=None,datasetSelection=None,imageSize=None,*args,**kwargs):
    varargin = loadData.varargin
    nargin = loadData.nargin

    # Create some strings for dealing with different filenames according to the dataset chosen
    if cellarray([16]) == imageSize:
        measurementType='AmpSLM_16x16'
# benchmarkTransmissionMatrix.m:88
    else:
        if cellarray([40]) == imageSize:
            measurementType='PhaseSLM_40x40'
# benchmarkTransmissionMatrix.m:90
        else:
            if cellarray([64]) == imageSize:
                measurementType='AmpSLM_64x64'
# benchmarkTransmissionMatrix.m:92
            else:
                error('illegal imageSize: %d. It should be chosen from {16, 40, 64}',imageSize)
    
    
    # Make sure the dataset selection is valid
    if imageSize == 16 or imageSize == 40:
        assert_(datasetSelection < 5)
    else:
        if imageSize == 64:
            assert_(datasetSelection < 6)
    
    
    ## Load the transmission matrix, ground truth image, and measurements
    dataRoot=getFolderPath('data')
# benchmarkTransmissionMatrix.m:105
    
    try:
        disp(concat(['Loading transmission matrix (A_prVAMP.mat), this may take several minutes']))
        load(strcat(dataRoot,'TransmissionMatrices/Coherent_Data/',measurementType,'/A_prVAMP.mat'))
        disp(concat(['Loading test image (XH_test.mat)']))
        load(strcat(dataRoot,'/TransmissionMatrices/Coherent_Data/',measurementType,'/XH_test.mat'))
        disp(concat(['Loading measurements (YH_squared_test.mat)']))
        load(strcat(dataRoot,'TransmissionMatrices/Coherent_Data/',measurementType,'/YH_squared_test.mat'))
    finally:
        pass
    
    # Only use the most accurate rows of the transmission matrix.  This is
    # determined by checking that the residuals found during row calcuation
    # are less than the specificed "residualConstant".
    findPixels=find(residual_vector < residualConstant)
# benchmarkTransmissionMatrix.m:122
    YH_squared_test=YH_squared_test(arange(),findPixels)
# benchmarkTransmissionMatrix.m:123
    A=A(findPixels,arange())
# benchmarkTransmissionMatrix.m:125
    
    Y=double(YH_squared_test(datasetSelection,arange()))
# benchmarkTransmissionMatrix.m:128
    b=sqrt(Y)
# benchmarkTransmissionMatrix.m:129
    
    # Note:  this image is real-valued
    Xt=double(XH_test(datasetSelection,arange())).T
# benchmarkTransmissionMatrix.m:133
    Xt=Xt / max(abs(Xt))
# benchmarkTransmissionMatrix.m:134
    
    originalImage=imresize(reshape(real(Xt),imageSize,imageSize),4)
# benchmarkTransmissionMatrix.m:136
    originalImageName=concat(['TM',num2str(imageSize),'-',num2str(datasetSelection),'-original'])
# benchmarkTransmissionMatrix.m:137
    imshow(originalImage,[])
    title(originalImageName,'fontsize',16)
    drawnow
    
    saveBenchmarkImage(originalImage,concat(['TM',num2str(imageSize)]),originalImageName)
    return A,Xt,b
    
if __name__ == '__main__':
    pass
    
    # Reconstruct a signal from a given measurement operator and set of
# measurements.  Compare the results to the ground-truth Xt, and report
# error.
    
@function
def reconstructSignal(A=None,Xt=None,Y=None,imageSize=None,datasetSelection=None,opts=None,*args,**kwargs):
    varargin = reconstructSignal.varargin
    nargin = reconstructSignal.nargin

    fprintf('    Reconstructing image...')
    
    # Convenient variables
    n=imageSize ** 2
# benchmarkTransmissionMatrix.m:150
    
    imageSizeStr=num2str(imageSize)
# benchmarkTransmissionMatrix.m:151
    
    ind=copy(datasetSelection)
# benchmarkTransmissionMatrix.m:152
    
    
    # Solve the PR problem
    X,outs,opts=solvePhaseRetrieval(A,[],ravel(Y),n,opts,nargout=3)
# benchmarkTransmissionMatrix.m:155
    
    # transform that maps the recovered solution onto the true solution
    Xrec=concat([ravel(X),ones(numel(X),1)])
# benchmarkTransmissionMatrix.m:159
    coeffs=numpy.linalg.solve(Xrec,ravel(Xt))
# benchmarkTransmissionMatrix.m:160
    Xrec=dot(Xrec,coeffs)
# benchmarkTransmissionMatrix.m:161
    
    realIm=imresize(reshape(real(Xrec),imageSize,imageSize),4)
# benchmarkTransmissionMatrix.m:164
    recoveredImageNameAbs=strcat('TM',imageSizeStr,'-',num2str(ind),'-',opts.algorithm,num2str(outs.iterationCount),'-real')
# benchmarkTransmissionMatrix.m:165
    imshow(realIm,[])
    title(recoveredImageNameAbs,'fontsize',16)
    drawnow
    realIm=realIm - min(ravel(realIm))
# benchmarkTransmissionMatrix.m:168
    realIm=realIm / max(ravel(realIm))
# benchmarkTransmissionMatrix.m:169
    saveBenchmarkImage(realIm,concat(['TM',imageSizeStr]),recoveredImageNameAbs)
    reconError=norm(ravel(Xt) - ravel(Xrec)) / norm(ravel(Xt))
# benchmarkTransmissionMatrix.m:173
    measurementError=norm(abs(dot(A,X)) - ravel(Y)) / norm(ravel(Y))
# benchmarkTransmissionMatrix.m:174
    
    fprintf('    Relative measurement error = %s\n',measurementError)
    return reconError,measurementError
    
if __name__ == '__main__':
    pass
    
    
    # Find the path for the specified folder by searching the current and
# parent directories
    
@function
def getFolderPath(fname=None,*args,**kwargs):
    varargin = getFolderPath.varargin
    nargin = getFolderPath.nargin

    d=copy(dir)
# benchmarkTransmissionMatrix.m:182
    
    if any(strcmp(fname,cellarray([d(concat([d.isdir])).name]))):
        path=concat([fname,'/'])
# benchmarkTransmissionMatrix.m:185
    else:
        # Look for folder fname in parent directory
        d=dir('../')
# benchmarkTransmissionMatrix.m:188
        if any(strcmp(fname,cellarray([d(concat([d.isdir])).name]))):
            path=concat(['../',fname,'/'])
# benchmarkTransmissionMatrix.m:190
        else:
            error(concat(['Unable to find path of folder: ',fname,'.  Make sure your current directory is the PhasePack root.']))
    
    return path
    
if __name__ == '__main__':
    pass
    
    # Save an image to a sub-folder inside 'benchmarkResults'
    
@function
def saveBenchmarkImage(image=None,folder=None,fname=None,*args,**kwargs):
    varargin = saveBenchmarkImage.varargin
    nargin = saveBenchmarkImage.nargin

    bmRoot=getFolderPath('benchmarkResults')
# benchmarkTransmissionMatrix.m:200
    if logical_not(exist(concat([bmRoot,folder]),'dir')):
        mkdir(concat([bmRoot,folder]))
    
    fullName=strcat(bmRoot,folder,'/',fname,'.png')
# benchmarkTransmissionMatrix.m:204
    fprintf('    Saving reconstructed image: %s\n',fullName)
    try:
        imwrite(image,fullName)
    finally:
        pass
    
    return
    
if __name__ == '__main__':
    pass
    