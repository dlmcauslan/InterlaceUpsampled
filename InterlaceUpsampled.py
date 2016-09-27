'''                              InterlaceUpsampled.py
created: 26/09/2016

Interlaces images for use with lenticular lenses that do not overlap exactly with an integer number of screen pixels.
It calculates the position of the lens relative to the screen pixels and assigns what fraction of each screen subpixel
is observed by each view.
Code was adapted from Matthew Hirsch's Matlab code for interlacing images:
http://alumni.media.mit.edu/~mhirsch/byo3d/
However it has been modified to perform the image assignment on a sub-pixel basis as opposed to a pixel basis which 
can improve image colour.
By increasing the upsampling parameter you will get better results. 15 is usually enough for testing; however, I usually
set it to 255 for the final image.

@author: dmcauslan
'''
#Import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize

## Define paramaters of the lens and screen.
# Define lenticular parameters.
lensWidth = 1.668       # width of lens (mm)
lensOffset = 0.45       # offset (number of lenses from left-hand side of screen)

# Define screen parameters.
pixelPitch = 0.294                                  # Pixel pitch (mm)
subpixelPitch = pixelPitch/3                        # Divide by 3 to get the subpixel pitch
nViews = int(round(lensWidth/pixelPitch))           # Number of views per lens
screenRes = np.array([1024, 1280])                  # Screen resolution [height width] (pixels)
screenDim  = pixelPitch*screenRes                   # Screen dimensions [height width] (mm)
imageRes = (screenRes[0], screenDim[1]//lensWidth+1)     # Resolution of each interlaced image

# Define filename etc.
imType         = 'numbersCrosstalk'           # type of image to be interlaced (e.g., 'calibration', 'general', etc.)
generalBase    = 'ferrari'           # base file name (for 'general' case)
generalDir     = 'H:/David/Google Drive/Lenticular Sheet/images/{}/'.format(generalBase) # base directory (for 'general' case)
generalExt     = 'jpg'               # file format    (for 'general' case)

# Define input interlacing options.
upsample = 15                  # upsampling factor for interlaced pattern (was 10)


## Loads/creates the images to be interlaced.
def imageLoader(imageRes, nViews, imType, generalDir, generalBase, generalExt):    
    print('> Loading the images to be interlaced...')
    # Loop over the number of views
    for i in range(nViews):
        print('  - Loading frame {} of {}...'.format(i+1, nViews))
        # Choose which set of images to create/load
        # Calibration image - all images are black, except the centre image which is white        
        if imType == 'calibration':
            # On first iteration create array lfImage to hold all images            
            if i==0:
                nImages = np.zeros(np.hstack((imageRes,3,nViews)), dtype = np.uint8)
            nImages[:,:,:,i] = 255*(i==round((nViews)/2))*np.ones(np.hstack((imageRes,3)), dtype = np.uint8)
        # Image used for alignment - alternating red, green, blue images.
        elif imType == 'redgreenblue':
            # On first iteration create array lfImage to hold all images            
            if i==0:
                nImages = np.zeros(np.hstack((imageRes,3,nViews)), dtype = np.uint8)
            if i%3 == 0:
                nImages[:,:,0,i] = 255*np.ones(imageRes)
            elif i%3 == 1:
                nImages[:,:,2,i] = 255*np.ones(imageRes)
            elif i%3 == 2:
                nImages[:,:,1,i] = 255*np.ones(imageRes)
        # Image used for measuring crosstalk between views
        elif imType == 'numbersCrosstalk':
            # How the numbers will be arranged, need to change this as we change nViews            
            nRow = 2
            nCol = 3
            # Load the numbers crosstalk images
            im1 = 255*mpimg.imread('H:/David/Google Drive/Lenticular Sheet/images/numbers crosstalk/numbers crosstalk 0{}.png'.format(i+1))
            sz = np.shape(im1)
            # Rearrange them so the different images are arranged in a grid
            im2 = np.zeros(np.hstack((nRow*sz[0], nCol*sz[1], 3)), dtype = np.uint8)
            rows = ((i%nRow)*sz[0]+np.arange(sz[0])).astype(int)
            cols = (np.floor(i/nRow)*sz[1]+np.arange(sz[1])).astype(int)
            im2[rows[:, np.newaxis], cols, :] = im1
            # On first iteration create array lfImage to hold all images
            if i == 0:
                nImages = np.zeros(np.hstack((np.shape(im2),nViews)),dtype = np.uint8)
            nImages[:,:,:,i] = im2
        # General image interlacing - for example ferrari images
        elif imType == 'general':
            # On first iteration create array lfImage to hold all images            
            if i == 0:
                tmp = mpimg.imread('{}{}-{:02d}.{}'.format(generalDir, generalBase, i+1, generalExt))
                nImages = np.zeros(np.hstack([np.shape(tmp),nViews]), dtype = np.uint8)
            nImages[:,:,:,i] = mpimg.imread('{}{}-{:02d}.{}'.format(generalDir, generalBase, i+1, generalExt))                     
        else:
            raise ValueError('You didnt correctly choose a set of images to load!')   
    
    # Plot images
    fig = plt.figure(1)
    plt.clf()    
    for n in range(nViews):
        ax = fig.add_subplot(231+n)
        ax.imshow(nImages[:,:,:,n],interpolation="nearest", aspect='auto')
        plt.title("View {}".format(n))
    plt.show()
    
    return nImages


## Evaluates the position of the screen pixels with respect to the lenses
def screenEvaluator(subpixelPitch, upsample, screenRes, lensWidth, lensOffset):
    # Display status.
    print('Lenticular Sheet Interlacer')
    print('> Evaluating resampling parameters...')
    # Create arrays to store results in
    screenU = np.zeros([3,upsample*screenRes[1]])
    screenS = np.zeros([3,upsample*screenRes[1]])

    # Evaluate the physical position of each (upsampled) subpixel.
    screenx = (subpixelPitch/upsample)*(np.arange(3*upsample*screenRes[1])+0.5)
    firstSub = np.reshape(np.matlib.repmat(np.arange(0,3*upsample*screenRes[1],3*upsample),upsample,1),-1, order='F')
    addOn = np.matlib.repmat(np.arange(0,upsample),1,screenRes[1])[0,:]
    indx_r = firstSub+addOn
    indx_g = indx_r+upsample
    indx_b = indx_g+upsample
    screen_x = np.array([screenx[indx_r], screenx[indx_g], screenx[indx_b]])
    # Determine light field sampling indices (i.e., the mapping from pixels to rays).
    screenU = np.floor((screen_x-lensWidth*lensOffset)/lensWidth)+1
    screenS = -(nViews/lensWidth)*(screen_x-lensWidth*lensOffset-lensWidth*screenU+lensWidth/2)+(nViews+1)/2+0.5
    
    return screenU, screenS


## Interlaces the images
def imageInterlacer(nImages, imageRes, nViews, upsample, screenRes, screenU, screenS):   
    # Resize each view to be the correct resolution (so that when interlaced the final image has the correct resolution).
    print('> Calculating the interlaced pattern...')
    patternImage = np.zeros(np.hstack((screenRes[0],upsample*screenRes[1],3)), dtype = np.uint8)
    # Loop over the nViews images    
    for i in range(nViews):
        # Reize the image
        tmpImage = resize(nImages[:,:,:,i], imageRes, preserve_range=True)
        # Interlace by subpixel - loop over 3 coloured subpixels
        for rgb in range(3):
            # Find the upsampled subpixel indices that correspond to image i
            validIdx = ((screenU[rgb,:] >= 0) & (screenU[rgb,:] < imageRes[1]) & (np.floor(screenS[rgb,:])-1 == i))
            # Assign the image to the correct subpixels of the total image
            patternImage[:,validIdx,rgb] = tmpImage[:,screenU[rgb,validIdx].astype(int),rgb]
   
    # Add the upsampled pixels together and divide by the upsampling rate to get an image that is the same resoltion as the screen
    if upsample > 1:
        I = np.zeros(np.hstack((screenRes,3)))
        for i in range(upsample):
            I = I+patternImage[:,i::upsample,:]
        patternImage = I/upsample
    patternImage = patternImage.astype(np.uint8)
    
    # Plot the final image
    fig3 = plt.figure(3)
    fig3.clf()
    plt.imshow(patternImage,interpolation="nearest")
    fig3.show()

    return patternImage


## Save the image
def saveImage(imageTot, imType, generalBase, nViews):
    if imType == 'general':
        fName = 'H:/David/Google Drive/Canopy/Interlaced Images/{}_{}view_upsampled.png'.format(generalBase,nViews)
    else:
        fName = 'H:/David/Google Drive/Canopy/Interlaced Images/{}_{}view_upsampled.png'.format(imType,nViews)
    mpimg.imsave(fName, imageTot) 

# Running the code
nImages = imageLoader(imageRes, nViews, imType, generalDir, generalBase, generalExt)    
screenU, screenS = screenEvaluator(subpixelPitch, upsample, screenRes, lensWidth, lensOffset)
imageTot = imageInterlacer(nImages, imageRes, nViews, upsample, screenRes, screenU, screenS)            
saveImage(imageTot, imType, generalBase, nViews)