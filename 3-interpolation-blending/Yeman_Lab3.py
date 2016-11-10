#Yeman Brhane
# MAIA 
#
#
# Import the library as show images, plot, etc.
import matplotlib.pyplot as plt
# Import the library as show images, plot, etc.
import matplotlib.pyplot as plt
# Import functionality for the color map
import matplotlib.cm as cm

# Import system specific parameters and function
import sys

## Other plotting libraries
# import seaborn as sns

# Import the library to mange the matrix and array
import numpy as np
import cv2
# Import the library to mange the ndimage
import scipy.ndimage
from scipy.misc import imresize
# Importing image processing toolbox
## Module to read, write,...
from skimage import io
## Module to convert the image on 8 bits
from skimage import img_as_ubyte
## Module to convert the image to float
from skimage import img_as_float
## Module for color conversion
from skimage import color
## Module image transform from skimage for resize
from skimage import transform
## Module misc from scipy for resize
from scipy import misc
## Module util from skimage
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from scipy.ndimage import convolve
import cv2

#############............................Image Rescaling
def nearest_scale(im,scale):
    [r,w]=im.shape
    new_r=scale*r
    new_w=scale*w
    new_img = misc.imresize(lena_im, (new_r,new_w),interp='nearest')
    return new_img

def bilinear_scale(im,scale):
    [r,w]=im.shape
    new_r=scale*r
    new_w=scale*w
    new_img = misc.imresize(lena_im, (new_r,new_w),interp='bilinear')
    return new_img
def bicubic_scale(im,scale):
    [r,w]=im.shape
    new_r=scale*r
    new_w=scale*w
    new_img = misc.imresize(lena_im, (new_r,new_w),interp='bicubic')
    return new_img
########################################
#...............Main starts here
Path = './images/'
lena_im = io.imread(Path.__add__('lena-grey.bmp'))
print("Size of the input image")
print(lena_im.shape)
scale=2;
plt.figure()
io.imshow(lena_im)
plt.title('Orginal Image')
nearest_im=nearest_scale(lena_im,scale)
plt.figure()
io.imshow(nearest_im)
plt.title('Nearest Neighbour Scaled Image,scale=2')
print("Size of the after nearest interpolation of image, scale=2")
print(nearest_im.shape)
#Bilinear Interpolation
bilinear_im=bilinear_scale(lena_im,scale)
plt.figure()
io.imshow(bilinear_im)
plt.title('Bilinear Scaled Image,scale=2')
print("Size of the after bilinear interpolation of image, scale=2")
print(bilinear_im.shape)
#bicubic interpolation
bicubic_im=bicubic_scale(lena_im,scale)
plt.figure()
io.imshow(bicubic_im)
plt.title('Bicubic Scaled Image,scale=2')
print("Size of the after Bicubic interpolation of image, scale=2")
print(bicubic_im.shape)


#  Image Blending
# Resize image to be blended
def image_Resize(image_1, image_2):
    m_image_1, n_image_1 = image_1.shape
    scaled_image_2 = imresize(image_2, (m_image_1, n_image_1), interp='bilinear')
    return scaled_image_2

# Simple blending function
def classical_blend(image_1, image_2):
    # Resize image to be blended
    scaled_image_2 = image_Resize(image_1, image_2)
    m_image_1, n_image_1 = image_1.shape
    m_image_2, n_image_2 = scaled_image_2.shape

    # Size of blended image
    mBlImage = m_image_1
    nBlImage = (n_image_1 + n_image_2) / 2

    # Take values of Left-half of image 1 and Right-half of Image 2
    classical_blended_image = np.zeros([mBlImage, nBlImage])
    classical_blended_image[:, 0:n_image_1 / 2 - 1] = image_1[:, 0:n_image_1 / 2 - 1]
    classical_blended_image[:, n_image_1 / 2:nBlImage - 1] = scaled_image_2[:, n_image_1 / 2:nBlImage - 1]

    # Return simple blended image
    return classical_blended_image

	
# Alpha blending function
def alpha_blend(image_1, image_2):
    # Resize image to be blended
    scaled_image_2 = image_Resize(image_1, image_2)
    m_image_1, n_image_1 = image_1.shape
    m_image_2, n_image_2 = scaled_image_2.shape

    classical_blended_image=classical_blend(image_1, image_2)

    # Get alpha blended image
    w = 200
    image_1_Blend = image_1[:, n_image_1 / 2 - w / 2:n_image_1 / 2 + w / 2]
    scaled_image_2_Blend = scaled_image_2[:, n_image_2 / 2 - w / 2: n_image_2 / 2 + w / 2]
    alpha_blended_image = classical_blended_image

    for i in range(0, w):
        alpha_orange = -1.0 / w * i + 1
        alpha_apple = 1 - alpha_orange
        alpha_blended_image[:, n_image_1 / 2 - w / 2 + i] = alpha_orange * image_1_Blend[:, i] + alpha_apple * scaled_image_2_Blend[:, i]

    # Return simple blended image
    return alpha_blended_image


# ---------------------------------------------------------------
# Pyramid blending function
def pyramid_blend(image_1, image_2):
    # Resize image to be power of 2
    scaled_image_1 = imresize(image_1, (512, 512), interp='bilinear')
    scaled_image_2 = imresize(image_2, (512, 512), interp='bilinear')

    # generate Gaussian pyramid for A
    G = scaled_image_1.copy()
    gpA = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpA.append(G)

    # generate Gaussian pyramid for B
    G = scaled_image_2.copy()
    gpB = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpB.append(G)

    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)

    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)

    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols = la.shape
        ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1, 6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    # image with direct connecting each half
    real = np.hstack((scaled_image_1[:, :cols / 2], scaled_image_2[:, cols / 2:]))
    return ls_


# Blending two images
orange = color.rgb2gray(io.imread('images/orange.jpeg'))
apple = color.rgb2gray(io.imread('images/apple.jpeg'))

# Simple blended
classical_blended_image=classical_blend(orange ,apple)
plt.figure()
plt.axis('off')
plt.imshow(classical_blended_image, cmap = 'Greys_r')
plt.title('Simple blended image between orange and apple')

# Alpha blended
alpha_blended_image=alpha_blend(orange,apple)
plt.figure()
plt.axis('off')
plt.imshow(alpha_blended_image, cmap = 'Greys_r')
plt.title('Alpha blended image between orange and apple')

# Pyramid blended
pyramidBlendedImage=pyramid_blend(orange ,apple)
plt.figure()
plt.axis('off')
plt.imshow(pyramidBlendedImage, cmap = 'Greys_r')
plt.title('Pyramid blended image between orange and apple')
plt.show()
