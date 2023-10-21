import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#load image as pixel array
img0 = mpimg.imread('peppers.png')
img0 = img0[26:410,83:595,:] #crop image

plt.figure()
plt.imshow(img0)
plt.title('Original RGB Image')
plt.savefig('originalimg.png') 

#get data type and shape of image
print('data type:', img0.dtype) 
print('height, width, dimension (number channels):', img0.shape) # 4 channels correspoding to red, green, blue and transparency

#colour transform 
def rgb2ycbcr(image): # RGB to YCbCr colourspace
    ycbcr = np.empty_like(img0)
    r = img0[:,:,0]
    g = img0[:,:,1]
    b = img0[:,:,2]  
    C1 = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]]) #transform matrix
    # Y
    ycbcr[:,:,0] = C1[0,0] * r + C1[0,1] * g + C1[0,2] * b 
    # Cb
    ycbcr[:,:,1] = C1[1,0] * r + C1[1,1] * g + C1[1,2] * b 
    # Cr
    ycbcr[:,:,2] = C1[2,0] * r + C1[2,1] * g + C1[2,2] * b  
    #Transparency
    ycbcr[:,:,3] = 1
    return (ycbcr)

def ycbcr2rgb(image): # YCbCr to RGB colourspace
    rgb = np.empty_like(img0)
    y   = img1[:,:,0]
    cb  = img1[:,:,1] 
    cr  = img1[:,:,2] 
    C2 = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]]) #transform matrix
    # R
    rgb[:,:,0] = C2[0,0] * y + C2[0,1] * cb + C2[0,2] * cr
    # G
    rgb[:,:,1] = C2[1,0] * y + C2[1,1] * cb + C2[1,2] * cr
    # B
    rgb[:,:,2] = C2[2,0] * y + C2[2,1] * cb + C2[2,2] * cr
    #Transparency
    rgb[:,:,3] = 1
    np.putmask(rgb, rgb > 1, 1) #ensures pixel values are in the correct range [0,1] for rgb images
    np.putmask(rgb, rgb < 0, 0)
    return (rgb)

img1 = rgb2ycbcr(img0) #transforms img0 so I can work with YCbCr data
Y = img1[:,:,0] 
Cb = img1[:,:,1] 
Cr = img1[:,:,2] 
