"""     
Designed by barun
Date 3 oct 2019
Exploring Computer Vision
     """ 
  
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('bat.jpg',0)
f = np.fft.fft2(img) # fast furier transform for 2d
fshift = np.fft.fftshift(f)
###########################

magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show() 
