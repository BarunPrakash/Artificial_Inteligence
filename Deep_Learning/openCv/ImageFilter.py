import cv2
import numpy as np
import matplotlib.pyplot as plt

bgr = cv2.imread('bat.jpg')
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
cv2.imwrite('san_francisco_grayscale.jpg',gray)

plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF    # 0xFF? To get the lowest byte.
    if k == 27: break            # Code for the ESC key

cv2.destroyAllWindows()
