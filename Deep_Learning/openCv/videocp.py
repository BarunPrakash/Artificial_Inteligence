'''
Designed by barun 
Building concept for direverless car

            '''


import numpy as np
import cv2



# Capture video from file
cap = cv2.VideoCapture('car.webm')

while(cap.isOpened()):
   
    ret, checkFram = cap.read()

    gray = cv2.cvtColor(checkFram, cv2.COLOR_BGR2GRAY)

   
    cv2.imshow('checkFram',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
