"""

Cross entropy 
Designed by Barun.
Exploring Deep Learnning
Date: 12 oct  2019

""" 

import numpy as np

y = np.array([1,0,0])
y_pred1 = np.array([0.7,0.2,0.1])
y_pred1 = np.array([0.1,0.3,0.6])
print("loss ",np.sum(-y*np.log(y_pred1)))
