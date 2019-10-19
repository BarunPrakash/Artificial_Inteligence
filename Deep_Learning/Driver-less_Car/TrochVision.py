"""

Cross entropy 
Designed by Barun.
Exploring: vision computing for driver less car
Date: 12 oct  2019

""" 

import torch
import numpy as np
import torchvision
from torchvision.datasets import MNIST
# importing matplotlib module  
from matplotlib import pyplot as plt 
  
#Dowlnload tranning dataset

dataset = MNIST(root ='data/',download =True)


print(len(dataset))

image ,label = dataset[0]
plt.imshow(image)
print("Label:",label.item())

#print(b.grad)









