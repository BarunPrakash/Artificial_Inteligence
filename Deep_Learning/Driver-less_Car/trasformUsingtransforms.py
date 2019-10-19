"""

Cross entropy 
Designed by Barun.
Exploring Deep Learnning
Date: 12 oct  2019

""" 

import torch
import numpy as np
import torchvision
from torchvision.datasets import MNIST
# importing matplotlib module  
from matplotlib import pyplot as plt 
import torchvision.transforms as transforms
  
#Dowlnload tranning dataset

dataset = MNIST(root ='data/',download =True,transform=transforms.ToTensor())


image ,label = dataset[0]
#plt.imshow(image)
#print(image.shape ,label)
print(image[:,10:15,10:15])
print(torch.max(image),torch.min(image))

plt.imshow(image[0],cmap ='gray');

#print(b.grad)









