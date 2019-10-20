"""

Cross entropy 
Designed by Barun.
Exploring Deep Learnning
Date: 20 oct  2019

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
#print(image[:,10:15,10:15])
#print(torch.max(image),torch.min(image))

#plt.imshow(image[0],cmap ='gray');

#print(b.grad)

def split_indicaces(n ,val_pct):
    nval = int(val_pct*n)
    idxs  = np.random.permutation(n)
    return idxs[nval:],idxs[:nval]


train_indices ,val_indices =split_indicaces(len(dataset),val_pct =0.2)

print(len(train_indices),len(val_indices))
print("sample val indices",val_indices[:20])





