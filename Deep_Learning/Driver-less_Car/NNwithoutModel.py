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

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
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

#print(len(train_indices),len(val_indice3s))
#print("sample val indices",val_indices[:20])

batch_size = 100
# Trainnig sampler and data loader
train_sampler =SubsetRandomSampler(train_indices)
trian_loader  = DataLoader(dataset,batch_size ,sampler = train_sampler)

#Validation sampler and data loader

val_sampler = SubsetRandomSampler(val_indices)
val_loader  =  DataLoader(dataset,batch_size ,sampler = val_sampler)

############################################################

input_size = 28 * 28
num_classes = 10


model = nn.Linear(input_size ,num_classes)

print(model.weight.shape)
model.weight
print(model.bias.shape)
model.bias

for images ,labels  in trian_loader:
    print(images.shape)
    output =model(images)
    break

