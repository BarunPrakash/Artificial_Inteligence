import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn






dataset = MNIST(root='data/',download = True)

test_dataset = MNIST(root='data/',train=False)
datasetTransform =MNIST(root='data/',train =True,transform =transforms.ToTensor())

batch_size =100
#print(len(dataset))
#print(test_dataset)
#print(dataset[0])
"""
image ,label =dataset[10]
plt.imshow(image)
print('Label',label.item())

img_tensor ,label =datasetTransform[0]
print(img_tensor.shape,label)

 # The image is now   conveted to a 1X28X28 tensor.
#The first dimension is used to track of clor channel.
#other datasets have images with color in which case there are three channel(RGB)
 
print(img_tensor[:,10:15,10:15])
print(torch.max(img_tensor),torch.min(img_tensor))
plt.imshow(img_tensor[0],cmap='gray')
"""
def split_indices(n, val_pct):
    #Determine the size of validation set
    n_val =int(val_pct*n)
    #create random permutation of 0 to n-1
    idxs  = np.random.permutation(n)
    return idxs[n_val:],idxs[:n_val]









train_indices,val_indices =split_indices(len(dataset),val_pct=0.2)
print(len(train_indices),len(val_indices))
print('Sample val indices:',val_indices[:20])

#Trainnig sampler and dataloader
train_sampler =SubsetRandomSampler(train_indices)
train_loader =DataLoader(dataset,batch_size,sampler=train_sampler)

#validation sampler data and loader

val_sampler =SubsetRandomSampler(train_indices)
val_loader =DataLoader(dataset,batch_size,sampler=val_sampler)

#########################################################
input_size =28*28
num_classes =10

#logistic regresion model
"""
model =nn.Linear(input_size,num_classes)

print(model.weight.shape)
print(model.weight)
print(model.bias.shape)

for images,labels in train_loader:
    print(images.shape)
    output =model(images)
    break
"""

####

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear =nn.Linear(input_size,num_classes)
        
    def forward(self,xb):
        xb =xb.reshape(-1,784)
        out =self.linear(xb)
        return out
    
model =MnistModel()


print(list(model.parameters()))


for images ,labels in train_loader:
    outputs =model(images)
    break
print('output.shape',output.shape)
