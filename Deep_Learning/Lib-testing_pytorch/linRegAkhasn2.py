"""

Cross entropy 
Designed by Barun.
Exploring Deep Learnning
Date: 12 oct  2019

""" 

import torch
import numpy as np
# importing matplotlib module  
from matplotlib import pyplot as plt 
  



inputs = np.array([[73,67,43],
                  [91,88,64],
                  [87,88,64],
                  [102,43,37],
                  [69,96,70]],dtype ='float32')


targets = np.array([[56,70],
                  [81,101],
                  [119,133],
                  [22,37],
                  [103,119]
                   ],dtype ='float32')





inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

#t1 =torch.Tensor([l1,l2])

#print(inputs)
#print(targets)

#weigth and biases
"""
torch.randn creats a tensor with the given shape ,with elements picked
randomly from random distribution 
,in probabilty theory 
"""
w =torch.randn(2,3,requires_grad=True)
b=torch.randn(2,requires_grad=True)

print(w)
print(b)

def  model(x):
    return x @ w.t()+b # @ matrix multiplications .t tranpose


preds = model(inputs)  # set of prediction for target value.

print(preds)

def MSE(t1,t2):
    diff = t1-t2
    return torch.sum(diff*diff)/diff.numel() # numel method returns the no of elements. 
 
    
loss =MSE(preds, targets)
print("loss",loss)
loss.backward()
print(w.grad)
print(b.grad)



#Adjust weigth and reset gradients
with torch.no_grad():
    w-=w.grad *1e-5
    b-=w.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()

"""plt.plot(loss,w)
plt.ylabel('loss')
plt.ylabel('Weigth')
plt.show()
"""
#w.grad.zero_( )
#b.grad_zero_()
#print(w.grad)
#print(b.grad)









