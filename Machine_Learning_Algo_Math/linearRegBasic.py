"""

Designed by Barun.
Exploring Deep Learnning
Date: 12 Nov  2019

(1) Generate Prediction
(2) calculate the loss
(3) compute the gradient with respect to weigth and biases
(4) Adjust thr weigth by subtraction small quality proportional to the gradient.
(5) Reset the gradient to zeoro.

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

#print(w)
#print(b)

def  model(x):
    return x @ w.t()+b # @ matrix multiplications .t tranpose


preds = model(inputs)  # set of prediction for target value.

#print(preds)

def MSE(t1,t2):
    diff = t1-t2
    return torch.sum(diff*diff)/diff.numel() # numel method returns the no of elements. 
 
    
loss =MSE(preds, targets)
#print("loss",loss)

loss.backward()
print(w)
print(w.grad)
plt.plot([-0.3872,-1.5023,1.15])
plt.ylabel('loss')
plt.show()

w.grad.zero_()
print(w.grad)



#Adjust weigth and reset gradients
with torch.no_grad():
    w-=w.grad * 0.01
    b-=w.grad * 0.01
    w.grad.zero_()
    b.grad.zero_()
"""
plt.plot(loss,w)
plt.ylabel('loss')
plt.ylabel('Weigth')
plt.show()
"""
#w.grad.zero_( )
#b.grad_zero_()
#print(w.grad)
#print(b.grad)

# train for 100 epochs
for i in range(100):
    preds = model(inputs)
    loss =MSE(preds,targets)
    loss.backward()
    with torch.no.grad():
         w-=w.grad * 0.001
         b-=w.grad * 0.001
         w.grad.zero_()
         b.grad.zero_()
         
         
         
         
         
         
preds =model(inputs)
loss =MSE(preds,targets)
print(loss)


#final comprision
print(preds)
print(targets)
    









