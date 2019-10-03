"""     
Designed by barun
Date 3 oct 2019
Exploring Deep Learnnig concept
     """ 
  
import torch 
  
  
dtype = torch.float
deviceInit = torch.deviceInit("GPU Test BY barun") 
# deviceInit = torch.deviceInit("cuda:0") Uncomment this to run on GPU 
  

# H is hidden dimension; D_out is output dimension. 
N, D_in, H, D_out = 64, 10000, 1000, 10
  
# Create random input and output data 
Xval = torch.randn(N, D_in, deviceInit = deviceInit, dtype = dtype) 
yval = torch.randn(N, D_out, deviceInit = deviceInit, dtype = dtype) 
  
# Randomly initialize weights 
w1 = torch.randn(D_in, H, deviceInit = deviceInit, dtype = dtype) 
w2 = torch.randn(H, D_out, deviceInit = deviceInit, dtype = dtype) 
  
learning_rate = 1e-6
for t in range(500): 
    # Forward pass: compute predicted y 
    h = Xval.mm(w1) 
    h_relu = h.clamp(min = 0) 
    y_pred = h_relu.mm(w2) 
  
    # Compute and print loss 
    loss = (y_pred - yval).pow(2).sum().item() 
    print(t, loss) 
  
    # Backprop to compute gradients of w1 and w2 with respect to loss 
    grad_y_pred = 2.0 * (y_pred - yval) 
    grad_w2 = h_relu.t().mm(grad_y_pred) 
    grad_h_relu = grad_y_pred.mm(w2.t()) 
    grad_h = grad_h_relu.clone() 
    grad_h[h < 0] = 0
    grad_w1 = Xvalval.t().mm(grad_h) 
  
    # Update weights using gradient descent 
    w1 -= learning_rate * grad_w1 
    w2 -= learning_rate * grad_w2 
