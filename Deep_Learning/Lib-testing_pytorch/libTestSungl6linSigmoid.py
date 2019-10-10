""" libsigmoid  test 
Designed by Barun
1 10 2019
"""



import torch 
from torch.autograd import Variable 
import torch.nn.functional as F
  
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0],[4.0]])) 
y_data = Variable(torch.Tensor([[0.0], [0.0], [1.0],[1.0]])) 
  
  
class LinearRegressionModel(torch.nn.Module): 
  
    def __init__(self): 
        super(LinearRegressionModel, self).__init__() 
        self.linear = torch.nn.Linear(1, 1)  # One in and one out 
  
    def forward(self, x): 
       # y_pred = self.linear(x) 
       y_pred =F.sigmoid(self.linear(x))
       return y_pred 
  
# our model 
our_model = LinearRegressionModel() 
  
#criterion = torch.nn.MSELoss(size_average = False)
criterion = torch.nn.BCELoss(size_average = False) 
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01) 
  
for epoch in range(1000): 
  
    # Forward pass: Compute predicted y by passing  
    # x to the model 
    pred_y = our_model(x_data) 
  
    # Compute and print loss 
    loss = criterion(pred_y, y_data)
    print("epoch",loss.data)
  
    # Zero gradients, perform a backward pass,  
    # and update the weights. 
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 
#    print('epoch {}, loss {}'.format(epoch, loss.data[0])) 
  #After Trainnig
hour_var = Variable(torch.Tensor([[1.0]]))
print("Predict 1 hour",1.0,our_model(hour_var).data[0][0]>0.5)
hour_var =Variable(torch.Tensor([[7.0]]))
print("Predict 1 hour",7.0,our_model(hour_var).data[0][0]>0.5)
#new_var = Variable(torch.Tensor([[4.0]])) 
#pred_y = our_model(new_var) 
#print("predict (after training)", 4, our_model(new_var).data[0][0])
