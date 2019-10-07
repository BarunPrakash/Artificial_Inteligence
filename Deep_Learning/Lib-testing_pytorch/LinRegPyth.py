
"""   
Designed by Barun 
Vision :- Exploring ML Back progation 
Date  : 7 oct 2019



"""


import torch
from torch.autograd import Variable

X_lable = Variable(torch.Tensor([[1.0],[2.0],[3.0]]))
y_lable =  Variable(torch.Tensor([[2.0],[4.0],[6.0]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear =torch.nn.Linear(1,1) #  one in one out 
        
    def forward(self ,x):
        
        ypred =self.linear(x)
        return ypred
    
    
    
model = Model()

# construct loss and optimizer
criterion = torch.nn.MSELoss(size_average =False)
optimiser = torch.optim.SGD(model.parameters(),lr =0.01)



for epoch in range(500):
    
    yped =model(X_lable)
    #compute loss
    loss =criterion(yped,X_lable)
   # print(epoch ,loss.data[0])
    # perform backward and update the wait
    optimiser.zero_grad()
    loss.backward(1)
    optimiser.step()
    
    #After Trainnig
    houre_var =Variable(torch.Tensor([[4.0]]))
    print("Predict after Trainnig",4,model.forward(houre_var).data[0][0])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
