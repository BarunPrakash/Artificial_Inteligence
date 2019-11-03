# lec 8
import torch 
import numpy as np
from torch.autograd import Variable 
#import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

class DiabetesDataset(Dataset):
    def __init__(self):
        xy =np.loadtxt('data-diabetes.csv',delimiter=',', dtype=np.float32) 
        self.len =xy.shape[0]
        self.x_data = Variable(torch.from_numpy(xy[:,0:-1])) 
        self.y_data = Variable(torch.Tensor(xy[:,[-1 ]])) 
  
    
    def __getitem_(self,index):
        return self.x_data ,self.y_data[index]
    def __len__(self):
        return self.len
    
dataset =DiabetesDataset()    
train_loader =DataLoader(dataset=dataset,batch_size=32,shuffle =True,num_workers=2)
  #Design your model using claaa
class Model(torch.nn.Module): 
  
    def __init__(self): 
        super(Model, self).__init__() 
        self.l1 = torch.nn.Linear(8, 6)  #wide and  deep 
        self.l2 = torch.nn.Linear(6, 4)  
        self.l3 = torch.nn.Linear(4, 1)  
        
        self.sigmoid = torch.nn.Sigmoid()
        
  
    def forward(self, x): 
       # y_pred = self.linear(x) 
       out1 =self.sigmoid(self.l1(x))
       out2 =self.sigmoid(self.l2(out1))
       y_pred =self.sigmoid(self.l3(out2))
       return y_pred 
  
# our model 
our_model = Model() 
  
#criterion = torch.nn.MSELoss(size_average = False)
#construct loss otimizer
criterion = torch.nn.BCELoss(size_average = False) 
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01) 
 
#trainnig cycle 
for epoch in range(2):
    
    #get inpit
    inputs,labels =data
    # Forward pass: Compute predicted y by passing  
    #wrap them into varible
    inputs ,labels = Variable(inputs),Variable(labels)
    # x to the model 
    pred_y = our_model(inputs) 
  
    # Compute and print loss 
    loss = criterion(pred_y,labels )
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