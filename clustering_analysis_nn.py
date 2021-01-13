# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 01:39:45 2020

@author: dhruv
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:07:23 2020

@author: dhruv
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import copy

###step 1######################

def create_data_blobs(n_samples=1000,n_features=2,number_class=3):
    '''
    

    Parameters
    ----------
    n_samples : TYPE Integer, optional
        DESCRIPTION. Number of data points to be generated. The default is 1000.
    n_features : TYPE Integer, optional
        DESCRIPTION. Number of features each data point should have.The default is 2.
    number_class : TYPE Integer, optional
        DESCRIPTION.Number of classes for the data points to be assigned. The default is 3.

    Returns
    -------
    x : TYPE Numpy Array of shape(n_samples,n_features)
        DESCRIPTION. Data points coordinates
    y : TYPE Numpy Array of shape(n_samples,1)
        DESCRIPTION. Assigned class values

    '''
    
    x=np.random.default_rng().uniform(-1000,1000,(n_samples,n_features))
   
    y=np.empty([n_samples,1])
    
    
    for i in range(x.shape[0]):
        rand=random.randint(0,number_class-1)
        y[i]=rand
 

    
    return x,y











def make_plot(x,y,num_classes,epoch):
    '''
    This function is used to plot the dataset.

    Parameters
    ----------
    x : TYPE Numpy Array of the shape (N,num_features)
        DESCRIPTION.  Data points coordinates 
    y : TYPE Numpy Array of the shape(N,1)
        DESCRIPTION. Assigned class values of the Data points
    num_classes : TYPE Integer
        DESCRIPTION. Number of classes present in the data points
    epoch : TYPE Integer
        DESCRIPTION. epoch number of training

    Returns
    -------
    None.

    '''
    
    for i in range(num_classes):
        x_class=[]
        y_class=[]
        for j in range(x.shape[0]):
            if y[j]==i:
                x_class.append(x[j][0])
                y_class.append(x[j][1])
        
        plt.scatter(x_class,y_class,label=str(i)+" class",marker='o')

    plt.title(str(epoch)+" epochs")
    plt.xlabel("x")
    plt.ylabel('y')
    plt.legend(loc="best")
    plt.show()

    
    
    
    
    
    
    
    
    
    
    

class clustering_model(torch.nn.Module):
    '''
    Neural Network with 14 layers
    '''
    def __init__(self,num_features,num_class):
        super(clustering_model,self).__init__()

        self.lin1=nn.Linear(num_features,256)
        self.lin2=nn.Linear(256,256)
        self.lin3=nn.Linear(256,512)
        self.lin4=nn.Linear(512,512)
        self.lin5=nn.Linear(512,1024)
        self.lin6=nn.Linear(1024,1024)
        self.lin7=nn.Linear(1024,2048)
        self.lin8=nn.Linear(2048,2048)
        self.lin9=nn.Linear(2048,1024)
        self.lin10=nn.Linear(1024,512)
        self.lin11=nn.Linear(512,256)
        self.lin12=nn.Linear(256,128)
        self.lin13=nn.Linear(128,64)
        self.lin14=nn.Linear(64,num_class)
    
       
    
    def forward(self,x):
     
        x=F.relu(self.lin1(x))
        x=F.relu(self.lin2(x))
        x=F.relu(self.lin3(x))
        x=F.relu(self.lin4(x))
        x=F.relu(self.lin5(x))
        x=F.relu(self.lin6(x))
        x=F.relu(self.lin7(x))
        x=F.relu(self.lin8(x))
        x=F.relu(self.lin9(x))
        x=F.relu(self.lin10(x))
        x=F.relu(self.lin11(x))
        x=F.relu(self.lin12(x))
        x=F.relu(self.lin13(x))
        x=self.lin14(x)
        
        
        
        return x


##intializing nueral network
num_features=2
num_class=3
model=clustering_model(num_features,num_class)
loss_fn=nn.CrossEntropyLoss()
lr=0.001
optimizer=torch.optim.SGD(params=model.parameters(),lr=lr)

transform_lr=10000000
#############


def train_one_epoch(x,y,transform_lr):
    x_copy=copy.deepcopy(x)
    y_copy=copy.deepcopy(y)
    x_train=torch.from_numpy(x_copy).float()
    y_train=torch.from_numpy(y_copy).long()
    y_train=y_train.squeeze(1)
  
    x_train.requires_grad=True
    optimizer.zero_grad()
    y_pred=model(x_train)
   
    loss=loss_fn(y_pred,y_train)  
    loss.backward(retain_graph=True)
    input_grad= torch.autograd.grad(loss, x_train)[0]
    input_grad_numpy=input_grad.numpy()
   
    
    x_copy=x_copy - input_grad_numpy*transform_lr
  
       
    optimizer.step()
        

    print(loss.item())
    return x_copy,y_copy
  

###################
n_samples=500
n_features=2
number_class=3
a,b=create_data_blobs(n_samples,n_features,number_class)


epochs=10000

        
for i in range(epochs):
    print(str(i)+" epoch",end=" ")
    a,b=train_one_epoch(a, b,transform_lr)
    if i%100==0:
        make_plot(a,b,number_class,i)
        
#############################
        

    






        
            
            

    
    
    
    
    







    
    
    
    