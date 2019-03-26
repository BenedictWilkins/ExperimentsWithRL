#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:08:44 2019

@author: ben
"""

import torch.nn as nn
import torch
import numpy as np
from Animation import SurfaceAnimation

#single linear layer of a network W.T*X
l = nn.Linear(2,5)
v = torch.FloatTensor([1,2])
l(v)
print(l(v))

#convert layers to a pipe, below is a 3 layer NN with softmax output.
#Note that dimension 0 is always batch_size (which is variable)
s = nn.Sequential(
        nn.Linear(2,5),
        nn.ReLU(),
        nn.Linear(5, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.Dropout(p=0.3),
        nn.Softmax(dim=1))

#run some values through the network, batch_size = 2 here
print(s(torch.FloatTensor([[1,2],[2,3]])))

# Modules are used to for reusability, example below
# on occasion one must register submodules, this is done automatically in this case by nn.Sequential
class ANN(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(ANN, self).__init__()
        self.pipe = nn.Sequential(
                nn.Linear(in_dim,3),
                nn.Sigmoid(),
                nn.Linear(3, 3),
                nn.Sigmoid(),
                nn.Linear(3, out_dim),
                nn.Sigmoid())
                #nn.Dropout(p=0.3),
                #nn.Softmax(dim=1))
    
    def forward(self, x):
        return self.pipe(x)
    
    
#print(ANN(2,2)(torch.FloatTensor([[1,1]])))


# Loss functions
# nn.MSELoss - mean squared error loss
# nn.BCELoss - binary cross entropy loss (expects probability value, usually output of sigmoid layer)
# nn.BCEWithLogits - binary cross entropy (applies sigmoid internally)
# nn.CrossEntropyLoss - expects raw scores for each class applies LogSoftMax internally
# nn.NLLLoss - (cross entropy loss) expects log probabilities as input

# One can write a custom loss function using the nn.Modules class

# Optimiser
# torch.optim.SGD - vanilla stochastic gradient descent
# torch.optim.RMSProp
# torch.Adagrad - adaptive gradient optimiser

# Example of the loop



def run_xor_ann(animated=False, epochs=10000):
    
    def iterate_batches(data, batch_size=1):
        (X, Y) = data
        data_len = len(X)
        for i in range(0, data_len, batch_size):
            yield  X[i:i+batch_size], Y[i:i+batch_size], epoch
        yield X[i:data_len], Y[i:data_len], epoch
    
    data = ([[0,0],[1,0],[0,1],[1,1]], [0,1,1,0]) #XOR
    net = ANN(2,1)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    optimizer.zero_grad()
    PRINTLOSS = 1000
    
    animation = None
    #for animation
    if animated:
        def update(i,X,Y,Z):
            #print("update animation: ", i)
            return X, Y, frames[i]
        
        ANIFRAME = 100
        frames = []
        l = np.linspace(0,1,20,dtype=np.float)
        animation = SurfaceAnimation(update,l,l)
        x = np.copy(animation.X)
        y = np.copy(animation.Y)
        x_ = x.reshape(x.shape[0]*x.shape[1])
        y_ = y.reshape(y.shape[0]*y.shape[1])
        animation_data = torch.as_tensor(np.vstack((x_,y_)).T,dtype=torch.float32)
        
    for epoch in range(epochs):    
        for batch_samples, batch_labels, epoch in iterate_batches(data, batch_size=1):
            batch_samples_t = torch.FloatTensor(batch_samples) #input tensor         
            batch_labels_t = torch.FloatTensor(batch_labels)   #targets
            #print(batch_samples_t)
            out_t = net(batch_samples_t)                  #predictions  
            loss_t = loss_function(out_t, batch_labels_t) # compute loss     
            loss_t.backward()                             # compute gradients w.r.t weights/bias in ANN  
            optimizer.step()                              # use these gradients to optimise weights via SGD
            optimizer.zero_grad()                         # zero out all gradients for reuse in the next step    
            
        if epoch % PRINTLOSS == 0: #print the current loss
            print("epoch: ", epoch, "loss: ", loss_t.item())
            net.eval()
            it = torch.FloatTensor(data[0])
            ht = torch.FloatTensor(np.array(data[1]).reshape(4,1))
            ot = net(it)
            print(torch.cat((it,ht,ot),1).detach().numpy())
            net.train()
            
        if animated and epoch % ANIFRAME == 0: #add frame to animation
            net.eval()
            out_animation = net(animation_data)
            out_animation = out_animation.detach().numpy().reshape((x.shape[0],x.shape[1]))
            frames.append(out_animation)
            net.train()
                  
    return net, animation
    
xor_net, ani = run_xor_ann(animated = True, epochs=10000)
ani.show()


# Models can be saved and loaded using torch.save(model.state_dict(), PATH) and model.load_state_dict(torch.load(PATH))
# Example below
print("\n\n")
print("Saved net:", xor_net.state_dict())

PATH = "models/xor_ann.pt"
#save
torch.save(xor_net.state_dict(), PATH)
#load
l_net = ANN(2,1)
l_net.load_state_dict(torch.load(PATH))
l_net.eval()
print("\n\n")
print("Loaded net:", l_net.state_dict())






    

    



