#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:48:00 2019

@author: ben
"""

import math
from tensorboardX import SummaryWriter

import torch.nn as nn
import torch
import numpy as np

writer = SummaryWriter()

#Visualise some functions

funcs = {"sin": math.sin, "cos": math.cos, "tan": math.tan}
for angle in range(-360, 360):
        angle_rad = angle * math.pi / 180
        for name, fun in funcs.items():
            val = fun(angle_rad)
            writer.add_scalar(name, val, angle)

#Visualise a model

class ANN(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(ANN, self).__init__()
        self.pipe = nn.Sequential(
                nn.Linear(in_dim,5),
                nn.ReLU(),
                nn.Linear(5, 20),
                nn.ReLU(),
                nn.Linear(20, out_dim),
                nn.Dropout(p=0.3),
                nn.Softmax(dim=1))
    
    def forward(self, x):
        return self.pipe(x)

# the graph requires some dummy input to be evaluated before it can be visualised
# the tuple is the input args to the forward function of the module
writer.add_graph(ANN(10,10), (torch.FloatTensor(np.zeros((1,10))))) 

writer.close()

#command - tensorboard --logdir runs (or what ever name was given in SummaryWritterz)