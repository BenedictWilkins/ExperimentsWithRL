#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:44:11 2019

@author: ben
"""

import torch.nn as nn
import torch

import common

class PolicyGradientAgent(common.Agent):
    
    def __init__(self, net, gamma = 0.99, learning_rate = 0.01):
        self.net = net
        self.gamma = gamma
        self.learning_rate = 0.01
    
    
    
    
    def q_values(self, rewards):
        res = []
        sum_r = 0.0
        for r in reversed(rewards):
            sum_r *= self.gamma
            sum_r += r
            res.append(sum_r)
        return list(reversed(res))
        
        
        
        
class PolicyGradientNetwork(nn.Module):
    
    def __init__(self, input_size, actions_size,):
        super(PolicyGradientNetwork, self).__init__()
        
        self.net = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, actions_size))
        
    def forward(self, x):
        self.net(x)
        

    
in_shape = env.observation_sapce.shape[0]
out_shape = env.action_space.n

env = gym.make("CartPole-v0")
writer = SummaryWriter(comment="-cartpole-reinforce")

net = PGN(in_shape, out_shape)
agent = ptan.agent.PolicyGradientAgent(net)

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
