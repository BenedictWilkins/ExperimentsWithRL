#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:53:22 2019

@author: ben
"""

from pyworld import agent as pwag
from pyworld import simulate as pwsim
from pyworld import common as pwcom


from tensorboardX import SummaryWriter
import torch
from torch import optim
from torch import nn

import DQN as dqn
import numpy as np
import cv2

class DQNAgent(pwag.Agent):
    
    def __init__(self, device, gamma=0.99):
        super(DQNAgent,self).__init__()
        self.device = device
        self.gamma = gamma
        self.current_state = None
    
    def reset(self, obs):
        state, _ = obs
        self.current_state = state[np.newaxis] # makes into batch of size 1
        
    def sense(self, obs):
        state, action, reward, nstate, time = obs
        self.current_state = nstate[np.newaxis] # makes into batch of size 1
        cv2.imshow("s1", state[0])
        #add the tuple to experience replay, batches will be sampled from it for training
        self.experience_replay.append(state, action, reward, nstate, int(time.done))
        #sample a batch and relative importance of each sample from experience replay buffer
        batch = self.experience_replay.sample() #TODO prioritised
        self.train(batch)
        
    def attempt(self):
        q_vals = self.model(torch.FloatTensor(self.current_state).to(self.device))
        self.actuator(q_vals.detach().squeeze().cpu().numpy())
        
    def train(self, batch):
        #unpack batch
        states, actions, rewards, next_states, dones = batch
        # Construct tensors from batch
        states_v = torch.FloatTensor(states).to(self.device)
        next_states_v = torch.FloatTensor(next_states).to(self.device)
        actions_v = torch.LongTensor(actions).to(self.device)
        rewards_v = torch.FloatTensor(rewards).to(self.device)
        done_mask = torch.ByteTensor(dones).to(self.device)
        self.optim.zero_grad()
        loss_v = self.loss(states_v, next_states_v, actions_v, rewards_v, done_mask)
        loss_v.backward()
        self.optim.step()
        
        
    def loss(self, states_v, next_states_v, actions_v, rewards_v, done_mask):
        return self.model.loss(self.model, states_v, next_states_v, actions_v, rewards_v, done_mask, self.gamma)
    
    def DQN(observation_shape, action_shape, **params):
        if params.get('noisy', False):
            net = dqn.NoisyDQN(observation_shape, action_shape)
        else:
            net = dqn.DQN(observation_shape, action_shape)
        if params.get('target', False):
            net = dqn.DQNT(net, params['double'])
        return net.to(params.get('device', 'cpu'))

if __name__ == "__main__":

    #change me!
    DQN_NOISY = False       # use a noisy network?
    DQN_DOUBLE = True       # the double DQN loss? (requires a target network)
    DQN_TARGET = True       # use a target network? (you should!)  
    
    LEARNING_RATE = 0.0001   # learning rate of the optimizer
    BATCH_SIZE = 8          # size of batches to pass to the DQN
    REWARD_STEPS = 6        # number of steps to look in future (discounted reward)
    GAMMA = 0.99            # reward discount (for infinite horizon problems)
    SKIP = 3                # number of frames to skip at each step (frames come in too fast, take the nth frame as the observation)
    STACK = 4               # number of frames to stack. helps make the state space markovian
   
    EPSILON_START = 1.0          # epsilon for e-greedy policy
    EPSILON_END = 0.02     
               
    if(torch.cuda.is_available()): # use cuda (GPU compute for faster training!)
        print("USING CUDA!")
        DEVICE = 'cuda'
    else:
        DEVICE  = 'cpu'
    
    sim = pwsim.GymSimulator('Breakout-v0')
    
    action_shape = sim.env.action_space.n
    observation_shape = [STACK, 84, 84] #todo find a way to do this nicely
    
    ag = DQNAgent(DEVICE, GAMMA)
    
    epsilon_tracker = pwag.EpsilonTracker(epsilon_start = EPSILON_START, epsilon_end = EPSILON_END)
    actuator = pwag.EpsilonGreedyActuator(epsilon_tracker)
    sensor = pwag.MaxPoolSensor(pwag.AtariImageSensor(pwag.BufferedSensor(ag))) 
    
    ag.add_component('sensor', sensor)
    ag.add_component('actuator', actuator)
    ag.add_component('experience_replay', pwag.ExperienceReplay())

    model = DQNAgent.DQN(observation_shape, action_shape, double=DQN_DOUBLE, target=DQN_TARGET, noisy=DQN_NOISY, device=DEVICE)
    ag.add_component('model', model)
    
    optimizer = optim.Adam(ag.model.parameters(), lr=LEARNING_RATE)
    ag.add_component('optim', optimizer)
    
    info = pwcom.Info(SummaryWriter(), 1000)
    ag.add_component('info', info) #logger
    
    sim.add_agent(ag)

    for t in sim:
       if cv2.waitKey(10) == ord('q'):
           sim.stop()
             
    cv2.destroyAllWindows()
    


            
    
    