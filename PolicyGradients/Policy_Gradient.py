#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:44:11 2019

@author: ben
"""
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import gym
import pyworld.agent as pwag
import numpy as np

from tensorboardX import SummaryWriter


class PolicyGradientAgent(pwag.LearningAgent):
    '''
        Implementaion of a PolicyGradientAgent that uses entropy regularisation and a baseline.
    '''
    
    def __init__(self, net, gamma = 0.99, learning_rate = 0.001, batch_size = 8, reward_steps = 6, entropy_beta = 0.01, debug = None):
        sensors = [pwag.UnrollSensor(batch_size, self.sense, gamma, reward_steps)]
        actuators = [pwag.ProbabilisticActuator()]
        super(PolicyGradientAgent,self).__init__(model=net,
                                                 optimizer=optim.Adam(net.parameters(), lr=learning_rate),
                                                 sensors = sensors, 
                                                 actuators = actuators)
        self.entropy_beta = entropy_beta
        self.baseline = 0
        self.i = 0

    
    def sense(self, batch):
        #print(batch)
        self.train(batch)
                
    def attempt(self, state):
        out = self.model(torch.FloatTensor(state))
        probs_v = nnf.softmax(out, dim=0)
        self.actuators[0](probs_v) #attempt the action based on action probs
            
    def train(self, batch):
        '''
            Performs one gradient step using the current batch
        '''
        self.optimizer.zero_grad()
        states_v = torch.FloatTensor(batch.state)
        actions_v = torch.LongTensor(batch.action)
        rewards_v = torch.FloatTensor(batch.reward)
        #moving average of baseline (mean unrolled reward)
        for j in range(len(batch.reward)):
            self.i += 1
            self.baseline = self.baseline + ((batch.reward[j] - self.baseline) / (self.i))
        #print(states_v)
        #print(actions_v)
        #print(rewards_v)
        #print(batch.done)
        #for debug info
        self.summary_info['baseline'] = self.baseline
        #self.summary_info['targets'] = rewards_v - self.baseline
        g_v = rewards_v - self.baseline
        
        objective = self.loss(states_v, actions_v, g_v)
        objective.backward()
        #self.losses.append(objective.item())
        self.optimizer.step()
        
        #some more debug info for our summary writer
        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        
        for p in self.model.parameters():
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1
            grad_max = max(grad_max, p.grad.abs().max().item())
            
        self.summary_info['grads/grad_l2'] = grad_means / grad_count
        self.summary_info['grads/grad_max'] = grad_max

         
    def loss(self, states_v, actions_v, g_v):
        '''
            Loss function for PolicyGradient with entropy regulariser.
            Recall that entropy is minimal when out policy is sure about which action to take and we dont want our agent to be to sure.
            Given that we are trying to maximise q * log prob we should subtract the entropy term. 
        '''
        logits_v = self.model(states_v)
        log_prob_v = nnf.log_softmax(logits_v, dim=1)
        log_prob_actions_v = g_v * log_prob_v[range(states_v.shape[0]), actions_v]
        #print(log_prob_v)
        #print(g_v )
        #tensor([1.5598e-05, 1.8231e-06, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,-0.0000e+00, -0.0000e+00],
        policy_loss_v = - log_prob_actions_v.mean()
        
        prob_v = nnf.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = - self.entropy_beta * entropy_v
        loss_v = entropy_loss_v + policy_loss_v
        
        #TODO look at KL divergence
        
        #for debug/summary writer
        self.summary_info['loss/entropy_loss'] = entropy_loss_v.item()
        self.summary_info['loss/policy_loss'] = policy_loss_v.item()
        self.summary_info['loss/loss'] = loss_v.item()
        self.update_summary = True
        
        return loss_v
        
class PolicyGradientNetwork(nn.Module):
    
    def __init__(self, input_size, actions_size,):
        super(PolicyGradientNetwork, self).__init__()
        
        self.net = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, actions_size))
        
    def forward(self, x):
        return self.net(x)
    
if __name__ == "__main__":
    
    import gym
    from pyworld import simulate as pwsim
    from pyworld.common import Info
    from tensorboardX import SummaryWriter
    
    TIMEOUT = 10000
    TARGET_REWARD = 2000
    
    ENV = 'CartPole-long-v0'
    gym.register(
        id=ENV,
        entry_point='gym.envs.classic_control:CartPoleEnv',
        tags={'wrapper_config.TimeLimit.max_episode_steps': 5000},
        reward_threshold=4750.0,
    )
    

    env = gym.make(ENV)
    
    input_shape = env.observation_space.shape[0]
    output_shape = env.action_space.n
    print('observation dim:', (input_shape, ))
    print('action dim:     ', (output_shape, ))
    
    #logging/debug
    info = Info(SummaryWriter(comment="-cartpole-pg"))
    
    ag = PolicyGradientAgent(PolicyGradientNetwork(input_shape,output_shape))
    sim = pwsim.GymSimulator(env, info)
    sim.add_agent(ag)
    
    print('Training: ', ENV)
    for t in sim:
        avg_reward = info.info[info.info_labels[0]]
        if t.episode > TIMEOUT or avg_reward > TARGET_REWARD: 
            sim.stop()
    
    ########## TEST and render

    print("TEST!")
    
    #env = gym.make('CartPole-long-v0')
    
    env = gym.wrappers.Monitor(env, './videos', force=True)
    sim = pwsim.GymSimulator(env)
    sim.add_agent(ag)
    
    sim.render = True
    
    for t in sim:
        if t.done or t.step > 2000:
             sim.stop()
             
    env.close()
    
   

    
        
        
    




