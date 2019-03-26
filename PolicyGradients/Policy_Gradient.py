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
import pyworld.common as pyw
import numpy as np

from tensorboardX import SummaryWriter


class ReinforceAgent(pyw.LearningAgent):
    '''
        Reinforce agent without baseline.
    '''
    
    
    def __init__(self, net, gamma = 0.99, learning_rate = 0.01, batch_size = 16, summary_writer = None):
        #two types of sensors, one accumulates episodes for training, the other is the identity sensor which allows the agent to make decisions
        sensors = [pyw.EpisodicSensor(self.sense)]
        actuators = [pyw.ProbabilisticActuator()]
        super(ReinforceAgent,self).__init__(model=net,optimizer=optim.Adam(net.parameters(), lr=learning_rate),
                                                         batch_labels=['state','action','qs'], batch_size=batch_size, 
                                                         sensors = sensors, actuators = actuators)
        self.net = net
        self.gamma = gamma
        self.batch_counter = 0
        self.summary_writer = summary_writer
    
        self.total_rewards = []
        self.losses = []
        
    def q_values(self, rewards):
        res = []
        sum_r = 0.0
        for r in reversed(rewards):
            sum_r *= self.gamma
            sum_r += r
            res.append(sum_r)
        return list(reversed(res))
    
    def sense(self, obs):
        '''
            Sense method that is called when an episode is completed by the agents EpisodicSensor, this data is used to train the agent. 
            args: 
                obs: (states, actions, rewards, total_reward)
        '''
        (states, actions, rewards, episode, total_reward) = obs
        qs = self.q_values(rewards)
        self.batch.state.extend(states)
        self.batch.action.extend(actions)
        self.batch.qs.extend(qs)
        self.batch_counter += 1
        self.total_rewards.append(total_reward)
        if self.batch_counter > self.batch_size:
            self.train()
            self.batch.state.clear()
            self.batch.action.clear()
            self.batch.qs.clear()
            self.batch_counter = 0
            
            print('INFO', episode, ': total reward:', np.mean(self.total_rewards[-20:]), 'loss: ', np.mean(self.losses[-20:]))
            
    def attempt(self, state, action_space):
        '''
            Attempt method that provides action probabilities to the agents ProbabilisticActuator.
            Args: 
                state: current state of the environment
                action_space: actions that may be attempted in the current state
            Returns: 
                action_probs
        '''
        out = self.net(torch.FloatTensor(state))
        probs_v = nnf.softmax(out, dim=0)
        return probs_v
    
    def train(self):
        '''
            Performs one gradient step using the currently stored episode data.
        '''
        self.optimizer.zero_grad()
        states_v = torch.FloatTensor(self.batch.state)
        actions_v = torch.LongTensor(self.batch.action)
        qs_v = torch.FloatTensor(self.batch.qs)
        objective = self.loss(states_v, actions_v, qs_v)
        objective.backward()
        self.optimizer.step()
        self.losses.append(objective.item())
    
    def loss(self, states_v, actions_v, qs_v):
        '''
            Loss function for PolicyGradient: - q * logprob actions
        '''
        logits_v = self.net(states_v)
        log_prob_v = nnf.log_softmax(logits_v, dim=1)
        log_prob_actions_v = qs_v * log_prob_v[range(states_v.shape[0]), actions_v]
        return - log_prob_actions_v.mean()
    
    
class PolicyGradientAgentImproved(pyw.LearningAgent):
    '''
        Another implementaion of the PolicyGradientAgent that uses entropy regularisation and a baseline.
    '''
    
    def __init__(self, net, gamma = 0.99, learning_rate = 0.01, batch_size = 16, entropy_beta = 0.01, debug = None):
        sensors = [pyw.UnrollSensor(self.sense, gamma, 6)]
        actuators = [pyw.ProbabilisticActuator()]
        super(PolicyGradientAgentImproved,self).__init__(model=net,optimizer=optim.Adam(net.parameters(), lr=learning_rate),
                                                         batch_labels=['state','action','g'], batch_size=batch_size, 
                                                         sensors = sensors, actuators = actuators)
        self.entropy_beta = entropy_beta
        self.baseline = 0

    
    def sense(self, obs):
        '''
            Unrolled sense method that uses an unrolled part of the trajectory
            args: obs = (pstate, action,  nstate, unrolled_reward, time, rewards), where time = (epsiode, step, gobal_step end).
        '''
        (pstate, action, unrolled_reward, nstate, time) = obs
        
        #moving average of baseline (mean unrolled reward)
        self.baseline = self.baseline + ((unrolled_reward - self.baseline) / time.global_step)
        
        self.batch.state.append(pstate)
        self.batch.action.append(int(action))
        self.batch.g.append(unrolled_reward - self.baseline)
        
        #for debug info
        self.summary_info['baseline'] = self.baseline
        self.summary_info['targets'] = unrolled_reward - self.baseline
                
        if time.end:
            if time.episode % self.batch_size == 0:
                self.train()
                self.batch.state.clear()
                self.batch.action.clear()
                self.batch.g.clear()
                
    def attempt(self, state, _):
        '''
            Attempt method that provides action probabilities to the agents ProbabilisticActuator.
            Args: 
                state: current state of the environment
                
                action_space: actions that may be attempted in the current state
            Returns: 
                action_probs
        '''

        out = self.model(torch.FloatTensor(state))
        probs_v = nnf.softmax(out, dim=0)
        return probs_v
            
    def train(self):
        '''
            Performs one gradient step using the current batch
        '''
        self.optimizer.zero_grad()
        states_v = torch.FloatTensor(self.batch.state)
        actions_v = torch.LongTensor(self.batch.action)
        g_v = torch.FloatTensor(self.batch.g)
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
        policy_loss_v = - log_prob_actions_v.mean()
        
        prob_v = nnf.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = - self.entropy_beta * entropy_v
        loss_v = entropy_loss_v + policy_loss_v
        
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




    




