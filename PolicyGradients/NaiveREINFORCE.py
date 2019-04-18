#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:15:48 2019

@author: ben
"""

from pyworld import agent as pwag

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim


class PolicyGradientNetwork(nn.Module):
    
    def __init__(self, input_size, actions_size,):
        super(PolicyGradientNetwork, self).__init__()
        
        self.net = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, actions_size))
        
    def forward(self, x):
        return self.net(x)

class ReinforceAgent(pwag.LearningAgent):
    '''
        Reinforce agent without baseline.
    '''
    def __init__(self, net, gamma = 0.99, learning_rate = 0.01, batch_size = 16, summary_writer = None):
        sensors = [pwag.EpisodicSensor(self.sense)]
        actuators = [pwag.ProbabilisticActuator()]
        super(ReinforceAgent,self).__init__(model=net,
                                            optimizer=optim.Adam(net.parameters(), lr=learning_rate),
                                            sensors = sensors, actuators = actuators)
        self.net = net
        self.gamma = gamma
        self.batch_counter = 0
        
    def q_values(self, rewards):
        res = []
        sum_r = 0.0
        for r in reversed(rewards):
            sum_r *= self.gamma
            sum_r += r
            res.append(sum_r)
        return list(reversed(res))
    
    def sense(self, obs):
        self.train(obs)
        #print('INFO', episode, ': total reward:', np.mean(self.total_rewards[-20:]), 'loss: ', np.mean(self.losses[-20:]))
            
    def attempt(self, state):
        out = self.model(torch.FloatTensor(state))
        probs_v = nnf.softmax(out, dim=0)
        self.actuators[0](probs_v) #attempt the action based on action probs
            
    def train(self, batch):
        '''
            Performs one gradient step using the current episode data.
        '''
        self.optimizer.zero_grad()
        states_v = torch.FloatTensor(batch.state)
        actions_v = torch.LongTensor(batch.action)
        qs_v = torch.FloatTensor(self.q_values(batch.reward))
        objective = self.loss(states_v, actions_v, qs_v)
        objective.backward()
        self.optimizer.step()
        #self.losses.append(objective.item())
    
    def loss(self, states_v, actions_v, qs_v):
        '''
            Loss function for PolicyGradient: - q * logprob actions
        '''
        logits_v = self.net(states_v)
        log_prob_v = nnf.log_softmax(logits_v, dim=1)
        log_prob_actions_v = qs_v * log_prob_v[range(states_v.shape[0]), actions_v]
        return - log_prob_actions_v.mean()
    
    
if __name__ == "__main__":
    
    import gym
    from pyworld import simulate as pwsim
    from pyworld.common import Info
    from tensorboardX import SummaryWriter
    
    TIMEOUT = 1000
    
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
    
    ag = ReinforceAgent(PolicyGradientNetwork(input_shape,output_shape))
    sim = pwsim.GymSimulator(env, info)
    sim.add_agent(ag)
    
    print('Training: ', ENV)
    for t in sim:
        avg_reward = info.info[info.info_labels[0]]
        if t.episode > TIMEOUT or avg_reward > 2000: 
            break
    
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
    
   

    
        
        