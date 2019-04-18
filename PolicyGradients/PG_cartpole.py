#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:33:10 2019

@author: ben
"""
import gym


import pyworld.common as pyw
from pyworld.common import Info

from Policy_Gradient import PolicyGradientAgentImproved, PolicyGradientNetwork
from tensorboardX import SummaryWriter


class PolicyGradientAgent(pwag.LearningAgent):
    '''
        Implementaion of a PolicyGradientAgent that uses entropy regularisation and a baseline.
    '''
    
    def __init__(self, net, gamma = 0.99, learning_rate = 0.01, batch_size = 16, entropy_beta = 0.01, debug = None):
        sensors = [pwag.UnrollSensor(self.sense, gamma, 6)]
        actuators = [pwag.ProbabilisticActuator()]
        super(PolicyGradientAgent,self).__init__(model=net,
                                                         optimizer=optim.Adam(net.parameters(), lr=learning_rate),
                                                         batch_labels=['state','action','g'], 
                                                         batch_size=batch_size, 
                                                         sensors = sensors, 
                                                         actuators = actuators)
        self.entropy_beta = entropy_beta
        self.baseline = 0

    
    def sense(self, obs):
        '''
            Unrolled sense method that uses an unrolled part of the trajectory
            args: obs = (pstate, nstate,  action, unrolled_reward, time), where time = (epsiode, step, gobal_step end).
        '''
        (pstate, action, unrolled_reward, nstate, time) = obs
        
        #moving average of baseline (mean unrolled reward)
        self.baseline = self.baseline + ((unrolled_reward - self.baseline) / time.global_step)
        
        # add observation to thecurrent batch 
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
                
    def attempt(self, state):
        out = self.model(torch.FloatTensor(state))
        probs_v = nnf.softmax(out, dim=0)
        self.actuators[0](probs_v) #attempt the action based on action probs
            
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

summary_writer = SummaryWriter(comment="-cartpole-pg")
debug = Info(summary_writer)
#ag = ReinforceAgent(PolicyGradientNetwork(input_shape,output_shape), batch_size=4, summary_writer=summary_writer)

ag = PolicyGradientAgentImproved(PolicyGradientNetwork(input_shape,output_shape), batch_size=4)
sim = pyw.GymSimulator(env, ag, debug)


print('Training: ', ENV)
for (episode, step, global_step, done) in sim:
    avg_reward = debug.info[debug.info_labels[0]]
    if episode > 10000 or avg_reward > 2000: 
        break

sim.close()

########## TEST and render

print("TEST!")

#env = gym.make('CartPole-long-v0')

env = gym.wrappers.Monitor(env, './videos', force=True)
sim = pyw.GymSimulator(env, ag)

sim.render = True

for (episode, step, global_step, done) in sim:
    if done or step > 2000:
        break

sim.close()

