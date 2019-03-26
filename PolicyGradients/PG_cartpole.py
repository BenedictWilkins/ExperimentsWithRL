#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:33:10 2019

@author: ben
"""
from Policy_Gradient import *
from pyworld.common import * 


gym.register(
    id='CartPole-long-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 5000},
    reward_threshold=4750.0,
)

ENV = 'CartPole-long-v0'
env = gym.make(ENV)
input_shape = env.observation_space.shape[0]
output_shape = env.action_space.n
print('observation dim:', (input_shape, ))
print('action dim:     ', (output_shape, ))

summary_writer = SummaryWriter(comment="-cartpole-pg")
debug = InfoWriter(summary_writer)
#ag = ReinforceAgent(PolicyGradientNetwork(input_shape,output_shape), batch_size=4, summary_writer=summary_writer)

ag = PolicyGradientAgentImproved(PolicyGradientNetwork(input_shape,output_shape), batch_size=4)
sim = pyw.GymSimulator(env, ag, debug)


print('Training: ', ENV)
for (episode, step, global_step, done) in sim:
    if episode > 10000: 
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
