#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:18:43 2019

@author: ben
"""

import gym


# Basic cartpole setup
env = gym.make("CartPole-v0")
done = False
total_reward = 0
total_steps = 0
env.reset()
while not done:
    action = env.action_space.sample()
    env.render()
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    total_steps += 1
    
print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))





    




