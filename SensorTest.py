#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:05:46 2019

@author: ben
"""
q
from pyworld import agent as pwag
from pyworld import simulate as pwsim
import collections
import cv2
import numpy as np
import gym
from abc import ABC, abstractmethod



class SensorTestAgent(pwag.Agent):
    
    def __init__(self, scale=2.0):
        super(SensorTestAgent, self).__init__()
        self.scale = scale
        
    def sense(self, obs):
        action, reward, state, timeq = obs
        #torch tensor to opencv img
        state = np.reshape(state, (state.shape[1], state.shape[2], state.shape[0]))
        state = cv2.resize(state, (0,0), interpolation=cv2.INTER_AREA, fx=self.scale, fy=self.scale)
        cv2.imshow('Test Sensor', state)
       
    def attempt(self):
        self.actuators[0]()

if __name__ == "__main__":

    sim = pwsim.GymSimulator('Breakout-v0')
    agent = SensorTestAgent(scale=8.0)
    actuator = pwag.RandomActuator(sim.env.action_space)
   # sensor = pwag.MaxPoolSensor(pwag.AtariImageSensor(agent))
    sensor = pwag.SimpleSensor(agent)
    agent.__add_sensor__(sensor)
    agent.__add_actuator__(actuator)
    
    sim.add_agent(agent)

    for t in sim:
       if cv2.waitKey(10) == ord('q'):
           sim.stop()
             
    cv2.destroyAllWindows()

