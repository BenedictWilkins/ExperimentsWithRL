#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:05:46 2019

@author: ben
"""

from pyworld import agent as pwag
from pyworld import simulate as pwsim
import collections
import cv2
import numpy as np
import gym
from abc import ABC, abstractmethod



class SensorTestAgent(pwag.Agent):
    
    def __init__(self, test_sensor, actuators, scale=2.0):
        super(SensorTestAgent, self).__init__([test_sensor, pwag.SimpleSensor(self.sense)], 
                                              actuators)
        test_sensor._callback = self.sense_test
        self.scale = scale
    
    def sense_test(self, obs):
        state, _, _, _, _ = obs
        print( )
        #torch tensor to opencv img
        state = np.reshape(state, (state.shape[1], state.shape[2], state.shape[0]))
        state = cv2.resize(state, (0,0), interpolation=cv2.INTER_AREA, fx=self.scale, fy=self.scale)
        cv2.imshow('Test Sensor', state)
        
    def sense(self, obs):
        state, _, _, _, _ = obs
        state = cv2.resize(state, (0,0), interpolation=cv2.INTER_AREA, fx=2, fy=2)
        cv2.imshow('Identity', state)
       
    def attempt(self, _):
        self.actuators[0]()

    
'''
if __name__ == "__main__":

    env = gym.make('Assault-v0')
    
    ag = SensorTestAgent(GymImageSensor(), [pwag.RandomActuator(env.action_space)], scale=4)
    sim = pwsim.GymSimulator(env)
    sim.add_agent(ag)
    
    for t in sim:
       if cv2.waitKey(10) == ord('q'):
           sim.stop()
             
    cv2.destroyAllWindows()
'''


