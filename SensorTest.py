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
    

class GymImageSensor:
    
    def __init__(self, frames=4):
        self.transform = GymImageTransform()
        self.buffer = np.zeros(self.transform.out_shape)
        
    def __call__(self, obs):
        state, action, reward, nstate, time = obs


class StateTransform(ABC):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, state):
        pass
    
class GymImageTransform(StateTransform):
    
    def __init__(self, out_shape=[1,84,84]):
        self.in_shape = [-1, -1, 3]
        self.out_shape = out_shape
    
    def __call__(self, state):
        if state.size == 210 * 160 * 3:
             img = np.reshape(state, [210, 160, 3]).astype(np.float32)
        elif state.size == 250 * 160 * 3:
            img = np.reshape(state, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = (img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114) 
        img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        return np.reshape(img[18:102, :], [1, 84, 84]) / 255.0
    
       
    
    
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
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        
'''