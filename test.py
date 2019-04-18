#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:20:26 2019

@author: ben
"""

from pyworld import agent as pwag
from pyworld import simulate as pwsim

class TestAgent(pwag.Agent):
    
    def __init__(self, sensors, actuators):
        super(TestAgent,self).__init__(sensors, actuators)
        sensors[0]._callback = self.sense
        sensors[1]._callback = self.sense
        
    def sense(self, obs):
        print(obs)
    
    def attempt(self, state):
        self.actuators[0]()


sim = pwsim.GymSimulator('CartPole-v0')
ag = TestAgent([pwag.EpisodicSensor(), pwag.SimpleSensor()],[pwag.RandomActuator(sim.env.action_space)])
sim.add_agent(ag)

for t in sim:
    if t.done:
        sim.stop()
    print(t)