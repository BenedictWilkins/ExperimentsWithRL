#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:53:22 2019

@author: ben
"""

from pyworld import agent as pwag
from pyworld import sim as pwsim
from torch import optim


class Model:
    
    def __init__(self, model):
        self.model = model
        

class TModel(Model):
    """
    Wrapper around model which provides copy of it instead of trained weights.
    CREDIT: ptan library.
    """
    def __init__(self, model):
        super(TModel, self).__init__(model)
        self.target_model = copy.deepcopy(model)

    def sync(self):
        """
            update the parameters of the target network with the real network.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)
        

class DQNAgent(pwag.LearningAgent):
        
    DEFAULT_PARAMETERS = {'learning_rate':0.001, 
              'batch_size':8, 
              'reward_steps':6, 
              'gamma':0.99, 
              'double':True}
    
    def __init__(self, net, **params):
        sensors = [pwag.UnrollSensor(params['batch_size'], self.sense, params['gamma'], params['reward_steps'])]
        actuators = [pwag.ProbabilisticActuator()]
        if params['double']:
            model = TModel(net)
        else:
            model = Model(net)
            
        super(DQNAgent,self).__init__(model=model,
                                             optimizer=optim.Adam(net.parameters(), params['learning_rate']),
                                             sensors = sensors, 
                                             actuators = actuators)
        

    def sense(self, obs):
        pass
    
    def attempt(self, state):
        



            
    
    