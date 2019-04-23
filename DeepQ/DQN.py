#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:07:11 2019

@author: ben
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy

def dqn_loss_target(net, states_v, next_states_v, actions_v, rewards_v, done_mask, gamma):
         # Q(s_t, a_t) = r_t + gamma * max_a Q'(s_t+1, a)
        q_vals = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)  #compute s_t q values from primary network
        q_vals_n = net.target_net(next_states_v).max(1)[0]               #compute s_t+n q values from target network
        q_vals_n[done_mask] = 0.0 #mask q_values if s_t+1 is terminal
        expected_q_vals = q_vals_n.detach() * gamma + rewards_v
        return nn.MSELoss()(q_vals, expected_q_vals)
    
def dqn_loss_target_double(net, states_v, next_states_v, actions_v, rewards_v, done_mask, gamma):
    #compute q values from primary network
    q_vals = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    # Q(s_t, a_t) = r_t + gamma * max_a Q'(s_t+1, argmax_a Q(s_t+1, a))
    actions_n = net(next_states_v).max(1)[1]                                                   #get qmax actions from primary network argmax_a Q(s_t+1, a)
    q_vals_n = net.target_net(next_states_v).gather(1, actions_n.unsqueeze(-1)).squeeze(-1)    #calculate q values from secondary network Q'(s_t, _) 
    q_vals_n[done_mask] = 0.0 #mask q_values if s_t+1 is terminal
    expected_q_vals = q_vals_n.detach() * gamma + rewards_v
    return nn.MSELoss()(q_vals, expected_q_vals)

def dqn_loss(net, states_v, next_states_v, actions_v, rewards_v, done_mask, gamma):
         # Q(s_t, a_t) = r_t + gamma * max_a Q'(s_t+1, a)
        q_vals = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)  #compute s_t q values
        q_vals_n = net(next_states_v).max(1)[0]                                #compute s_t+n q values
        q_vals_n[done_mask] = 0.0 #mask q_values if s_t+1 is terminal
        expected_q_vals = q_vals_n.detach() * gamma + rewards_v
        return nn.MSELoss()(q_vals, expected_q_vals) 
    
class DQNT(nn.Module):
   
    def __init__(self, net, double):
        super(DQNT, self).__init__()
        self.net = net
        self.target_net = copy.deepcopy(net)
        if double:
            self.loss = dqn_loss_target_double
        else:
            self.loss = dqn_loss_target

    def sync(self):
        """
            update the parameters of the target network with the real network.
        """
        self.target_net.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.net.state_dict()
        tgt_state = self.target_net.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_net.load_state_dict(tgt_state)
        
    def forward(self, x):
        return self.net.forward(x)
     

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.loss = dqn_loss

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
    
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()
        self.loss = dqn_loss

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.data, bias)


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise

    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))

    def forward(self, input):
        self.epsison_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)



class NoisyDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.noisy_layers = [
            NoisyLinear(conv_out_size, 512),
            NoisyLinear(512, n_actions)
        ]
        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

    def noisy_layers_sigma_snr(self):
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]

