#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:17:25 2019

@author: ben
"""

import pyworld.common as pyw

ag = pyw.RandomAgent()
sim = pyw.GymSimulator('CartPole-v0', ag)


'''
for t in sim:
    (episode, step, end) = t
    print(t)
    if episode >= 3 and end:
        break
'''