#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:12:06 2019

@author: ben
"""
import numpy as np

steps = 1
t = np.zeros(steps)
i = 0
r = 1
g = 0.99

def q0(t, r):
    gg = 1
    l = len(t)
    for j in range(len(t)):
        t[l - j - 1] += (gg * r)
        gg *= g

def q(t, r, s):
    gg = 1
    for j in range(steps):
        t[(s + steps - j) % steps] += (gg * r)
        gg *= g
'''
for i in range(0, steps):
    r = np.random.randint(1,10)
    q0(t[0:i+1], r)
    print(t, r)
'''   
print()
for i in range(0, steps * 2):
    r = 1 #np.random.randint(1,10)
    i = i % steps
    print(t[i], i)
    t[i] = 0
    q(t, r, i)
    print(t, r)
    




