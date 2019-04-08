#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:13:55 2019

@author: ben
"""

import torch

print(torch.cuda.current_device())
device_name = torch.cuda.get_device_name(0)
print(device_name)

print(torch.version.cuda)