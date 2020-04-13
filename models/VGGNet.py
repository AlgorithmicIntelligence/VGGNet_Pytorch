#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:56:01 2020

@author: lds
"""

import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()
    def forward(self, x):
        return torch.squeeze(x)
        
class VGGNet(nn.Module):
    def __init__(self, num_classes, mode='E'):
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        self.mode = mode     
        self.features = self.feature_build()
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        ) 
        self.squeeze = Squeeze()
        self.feature_maps = dict()
        self.pool_locs = dict()
        
    def feature_build(self):
        features_layer_info = {
            'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
            'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }  
        features_dict = OrderedDict()
        name_level = 1
        name_sublevel = 1
        in_channels = 3
        for idx, layer_info in enumerate(features_layer_info[self.mode]):
            if layer_info == 'M':
                features_dict['{}_{}'.format(name_level, name_sublevel)] = nn.MaxPool2d(kernel_size=2, stride=2)
                name_level += 1
                name_sublevel = 1
                              
            elif self.mode == 'C' and idx in [8, 12, 16]:
                features_dict['{}_{}'.format(name_level, name_sublevel)] = nn.Conv2d(in_channels, layer_info, kernel_size=1)
                name_sublevel += 1
                features_dict['{}_{}'.format(name_level, name_sublevel)] =  nn.ReLU(inplace=True)
                name_sublevel += 1
                in_channels = layer_info
            else:
                features_dict['{}_{}'.format(name_level, name_sublevel)] = nn.Conv2d(in_channels, layer_info, kernel_size=3, padding=1)
                name_sublevel += 1
                features_dict['{}_{}'.format(name_level, name_sublevel)] =  nn.ReLU(inplace=True)
                name_sublevel += 1
                in_channels = layer_info
        return nn.Sequential(features_dict)
    
    def init_weights(self, init_mode='VGG'):
        def init_function(m, init_mode):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_mode == 'VGG':
                    torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                elif init_mode == 'XAVIER': 
                    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                    std = (2.0 / float(fan_in + fan_out)) ** 0.5
                    a = (3.0)**0.5 * std
                    with torch.no_grad():
                        m.weight.uniform_(-a, a)
                elif init_mode == 'KAMING':
                     torch.nn.init.kaiming_uniform_(m.weight)
                
                m.bias.data.fill_(0)    
        _ = self.apply(partial(init_function, init_mode=init_mode))
    def store_feature_maps(self):
        def hook(module, input, output, key):
            if isinstance(module, nn.MaxPool2d):
                self.feature_maps[key] = output[0]
                self.pool_locs[key] = output[1]
            else:
                self.feature_maps[key] = output
        for idx, layer in enumerate(self.features):
            layer.register_forward_hook(partial(hook, key=idx))
    
    def forward(self, x):        
        self.feature_maps = dict()
        self.pool_locs = dict()       
        x = self.features(x)     
        x = self.classifier(x)    
        x = self.squeeze(x)
        return x


