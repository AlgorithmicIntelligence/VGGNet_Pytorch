#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:46:40 2020

@author: nickwang
"""

import numpy as np
import matplotlib.pyplot as plt

plt.figure()
plt.xlabel('epochs')
plt.ylabel('TrainAccuracy')
plt.ylim(0, 1)
legend_list = list()
for model_mode in ['A', 'B', 'C', 'D', 'E']:
    train_accuracy_list = np.load('train_acc_{}_epoch1-74.npy'.format(model_mode))
    x = np.arange(len(train_accuracy_list) + 1)
    plt.plot(x, [0] + list(train_accuracy_list))
    legend_list.append('VGG_{}'.format(model_mode))
plt.legend(legend_list, loc='upper right')
plt.grid(True)
plt.savefig('./README/TrainAccuracy.png') 
plt.show()

plt.figure()
plt.xlabel('epochs')
plt.ylabel('TestAccuracy')
plt.ylim(0, 1)
legend_list = list()
for model_mode in ['A', 'B', 'C', 'D', 'E']:
    test_accuracy_list = np.load('test_acc_{}_epoch1-74.npy'.format(model_mode))
    x = np.arange(len(test_accuracy_list) + 1)
    plt.plot(x, [0] + list(test_accuracy_list))
    legend_list.append('VGG_{}'.format(model_mode))
plt.legend(legend_list, loc='upper right')
plt.grid(True)
plt.savefig('./README/TestAccuracy.png') 
plt.show()

plt.figure()
plt.xlabel('epochs')
plt.ylabel('TrainLoss')
legend_list = list()
for model_mode in ['A', 'B', 'C', 'D', 'E']:
    train_loss_list = np.load('train_loss_{}_epoch1-74.npy'.format(model_mode))
    x = np.arange(len(train_loss_list) + 1)
    plt.plot(x, list(train_loss_list[0:1]) + list(train_loss_list))
    legend_list.append('VGG_{}'.format(model_mode))
plt.legend(legend_list, loc='upper right')
plt.grid(True)
plt.savefig('./README/TrainLoss.png') 
plt.show()

plt.figure()
plt.xlabel('epochs')
plt.ylabel('TestLoss')
legend_list = list()
for model_mode in ['A', 'B', 'C', 'D', 'E']:
    test_loss_list = np.load('test_loss_{}_epoch1-74.npy'.format(model_mode))
    x = np.arange(len(test_loss_list) + 1)
    plt.plot(x, list(test_loss_list[0:1]) + list(test_loss_list))
    legend_list.append('VGG_{}'.format(model_mode))
plt.legend(legend_list, loc='upper right')
plt.grid(True)
plt.savefig('./README/TestLoss.png') 
plt.show()

