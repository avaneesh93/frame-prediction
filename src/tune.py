#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 22:10:50 2018

@author: roshanprakash
"""

from dataset_loader import *
from encode_decode import *
from train_model import *
from test_model import *
import matplotlib.pyplot as plt
import numpy as np
import os

# first load the data
print('Loading data...')
data = dataset_loader(delta_t = 40.0, k = 10, offset = 10) 
print('Loaded data!')

# setup some data specific variables
dt = data.delta_t
X_train = data.X_train['walking']
y_train = data.y_train['walking']
opt_flows_train = data.optical_flows_train['walking']
motion_representations_train = data.motion_representations_train['walking']

X_test = data.X_test['walking']
y_test = data.y_test['walking']
opt_flows_test = data.optical_flows_test['walking']

# now, tune the model
learning_rates = [1e-7, 1e-6, 1e-4, 1e-3, 1e-2]
losses = []
best_rate = None
best_loss = np.inf

"""
Runs 100 epochs with different learning rates and writes losses to file
"""

for lr in learning_rates:
    print('Training for learning rate {}'.format(lr))
    epoch_losses = train_model(X_train, y_train, dt, \
                               opt_flows_train, motion_representations_train, baseline = False, \
                                lr, epochs = 100, tune = True, print_every = 10)
    print()
    losses.append(epoch_losses)
    if epoch_losses[-1] < best_loss:
        best_loss = epoch_losses[-1]
        best_rate = lr  
    np.save(os.path.dirname(os.getcwd()) + '/tune/{}_lr'.format(lr), epoch_losses)
print('Best learning rate found for 100 epochs of training {}'.format(best_rate))
