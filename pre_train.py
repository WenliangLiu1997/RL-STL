#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:48:30 2020

@author: liuwenliang
"""

import torch
import numpy as np

from torch import nn
import torch.utils.data as Data
import datetime
import torch.nn.functional as F


def pre_train(Q,U,rnn):
    # Hyper Parameters
    EPOCH = 300        
    BATCH_SIZE = 256
    LR = 0.005               # learning rate
    
    

    
    Q = Q[:,:-1,:]
    print(Q.size(),U.size())
    Dataset = Data.TensorDataset(Q, U)
    
    
    loader = Data.DataLoader(
        dataset=Dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               
        num_workers=0,              
    )

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  
    loss_func = nn.MSELoss()                       
    start = datetime.datetime.now()
    # training and testing
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(loader):        # gives batch data
    
            output, h_state = rnn(b_x,None)                               # rnn output
            loss = loss_func(output, b_y)                 
            optimizer.zero_grad()                           # clear gradients for this training step
            loss.backward()                                 # backpropagation, compute gradients
            optimizer.step()                                # apply gradients
            
            if step % 50 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
    # torch.save(rnn, 'rnn_pre_train.pkl')
    end = datetime.datetime.now()
    print('Training time:')
    print(end-start)
    return rnn
