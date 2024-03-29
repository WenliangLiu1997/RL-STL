#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:44:42 2020

@author: liuwenliang
"""

import sys
sys.path.insert(0, 'src')
import stlcg_new as stlcg

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch import nn
import math
import torch.utils.data as Data
import torch.nn.functional as F
import datetime



dt = 1/3
T = 1000 / dt
dp = 0.05
Di = 0.7



nb_agent = 3
Q_dim = 4*nb_agent     
U_dim = 2*nb_agent  



class Regressor_g(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.linear = nn.Linear(input_dim, output_dim)
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dp)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(self.dropout(x)))
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x * self.mask1))
        return self.linear3(self.dropout(x))
    
class Regressor_f(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.linear = nn.Linear(input_dim, output_dim)
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dp)
        
    def forward(self, x):
        y = F.relu(self.linear1(x))
        y = F.relu(self.linear2(self.dropout(y)))
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x * self.mask1))
        return self.linear3(self.dropout(y)) + x


def system(q0, steps): # q0: [1, q_dim]
    # system dynamic
    q = q0
    Q = q0
    U = torch.empty((0,U_dim))
    for i in range(steps):
        q_new = torch.zeros((1,Q_dim))
        u = torch.rand((1,U_dim))
        u[0,[0,2,4]] = u[0,[0,2,4]]*0.7*dt
        u[0,[1,3,5]] = (u[0,[1,3,5]]-0.5)*2*1.5*dt
        noise = torch.zeros((1,Q_dim))
        noise[:,[0,1,4,5,8,9]] = torch.normal(mean=0,std=0.05*torch.ones((1,nb_agent*2)))
        for j in range(nb_agent):
            q_new[0,4*j+0] = q[0,4*j+0] + u[0,2*j+0]/u[0,2*j+1] * (q[0,4*j+2]*torch.cos(u[0,2*j+1]) + q[0,4*j+3]*torch.sin(u[0,2*j+1]) - q[0,4*j+2])
            q_new[0,4*j+1] = q[0,4*j+1] + u[0,2*j+0]/u[0,2*j+1] * (q[0,4*j+3] - q[0,4*j+3]*torch.cos(u[0,2*j+1]) + q[0,4*j+2]*torch.sin(u[0,2*j+1]))
            q_new[0,4*j+2] = q[0,4*j+2]*torch.cos(u[0,2*j+1]) + q[0,4*j+3]*torch.sin(u[0,2*j+1])
            q_new[0,4*j+3] = q[0,4*j+3]*torch.cos(u[0,2*j+1]) - q[0,4*j+2]*torch.sin(u[0,2*j+1])
        # q_new[0,[0,1,4,5,8,9]] = q[0,[0,1,4,5,8,9]] + u[0,:]
        q = q_new
        Q = torch.cat((Q,q+noise),dim=0) # Q: [time_step+1, q_dim]
        U = torch.cat((U,u),dim=0) # U: [time_step, u_dim]
        
        d_12 = torch.sqrt((q[:,0] - q[:,4])**2 + (q[:,1] - q[:,5])**2).view(-1)
        d_13 = torch.sqrt((q[:,0] - q[:,8])**2 + (q[:,1] - q[:,9])**2).view(-1)
        d_23 = torch.sqrt((q[:,4] - q[:,8])**2 + (q[:,5] - q[:,9])**2).view(-1)
        if (d_12<Di).float()+(d_13<Di).float()+(d_23<Di).float() > 0:
            break
        out_of_bound = 0
        obstacle = 0
        for k in range(nb_agent):
            if q[0,4*k+0]<0 or q[0,4*k+0]>10 or q[0,4*k+1]<1 or q[0,4*k+1]>9:
                out_of_bound = 1
        if out_of_bound == 1 or obstacle == 1:
            break

    return Q,U

batch_size = 40

Q0_test = torch.zeros((batch_size,Q_dim))
batch=0
while batch<batch_size:
    q0_test = torch.rand((1,Q_dim))
    theta_test = (torch.rand((1,nb_agent))-0.5)*math.pi/2
    theta_test[:,2] += math.pi 
    for i in range(nb_agent-1): 
        q0_test[:,4*i]   = 1.5*q0_test[:,4*i]+0.5
        q0_test[:,4*i+1] = q0_test[:,4*i+1]*4+3
        q0_test[:,4*i+2] = torch.sin(theta_test[:,i])
        q0_test[:,4*i+3] = torch.cos(theta_test[:,i])
    q0_test[:,8] = 1.5*q0_test[:,8] + 8
    q0_test[:,9] = 2 * q0_test[:,9] + 4
    q0_test[:,10] = torch.sin(theta_test[:,2])
    q0_test[:,11] = torch.cos(theta_test[:,2]) 
    d0_test_12 = torch.sqrt((q0_test[:,0] - q0_test[:,4])**2 + (q0_test[:,1] - q0_test[:,5])**2).view(-1)
    d0_test_13 = torch.sqrt((q0_test[:,0] - q0_test[:,8])**2 + (q0_test[:,1] - q0_test[:,9])**2).view(-1)
    d0_test_23 = torch.sqrt((q0_test[:,4] - q0_test[:,8])**2 + (q0_test[:,5] - q0_test[:,9])**2).view(-1)       

    if (d0_test_12>Di).float()+(d0_test_13>Di).float()+(d0_test_23>Di).float()==3:
        Q0_test[batch,:] = q0_test[0,:]
        batch=batch+1


X = torch.empty((0,Q_dim+U_dim))
Y = torch.empty((0,Q_dim))
with torch.no_grad():
    for i in range(batch_size):
        Q, U = system(Q0_test[i,:].view(1,Q_dim),int(T))
        x = torch.cat((Q[:-1,:],U),dim=1) # x: [time_step, q_dim+u_dim]
        Y  = torch.cat((Y,Q[1:,:]),dim=0)
        X = torch.cat((X,x),dim=0)

        ax = plt.gca()
        ax.cla() # clear things for fresh plot
        rectF = patches.Rectangle((8,4),1.5,2,edgecolor='none',facecolor='lightblue')
        rectH = patches.Rectangle((0.5,3),1.5,4,edgecolor='none',facecolor='lightblue')
        rectB = patches.Rectangle((0,1),10,8.0,edgecolor='g',facecolor='none')
        # rectO = patches.Rectangle((6,2.5),1.2,1.2,edgecolor='none',facecolor='silver')
        
        ax.add_patch(rectF)
        ax.add_patch(rectH)
        ax.add_patch(rectB)
        # ax.add_patch(rectO)
        circle1 = plt.Circle([5,5], 0.94, color='lightcoral', fill=True)
        ax.add_artist(circle1)
        circle2 = plt.Circle([7,7], 0.36, color='gray', fill=True)
        ax.add_artist(circle2)
        
        # circle2 = plt.Circle([5.,5.], 3., color='g', fill=False)
        # ax.add_artist(circle2)
        plt.text(8.15,4.15,'$R_{FS}$',fontsize=14)
        plt.text(0.65,3.15,'$R_{Hos}$',fontsize=14)
        # plt.text(6.05,2.65,'$R_{Obs}$',fontsize=14)
        plt.text(4.7,4.2,'$R_{F}$',fontsize=14)
        plt.text(6.9,7.4,'$R_{Hyd}$',fontsize=14)
        ax.set_xlim((-0.02, 10.02))
        ax.set_ylim((0.98, 9.02))
        plt.axis('off')        
        plt.plot(Q[:,0], Q[:,1])
        plt.scatter(Q[:,0], Q[:,1],alpha=0.6, zorder = 30)
        plt.plot(Q[:,4], Q[:,5])
        plt.scatter(Q[:,4], Q[:,5],alpha=0.6, zorder = 30)
        plt.plot(Q[:,8], Q[:,9])
        plt.scatter(Q[:,8], Q[:,9],alpha=0.6, zorder = 30)
        ax.set_aspect(1)
        plt.show()

# regressor = Regressor_model(Q_dim+U_dim,Q_dim)
regressor_f = Regressor_f(Q_dim,Q_dim)
regressor_g = Regressor_g(Q_dim,Q_dim*U_dim)
 
# regressor.cuda()
Dataset = Data.TensorDataset(X, Y)
loader = Data.DataLoader(
    dataset=Dataset,      # torch TensorDataset format
    batch_size=32,      # mini batch size
    shuffle=True,               
    num_workers=0,              
)


optimizer = torch.optim.Adam(list(regressor_f.parameters()) + list(regressor_g.parameters()), lr=0.001)
criterion = nn.MSELoss()

start=datetime.datetime.now()

best_loss = 10
for epoch in range(200):
    for i, (datapoints, labels) in enumerate(loader):
        # datapoints = datapoints.cuda()
        # labels = labels.cuda()
        optimizer.zero_grad()
        loss = criterion(regressor_f(datapoints[:,:Q_dim]) + torch.bmm(regressor_g(datapoints[:,:Q_dim]).view(-1,Q_dim, U_dim),datapoints[:,Q_dim:].view(-1,U_dim,1)).view(-1,Q_dim), labels)
        if loss < best_loss:
            torch.save(regressor_f.state_dict(), 'system_model_f0.pkl')
            torch.save(regressor_g.state_dict(), 'system_model_g0.pkl')
            best_loss = loss
        loss.backward()
        optimizer.step()
        
        # regressor.eval()
        # loss_t = criterion(regressor(X_t),Y_t)
        # regressor.train()
        if i % 20 == 0: 
            print('Epoch: ', epoch, '| train loss: %.5f' % loss.data)
end = datetime.datetime.now()

print(end-start)

# loss_t = criterion(regressor_f(X[1000:,:Q_dim]) + torch.bmm(regressor_g(X[1000:,:Q_dim]).view(-1,Q_dim, U_dim),X[1000:,Q_dim:].view(-1,U_dim,1)).view(-1,Q_dim), Y[1000:])

# print(loss_t)
print(X.size())
np.save('Model_dataset_X',X.numpy())
np.save('Model_dataset_Y',Y.numpy())

