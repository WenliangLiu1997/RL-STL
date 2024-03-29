#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:37:22 2020

@author: liuwenliang
"""

import torch
from torch import nn
import torch.utils.data as Data
import math
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
import gurobipy as gp
from gurobipy import GRB

l=0.01
alpha_c = 0.7
alpha_b = 0.7
gamma = 0.03
D = 0.3

nb_agent = 3
Q_dim = 4*nb_agent
U_dim = 2*nb_agent
dp = 0.05
nb_samples = 20
k=3


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


def std1(q_new,cov_mat): #q:[3]
    B_grad_q = np.zeros((1,Q_dim))
    B_grad_q[0,0] = 2*(q_new[0]-q_new[4])
    B_grad_q[0,1] = 2*(q_new[1]-q_new[5])
    B_grad_q[0,4] = 2*(q_new[4]-q_new[0])
    B_grad_q[0,5] = 2*(q_new[5]-q_new[1])
    variance = B_grad_q @ cov_mat @ np.transpose(B_grad_q)
    # print(variance)
    return np.sqrt(variance)

def std2(q_new,cov_mat): 
    B_grad_q = np.zeros((1,Q_dim))
    B_grad_q[0,0] = 2*(q_new[0]-q_new[8])
    B_grad_q[0,1] = 2*(q_new[1]-q_new[9])
    B_grad_q[0,8] = 2*(q_new[8]-q_new[0])
    B_grad_q[0,9] = 2*(q_new[9]-q_new[1])
    variance = B_grad_q @ cov_mat @ np.transpose(B_grad_q)
    # print(variance)
    return np.sqrt(variance)

def std3(q_new,cov_mat): #q:[3]
    B_grad_q = np.zeros((1,Q_dim))
    B_grad_q[0,4] = 2*(q_new[4]-q_new[8])
    B_grad_q[0,5] = 2*(q_new[5]-q_new[9])
    B_grad_q[0,8] = 2*(q_new[8]-q_new[4])
    B_grad_q[0,9] = 2*(q_new[9]-q_new[5])
    variance = B_grad_q @ cov_mat @ np.transpose(B_grad_q)
    # print(variance)
    return np.sqrt(variance)
    
def system(q0, steps, rnn, regressor_old_f, regressor_old_g, dt): # q0: [1, q_dim]
    # system dynamic
    q = q0
    Q = q0
    U = torch.empty((0,U_dim))
    h_state = None
    start = datetime.datetime.now()
    collision = False
    out_of_bound = False
    for i in range(steps):
        with torch.no_grad():
            u_ref, h_state_n = rnn(q.view(1,1,Q_dim),h_state) # u: [1,1, u_dim]
            h_state = h_state_n
            
            regressor_old_f.train()
            regressor_old_g.train()
        
            x_test = q.repeat(nb_samples,1) # [nb_samples, Q_dim]
            u_test = u_ref.view(1,U_dim).repeat(nb_samples,1) #[nb_samples, U_dim]
            y_test = regressor_old_f(x_test) + torch.bmm(regressor_old_g(x_test).view(-1,Q_dim,U_dim), u_test.view(-1,U_dim,1)).view(-1,Q_dim) # [nb_samples,Q_dim]
            y_test = y_test.numpy()
            cov_mat = np.cov(np.transpose(y_test))
            # print('Covariance:', cov_mat)
        
        
            u_ref = u_ref.view(-1).detach().numpy() #[U_dim]
            regressor_old_f.eval()
            regressor_old_g.eval()
            
        m = gp.Model('qcqp')
        # u0 = m.addVar(lb=0, ub = 0.5*dt, vtype=GRB.CONTINUOUS, name="u0")
        # u1 = m.addVar(lb=-1*dt, ub = 1*dt, vtype=GRB.CONTINUOUS, name="u1")
        # u2 = m.addVar(lb=0, ub = 0.5*dt, vtype=GRB.CONTINUOUS, name="u2")
        # u3 = m.addVar(lb=-1*dt, ub = 1*dt, vtype=GRB.CONTINUOUS, name="u3")
        # u4 = m.addVar(lb=0, ub = 0.5*dt, vtype=GRB.CONTINUOUS, name="u4")
        # u5 = m.addVar(lb=-1*dt, ub = 1*dt, vtype=GRB.CONTINUOUS, name="u5")
        
        # m.setObjective((u0-u_ref[0])*(u0-u_ref[0]) + gamma*(u1-u_ref[1])*(u1-u_ref[1]) \
        #              + (u2-u_ref[2])*(u2-u_ref[2]) + gamma*(u3-u_ref[3])*(u3-u_ref[3]) \
        #              + (u4-u_ref[4])*(u4-u_ref[4]) + gamma*(u5-u_ref[5])*(u5-u_ref[5]), GRB.MINIMIZE)
        
        u_mat = m.addMVar(shape=U_dim, lb=[0,-1.5*dt,0,-1.5*dt,0,-1.5*dt], ub=[0.7*dt,1.5*dt,0.7*dt,1.5*dt,0.7*dt,1.5*dt],vtype=GRB.CONTINUOUS, name="u_mat")
        Gamma = np.diag([1,gamma,1,gamma,1,gamma])
        m.setObjective(u_mat@Gamma@u_mat  - 2*u_ref@Gamma@u_mat + u_ref@Gamma@u_ref, GRB.MINIMIZE)    
        
        f = regressor_old_f(q).view(-1).detach().numpy() #[Q_dim]
        g = regressor_old_g(q).view(Q_dim,U_dim).detach().numpy() #[Q_dim*U_dim]
        q_est = f + g @ u_ref
        
        sgm1 = std1(q_est,cov_mat)
        sgm2 = std2(q_est,cov_mat)
        sgm3 = std3(q_est,cov_mat)
        
        m.addConstr((f[0]-f[4])**2 + 2*(f[0]-f[4])*(g[0]-g[4])@u_mat + u_mat@((g[0]-g[4]).reshape(U_dim,1)@(g[0]-g[4]).reshape(1,U_dim))@u_mat \
                   +(f[1]-f[5])**2 + 2*(f[1]-f[5])*(g[1]-g[5])@u_mat + u_mat@((g[1]-g[5]).reshape(U_dim,1)@(g[1]-g[5]).reshape(1,U_dim))@u_mat - D**2 +(alpha_c-1)*((q[0,0]-q[0,4])**2 + (q[0,1]-q[0,5])**2 - D**2) >= k*sgm1, name='c1')
        m.addConstr((f[0]-f[8])**2 + 2*(f[0]-f[8])*(g[0]-g[8])@u_mat + u_mat@((g[0]-g[8]).reshape(U_dim,1)@(g[0]-g[8]).reshape(1,U_dim))@u_mat \
                   +(f[1]-f[9])**2 + 2*(f[1]-f[9])*(g[1]-g[9])@u_mat + u_mat@((g[1]-g[9]).reshape(U_dim,1)@(g[1]-g[9]).reshape(1,U_dim))@u_mat - D**2 +(alpha_c-1)*((q[0,0]-q[0,8])**2 + (q[0,1]-q[0,9])**2 - D**2) >= k*sgm2, name='c2')
        m.addConstr((f[4]-f[8])**2 + 2*(f[4]-f[8])*(g[4]-g[8])@u_mat + u_mat@((g[4]-g[8]).reshape(U_dim,1)@(g[4]-g[8]).reshape(1,U_dim))@u_mat \
                   +(f[5]-f[9])**2 + 2*(f[5]-f[9])*(g[5]-g[9])@u_mat + u_mat@((g[5]-g[9]).reshape(U_dim,1)@(g[5]-g[9]).reshape(1,U_dim))@u_mat - D**2 +(alpha_c-1)*((q[0,4]-q[0,8])**2 + (q[0,5]-q[0,9])**2 - D**2) >= k*sgm3, name='c3')
        for j in range(nb_agent):
            m.addConstr(f[4*j+0]+g[4*j+0]@u_mat-0 + (alpha_b-1)*(q[0,4*j+0]-0) >= k*cov_mat[4*j+0,4*j+0], name='c%s'%(4*j+4)) 
            m.addConstr(10-f[4*j+0]-g[4*j+0]@u_mat + (alpha_b-1)*(10-q[0,4*j+0]) >= k*cov_mat[4*j+0,4*j+0], name='c%s'%(4*j+5)) 
            m.addConstr(f[4*j+1]+g[4*j+1]@u_mat-1 + (alpha_b-1)*(q[0,4*j+1]-1) >= k*cov_mat[4*j+1,4*j+1], name='c%s'%(4*j+6)) 
            m.addConstr(9-f[4*j+1]-g[4*j+1]@u_mat + (alpha_b-1)*(9-q[0,4*j+1]) >= k*cov_mat[4*j+1,4*j+1], name='c%s'%(4*j+7))         
        m.setParam("NonConvex", 2, verbose = False)
        m.setParam('LogToConsole', 0)
        m.optimize()
        # print(m.status)
        if m.status == 2:
            u = torch.from_numpy(u_mat.X).view(1,-1)
        elif m.status == 3 or m.status == 4:
            u = torch.from_numpy(u_ref).view(1,-1)
        else:
            print(m.status)
            raise ValueError('A very specific bad thing happened.')
        q_new = torch.zeros((1,Q_dim))
        noise = torch.zeros((1,Q_dim))
        noise[:,[0,1,4,5,8,9]] = torch.normal(mean=0,std=0.05*torch.ones((1,nb_agent*2)))
        for j in range(nb_agent):
            q_new[0,4*j+0] = q[0,4*j+0] + u[0,2*j+0]/u[0,2*j+1] * (q[0,4*j+2]*torch.cos(u[0,2*j+1]) + q[0,4*j+3]*torch.sin(u[0,2*j+1]) - q[0,4*j+2])
            q_new[0,4*j+1] = q[0,4*j+1] + u[0,2*j+0]/u[0,2*j+1] * (q[0,4*j+3] - q[0,4*j+3]*torch.cos(u[0,2*j+1]) + q[0,4*j+2]*torch.sin(u[0,2*j+1]))
            q_new[0,4*j+2] = q[0,4*j+2]*torch.cos(u[0,2*j+1]) + q[0,4*j+3]*torch.sin(u[0,2*j+1])
            q_new[0,4*j+3] = q[0,4*j+3]*torch.cos(u[0,2*j+1]) - q[0,4*j+2]*torch.sin(u[0,2*j+1])
        q = q_new
        d_12 = torch.sqrt((q[:,0] - q[:,4])**2 + (q[:,1] - q[:,5])**2).view(-1)
        d_13 = torch.sqrt((q[:,0] - q[:,8])**2 + (q[:,1] - q[:,9])**2).view(-1)
        d_23 = torch.sqrt((q[:,4] - q[:,8])**2 + (q[:,5] - q[:,9])**2).view(-1)
        if (d_12<D).float()+(d_13<D).float()+(d_23<D).float() > 0:
            collision = True

        for j in range(nb_agent):
            if q[0,4*j+0]<0 or q[0,4*j+0]>10 or q[0,4*j+1]<1 or q[0,4*j+1]>9:
                out_of_bound = True
        Q = torch.cat((Q,q+noise),dim=0) # Q: [time_step+1, q_dim]
        U = torch.cat((U,u.float()),dim=0) # U: [time_step, u_dim]
        if collision or out_of_bound:
            break
    end = datetime.datetime.now()
    print(end-start)
    return Q,U
    
def train_system_model(X, Y, rnn, system_model_f, system_model_g, batch_size, steps, BATCH_SIZE, MAX_ITER, iteration,dt,Di):

    T = 15 / dt
    regressor_f = Regressor_f(Q_dim,Q_dim)
    regressor_f.train()
    regressor_g = Regressor_g(Q_dim,Q_dim*U_dim)
    regressor_g.train()
    
    regressor_old_f = Regressor_f(Q_dim,Q_dim)
    regressor_old_f.load_state_dict(system_model_f.state_dict())
    regressor_old_g = Regressor_g(Q_dim,Q_dim*U_dim)
    regressor_old_g.load_state_dict(system_model_g.state_dict())
    
    # with torch.no_grad():1
        # q0_test = 10*torch.rand(10,6)
        # u_test = 0.6 * (torch.rand(10,6)-0.5)
        # x_test = torch.cat((q0_test,u_test),dim=1) # [nb_points, 5]
        # y_test = torch.zeros((20,10,6)) # [nb_samples,nb_points,3]
    
        # for i in range(20):
        #     y_test[i,:,:] = regressor_old(x_test)
        # y_test = y_test.numpy()
        # cov_mat = np.zeros((Q_dim,Q_dim))
        # for i in range(10):
        #     cov_mat = cov_mat + np.cov(np.transpose(y_test[:,i,:]))
        # cov_mat = cov_mat/10. #[3*3]
        # # y_var = torch.var(y_test,dim=0) #[nb_points,3]
        # # y_var_mean = torch.mean(y_var,dim=0) #[3]

        # print('Covariance:', cov_mat)
        
            
    Q0 = torch.zeros((batch_size,Q_dim))
    batch=0
    while batch<batch_size:
        q0 = torch.rand((1,Q_dim))
        theta = (torch.rand((1,nb_agent))-0.5)*math.pi/2
        theta[:,2] += math.pi
        for i in range(nb_agent-1):    
            q0[:,4*i]   = 1.5*q0[:,4*i]+0.5
            q0[:,4*i+1] = q0[:,4*i+1]*4+3
            q0[:,4*i+2] = torch.sin(theta[:,i])
            q0[:,4*i+3] = torch.cos(theta[:,i]) 
        q0[:,8] = 1.5*q0[:,8] + 8
        q0[:,9] = 2 * q0[:,9] + 4
        q0[:,10] = torch.sin(theta[:,2])
        q0[:,11] = torch.cos(theta[:,2])    
        d0_12 = torch.sqrt((q0[:,0] - q0[:,4])**2 + (q0[:,1] - q0[:,5])**2).view(-1)
        d0_13 = torch.sqrt((q0[:,0] - q0[:,8])**2 + (q0[:,1] - q0[:,9])**2).view(-1)
        d0_23 = torch.sqrt((q0[:,4] - q0[:,8])**2 + (q0[:,5] - q0[:,9])**2).view(-1)       
        if (d0_12>Di).float()+(d0_13>Di).float()+(d0_23>Di).float()==3:
            Q0[batch,:] = q0[0,:]
            batch=batch+1

    for i in range(batch_size):
        Q, U = system(Q0[i,:].view(1,Q_dim),int(T),rnn,regressor_old_f,regressor_old_g,dt)

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
    
    # X = torch.cat([Q[:,:20,:].reshape(-1,3),U.reshape(-1,2)],dim=1)
    # Y = Delta.reshape(-1,3)
    
    # X_t = torch.cat([Q_t[:,:20,:].reshape(-1,3),U_t.reshape(-1,2)],dim=1)
    # Y_t = Delta_t.reshape(-1,3)
    
    Dataset = Data.TensorDataset(X, Y)
    loader = Data.DataLoader(
        dataset=Dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               
        num_workers=0,              
    )
    
    # class Regressor(nn.Module):
    #     def __init__(self, input_dim, output_dim):
    #         super().__init__()
    #         #self.linear = nn.Linear(input_dim, output_dim)
    #         self.linear1 = nn.Linear(input_dim, 32)
    #         self.linear2 = nn.Linear(32, 32)
    #         self.linear3 = nn.Linear(32, output_dim)
    #         # self.dropout = nn.Dropout(0.1)
            
    #     def forward(self, x):
    #         x = F.relu(self.linear1(x))
    #         x = F.relu(self.linear2(x))
    #         return self.linear3(x)
        
    # regressor = Regressor(5, 3)
    
    optimizer = torch.optim.Adam(list(regressor_f.parameters()) + list(regressor_g.parameters()), lr=0.001)
    criterion = nn.MSELoss()
    best_loss = 10
    
    for epoch in range(200):
        for i, (datapoints, labels) in enumerate(loader):
            # datapoints = datapoints.cuda()
            # labels = labels.cuda()
            optimizer.zero_grad()
            loss = criterion(regressor_f(datapoints[:,:Q_dim]) + torch.bmm(regressor_g(datapoints[:,:Q_dim]).view(-1,Q_dim, U_dim),datapoints[:,Q_dim:].view(-1,U_dim,1)).view(-1,Q_dim), labels)
            if loss < best_loss:
                torch.save(regressor_f.state_dict(), 'system_model_f'+str(iteration)+'.pkl')
                torch.save(regressor_g.state_dict(), 'system_model_g'+str(iteration)+'.pkl')
                best_loss = loss
            loss.backward()
            optimizer.step()
            
            # regressor.eval()
            # loss_t = criterion(regressor(X_t),Y_t)
            # regressor.train()
            if i % 20 == 0: 
                print('Epoch: ', epoch, '| train loss: %.5f' % loss.data)
    
    return X, Y
    
