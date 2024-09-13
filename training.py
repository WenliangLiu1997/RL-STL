#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:36:45 2020

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
from scipy import optimize
import datetime
from train_system_model import train_system_model
import torch.nn.functional as F
from rnn_test_function_cbf import rnn_test_function
from pre_train import pre_train
import copy
import pickle

# torch.autograd.set_detect_anomaly(True)
# device = 'cuda:0'
device = 'cpu'
dev = torch.device(device)
scale_w = 1.
Rx = 10.
Ry = 10.
dt = 1/3
t1 = 8. / dt
t2 = 10. / dt
T = 15. / dt

nb_agent = 3
Q_dim = 4*nb_agent     
U_dim = 2*nb_agent   
HIDDEN_SIZE = 64
NUM_LAYERS = 2
batch_size_guide = 512
batch_size = 700
EPOCH_guide = 400
EPOCH = 400
learning_rate_guide = 0.005
learning_rate = 0.005
beta1 = 0.9
beta2 = 0.999
epsilon =1e-8
alpha = 1
Iter = 3
dropout = 0.05
clip_value = 10

Di = 0.7

XB1=torch.tensor(10., dtype=torch.float, requires_grad=False)
XB2=torch.tensor(0., dtype=torch.float, requires_grad=False)
YB1=torch.tensor(9., dtype=torch.float, requires_grad=False)
YB2=torch.tensor(1., dtype=torch.float, requires_grad=False)

XF=torch.tensor(5, dtype=torch.float, requires_grad=False)
YF=torch.tensor(5, dtype=torch.float, requires_grad=False)
RF2=torch.tensor(0.94, dtype=torch.float, requires_grad=False)

XHyd=torch.tensor(7, dtype=torch.float, requires_grad=False)
YHyd=torch.tensor(7, dtype=torch.float, requires_grad=False)
RHyd2=torch.tensor(0.36, dtype=torch.float, requires_grad=False)


XH1=torch.tensor(2, dtype=torch.float, requires_grad=False)
XH2=torch.tensor(0.5, dtype=torch.float, requires_grad=False)
YH1=torch.tensor(7, dtype=torch.float, requires_grad=False)
YH2=torch.tensor(3, dtype=torch.float, requires_grad=False)

D=torch.tensor(0.3, dtype=torch.float, requires_grad=False)

XB1_exp = stlcg.Expression(XB1)
XB2_exp = stlcg.Expression(XB2)
YB1_exp = stlcg.Expression(YB1)
YB2_exp = stlcg.Expression(YB2)
XH1_exp = stlcg.Expression(XH1)
XH2_exp = stlcg.Expression(XH2)
YH1_exp = stlcg.Expression(YH1)
YH2_exp = stlcg.Expression(YH2)
RF2_exp = stlcg.Expression(RF2)
RHyd2_exp = stlcg.Expression(RHyd2)

D_exp = stlcg.Expression(D)
# torch.manual_seed(0)

def tansig(x,u_min,u_max):
    return u_min + (u_max-u_min)*(torch.tanh(x)+1)/2
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=Q_dim,
            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
            num_layers=NUM_LAYERS,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
#            dropout = 0.5,
#            nonlinearity = 'relu'
        )
        
        self.out = nn.Linear(HIDDEN_SIZE, U_dim)

    def forward(self, x, h_state):    
        # x (batch, time_step, input_size)
        # r_out (batch, time_step, hidden_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(tansig(self.out(r_out[:, time_step, :]),torch.tensor([0,-1.5*dt,0,-1.5*dt,0,-1.5*dt]).to(dev),torch.tensor([0.7*dt,1.5*dt,0.7*dt,1.5*dt,0.7*dt,1.5*dt]).to(dev)))
        return torch.stack(outs, dim=1), h_state
    
rnn = RNN()
rnn.to(dev)
# for p in rnn.parameters():
#     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

class Regressor_control_g(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.linear = nn.Linear(input_dim, output_dim)
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, output_dim)
        # self.mask1 = torch.bernoulli(torch.ones([batch_size, 64])*0.9)
        # self.mask2 = torch.bernoulli(torch.ones([batch_size, 64])*0.9)
        
    def forward(self, x, mask):
        # x = F.relu(self.linear1(self.dropout(x)))
        # x = F.relu(self.linear2(self.dropout(x)))
        # if len(x.size()) == 3:
        #     mask1 = self.mask1.unsqueeze(dim=1).repeat(1,x.size(1),1)
        #     mask2 = self.mask2.unsqueeze(dim=1).repeat(1,x.size(1),1)
        # else:
        #     mask1 = self.mask1
        #     mask2 = self.mask2
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x * mask[0] ))
        return self.linear3(x * mask[1] )

class Regressor_control_f(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.linear = nn.Linear(input_dim, output_dim)
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, output_dim)
        # self.mask1 = torch.bernoulli(torch.ones([batch_size, 64])*0.9)
        # self.mask2 = torch.bernoulli(torch.ones([batch_size, 64])*0.9)
        
    def forward(self, x, mask):
        # x = F.relu(self.linear1(self.dropout(x)))
        # x = F.relu(self.linear2(self.dropout(x)))
        # if len(x.size()) == 3:
        #     mask1 = self.mask1.unsqueeze(dim=1).repeat(1,x.size(1),1)
        #     mask2 = self.mask2.unsqueeze(dim=1).repeat(1,x.size(1),1)
        # else:
        #     mask1 = self.mask1
        #     mask2 = self.mask2
        y = F.relu(self.linear1(x))
        y = F.relu(self.linear2(y * mask[0] ))
        return self.linear3(y * mask[1] ) + x


def system_nn(q0, steps): # q0: [batch_size, q_dim]
    # system dynamic (learned)
    q = torch.zeros((q0.size(0),steps+1,Q_dim)).to(dev) # [batch_size, time_steps+1, q_dim]
    u = torch.zeros((q0.size(0),steps,U_dim)).to(dev) # [batch_size, time_steps, u_dim]
    q[:,0,:] = q0
    h_state = None
    for i in range(steps):
        u_i, h_state_n = rnn(q[:,i,:].view(-1,1,Q_dim).clone(),h_state) # u_i: [batch_size, time_step = 1, u_dim]
        u[:,i,:] = u_i[:,0,:] 
        h_state = h_state_n

        # Delta = system_model(torch.cat((q[:,i,:],u[:,i,:]),dim=1),mask)
        f = system_model_f(q[:,i,:].clone(),mask_f)
        g = system_model_g(q[:,i,:].clone(),mask_g).view(-1,Q_dim,U_dim)
        q[:,i+1,:] = f + torch.bmm(g, u[:,i,:].clone().view(-1,U_dim,1)).view(-1,Q_dim)
        for j in range(nb_agent):
            q[:,i+1,[4*j+2,4*j+3]] = F.normalize(q[:,i+1,[4*j+2,4*j+3]],dim=1)
    return q, u


X = torch.from_numpy(np.load('Model_dataset_X.npy')).float()
Y = torch.from_numpy(np.load('Model_dataset_Y.npy')).float()


suc_rate = np.zeros(Iter)
Ro_mean = np.zeros(Iter)
nb_collision = np.zeros(Iter)
nb_out_of_bound = np.zeros(Iter)
start = datetime.datetime.now()


L_all = np.empty(0)
nb_iter = np.zeros(Iter)

nbParas=0
m_old=[]
v_old=[]
m_new=[]
v_new=[]
for p in rnn.parameters():
    nbParas += 1
    m_old.append(torch.zeros(p.data.size()))
    v_old.append(torch.zeros(p.data.size()))
    m_new.append(torch.zeros(p.data.size()))
    v_new.append(torch.zeros(p.data.size()))    
t = 0


q0 = torch.rand((batch_size,Q_dim))
theta = (torch.rand((batch_size,nb_agent))-0.5)*math.pi/2
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
eff_idx = torch.nonzero((d0_12>Di).float()+(d0_13>Di).float()+(d0_23>Di).float()==3,as_tuple=True)
batch_size_eff = len(eff_idx[0])
print('batch_size_eff: ',batch_size_eff)
q0 = q0[eff_idx]



maskf1 = torch.bernoulli(torch.ones([batch_size_eff, 64])*(1-dropout)) / (1-dropout)
maskf2 = torch.bernoulli(torch.ones([batch_size_eff, 64])*(1-dropout)) / (1-dropout)
mask_f = [maskf1, maskf2]
maskg1 = torch.bernoulli(torch.ones([batch_size_eff, 64])*(1-dropout)) / (1-dropout)
maskg2 = torch.bernoulli(torch.ones([batch_size_eff, 64])*(1-dropout)) / (1-dropout)
mask_g = [maskg1, maskg2]

training_time = datetime.datetime.now() - datetime.datetime.now()

for iteration in range(Iter):


    
    if iteration == 0:
        system_model_f = Regressor_control_f(Q_dim, Q_dim)
        system_model_f.load_state_dict(torch.load('system_model_f0.pkl'))
        system_model_f.to(dev)
        system_model_g = Regressor_control_g(Q_dim, Q_dim*U_dim)
        system_model_g.load_state_dict(torch.load('system_model_g0.pkl'))
        system_model_g.to(dev)
        

    else:
        start_model_train = datetime.datetime.now()
        X, Y = train_system_model(X, Y, rnn, system_model_f, system_model_g, 40, int(T), 32, 200,iteration,dt,Di)
        system_model_f = Regressor_control_f(Q_dim, Q_dim)
        system_model_f.load_state_dict(torch.load('system_model_f'+str(iteration)+'.pkl'))
        system_model_f.to(dev)
        system_model_g = Regressor_control_g(Q_dim, Q_dim*U_dim)
        system_model_g.load_state_dict(torch.load('system_model_g'+str(iteration)+'.pkl'))
        system_model_g.to(dev)
        end_model_train = datetime.datetime.now()
        training_time += end_model_train - start_model_train



    with torch.no_grad():
        q = system_nn(q0,int(T))[0]
    
    df1_exp = stlcg.Expression('df1',torch.sqrt((((q[:,:,0]-XF)**2).flip(1)+((q[:,:,1]-YF)**2).flip(1)).view(q0.size(0),-1,1))) # [batch_size, time_steps+1, 1]
    df2_exp = stlcg.Expression('df2',torch.sqrt((((q[:,:,4]-XF)**2).flip(1)+((q[:,:,5]-YF)**2).flip(1)).view(q0.size(0),-1,1))) # [batch_size, time_steps+1, 1]
    df3_exp = stlcg.Expression('df3',torch.sqrt((((q[:,:,8]-XF)**2).flip(1)+((q[:,:,9]-YF)**2).flip(1)).view(q0.size(0),-1,1))) # [batch_size, time_steps+1, 1]

    dhyd3_exp = stlcg.Expression('dhyd3',torch.sqrt((((q[:,:,8]-XHyd)**2).flip(1)+((q[:,:,9]-YHyd)**2).flip(1)).view(q0.size(0),-1,1))) # [batch_size, time_steps+1, 1]

    d12_exp = stlcg.Expression('d12',torch.sqrt((((q[:,:,0]-q[:,:,4])**2).flip(1)+((q[:,:,1]-q[:,:,5])**2).flip(1)).view(q0.size(0),-1,1))) # [batch_size, time_steps+1, 1]
    d13_exp = stlcg.Expression('d13',torch.sqrt((((q[:,:,0]-q[:,:,8])**2).flip(1)+((q[:,:,1]-q[:,:,9])**2).flip(1)).view(q0.size(0),-1,1))) # [batch_size, time_steps+1, 1]
    d23_exp = stlcg.Expression('d23',torch.sqrt((((q[:,:,4]-q[:,:,8])**2).flip(1)+((q[:,:,5]-q[:,:,9])**2).flip(1)).view(q0.size(0),-1,1))) # [batch_size, time_steps+1, 1]
    x1_exp = stlcg.Expression('x1',q[:,:,0].flip(1).view(q0.size(0),-1,1))
    y1_exp = stlcg.Expression('y1',q[:,:,1].flip(1).view(q0.size(0),-1,1))
    x2_exp = stlcg.Expression('x2',q[:,:,4].flip(1).view(q0.size(0),-1,1))
    y2_exp = stlcg.Expression('y2',q[:,:,5].flip(1).view(q0.size(0),-1,1))
    x3_exp = stlcg.Expression('x3',q[:,:,8].flip(1).view(q0.size(0),-1,1))
    y3_exp = stlcg.Expression('y3',q[:,:,9].flip(1).view(q0.size(0),-1,1))
    
    # STL formula
    phiF1 = df1_exp <= RF2_exp
    phiF2 = df2_exp <= RF2_exp
    phiF3 = df3_exp <= RF2_exp
    
    phiHyd3 = dhyd3_exp <= RHyd2_exp


    phiD12 = d12_exp >= D_exp
    phiD13 = d13_exp >= D_exp
    phiD23 = d23_exp >= D_exp
    
    phiB11 = x1_exp <= XB1_exp
    phiB12 = x1_exp >= XB2_exp
    phiB13 = y1_exp <= YB1_exp
    phiB14 = y1_exp >= YB2_exp
    
    phiB21 = x2_exp <= XB1_exp
    phiB22 = x2_exp >= XB2_exp
    phiB23 = y2_exp <= YB1_exp
    phiB24 = y2_exp >= YB2_exp
    
    phiB31 = x3_exp <= XB1_exp
    phiB32 = x3_exp >= XB2_exp
    phiB33 = y3_exp <= YB1_exp
    phiB34 = y3_exp >= YB2_exp   
    
    phiH11 = x1_exp <= XH1_exp
    phiH12 = x1_exp >= XH2_exp
    phiH13 = y1_exp <= YH1_exp
    phiH14 = y1_exp >= YH2_exp    
    
    phiH21 = x2_exp <= XH1_exp
    phiH22 = x2_exp >= XH2_exp
    phiH23 = y2_exp <= YH1_exp
    phiH24 = y2_exp >= YH2_exp  
    
       

    psiF1 = stlcg.Eventually(subformula=phiF1,interval=[0,int(t1)])
    psiF2 = stlcg.Eventually(subformula=phiF2,interval=[0,int(t1)])
    psiF3 = stlcg.Always(subformula=phiF3,interval=[int(t2),int(T)])
    psiHyd3 = stlcg.Eventually(subformula=phiHyd3,interval=[0,int(t2)])
    
    phiH1 = stlcg.And(phiH11,phiH12,phiH13,phiH14)
    psiH1 = stlcg.Eventually(subformula=phiH1,interval=[int(t1),int(T)])
    
    phiH2 = stlcg.And(phiH21,phiH22,phiH23,phiH24)
    psiH2 = stlcg.Eventually(subformula=phiH2,interval=[int(t1),int(T)])
    
    phiD = stlcg.And(phiD12,phiD13,phiD23)
    psiD = stlcg.Always(subformula=phiD,interval=[0,int(T)])
    
    phiB = stlcg.And(phiB11,phiB12,phiB13,phiB14,phiB21,phiB22,phiB23,phiB24,phiB31,phiB32,phiB33,phiB34)
    psiB = stlcg.Always(subformula=phiB,interval=[0,int(T)])
    
    
    formula1 = stlcg.And(psiF1,psiF2,psiF3,psiHyd3,psiH1,psiH2,psiD,psiB)
    # formula1 = psiO
    print(formula1)
    

    L_mean_guide = np.zeros(EPOCH_guide)
    L_mean = np.zeros(EPOCH)


    
    
    
    if iteration<0:
        pass

    else:
        # TRAINING

        # nbParas=0
        # m_old=[]
        # v_old=[]
        # m_new=[]
        # v_new=[]
        # for p in rnn.parameters():
        #     nbParas += 1
        #     m_old.append(torch.zeros(p.data.size()))
        #     v_old.append(torch.zeros(p.data.size()))
        #     m_new.append(torch.zeros(p.data.size()))
        #     v_new.append(torch.zeros(p.data.size()))    
        # t = 0
            
        ro_mean_best = -10
    
        epoch_no_inc = 0
        epoch_after_suc = 0
        
        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
        for epoch in range(EPOCH):
            # optimizer.zero_grad()
            start_epoch = datetime.datetime.now()
            
            q0 = torch.rand((batch_size,Q_dim))
            theta = (torch.rand((batch_size,nb_agent))-0.5)*math.pi/2
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
            eff_idx = torch.nonzero((d0_12>Di).float()+(d0_13>Di).float()+(d0_23>Di).float()==3,as_tuple=True)
            batch_size_eff = len(eff_idx[0])
            print('batch_size_eff: ',batch_size_eff)
            q0 = q0[eff_idx]
            
            # mask0 = torch.bernoulli(torch.ones([batch_size_eff, Q_dim+U_dim])*(1-dropout)) / (1-dropout)
            maskf1 = torch.bernoulli(torch.ones([batch_size_eff, 64])*(1-dropout)) / (1-dropout)
            maskf2 = torch.bernoulli(torch.ones([batch_size_eff, 64])*(1-dropout)) / (1-dropout)
            mask_f = [maskf1, maskf2]
            maskg1 = torch.bernoulli(torch.ones([batch_size_eff, 64])*(1-dropout)) / (1-dropout)
            maskg2 = torch.bernoulli(torch.ones([batch_size_eff, 64])*(1-dropout)) / (1-dropout)
            mask_g = [maskg1, maskg2]
            
            print('Trail ',iteration, ', epoch ', epoch)
            
            epoch_no_inc = epoch_no_inc + 1

            q_wGrad, u = system_nn(q0,int(T))
            q_wGrad.retain_grad()
            u.retain_grad()

                    
            df1_exp = stlcg.Expression('df1',torch.sqrt((((q_wGrad[:,:,0]-XF)**2).flip(1)+((q_wGrad[:,:,1]-YF)**2).flip(1)).view(q0.size(0),-1,1))) # [batch_size, time_steps+1, 1]
            df2_exp = stlcg.Expression('df2',torch.sqrt((((q_wGrad[:,:,4]-XF)**2).flip(1)+((q_wGrad[:,:,5]-YF)**2).flip(1)).view(q0.size(0),-1,1))) # [batch_size, time_steps+1, 1]
            df3_exp = stlcg.Expression('df3',torch.sqrt((((q_wGrad[:,:,8]-XF)**2).flip(1)+((q_wGrad[:,:,9]-YF)**2).flip(1)).view(q0.size(0),-1,1))) # [batch_size, time_steps+1, 1]
            dhyd3_exp = stlcg.Expression('do3',torch.sqrt((((q_wGrad[:,:,8]-XHyd)**2).flip(1)+((q_wGrad[:,:,9]-YHyd)**2).flip(1)).view(q0.size(0),-1,1))) # [batch_size, time_steps+1, 1]
            d12_exp = stlcg.Expression('d12',torch.sqrt((((q_wGrad[:,:,0]-q_wGrad[:,:,4])**2).flip(1)+((q_wGrad[:,:,1]-q_wGrad[:,:,5])**2).flip(1)).view(q0.size(0),-1,1))) # [batch_size, time_steps+1, 1]
            d13_exp = stlcg.Expression('d13',torch.sqrt((((q_wGrad[:,:,0]-q_wGrad[:,:,8])**2).flip(1)+((q_wGrad[:,:,1]-q_wGrad[:,:,9])**2).flip(1)).view(q0.size(0),-1,1))) # [batch_size, time_steps+1, 1]
            d23_exp = stlcg.Expression('d23',torch.sqrt((((q_wGrad[:,:,4]-q_wGrad[:,:,8])**2).flip(1)+((q_wGrad[:,:,5]-q_wGrad[:,:,9])**2).flip(1)).view(q0.size(0),-1,1))) # [batch_size, time_steps+1, 1]
            x1_exp = stlcg.Expression('x1',q_wGrad[:,:,0].flip(1).view(q0.size(0),-1,1))
            y1_exp = stlcg.Expression('y1',q_wGrad[:,:,1].flip(1).view(q0.size(0),-1,1))
            x2_exp = stlcg.Expression('x2',q_wGrad[:,:,4].flip(1).view(q0.size(0),-1,1))
            y2_exp = stlcg.Expression('y2',q_wGrad[:,:,5].flip(1).view(q0.size(0),-1,1))
            x3_exp = stlcg.Expression('x3',q_wGrad[:,:,8].flip(1).view(q0.size(0),-1,1))
            y3_exp = stlcg.Expression('y3',q_wGrad[:,:,9].flip(1).view(q0.size(0),-1,1))
            inputs1 = (df1_exp,df2_exp,df3_exp,dhyd3_exp,(x1_exp,x1_exp,y1_exp,y1_exp),(x2_exp,x2_exp,y2_exp,y2_exp),(d12_exp,d13_exp,d23_exp),(x1_exp,x1_exp,y1_exp,y1_exp,x2_exp,x2_exp,y2_exp,y2_exp,x3_exp,x3_exp,y3_exp,y3_exp))#,(do1_exp,do2_exp,do3_exp))

            ro = formula1.robustness(inputs1, pscale = 1, scale=3)    

            
            ro_sum = ro.sum()
            ro_mean = ro_sum/batch_size_eff
            ro_min = ro.min()
            if ro_mean>ro_mean_best and epoch!=0:
                ro_mean_best = ro_mean
                torch.save(rnn, 'rnn_new'+str(iteration)+'.pkl')
                
                epoch_no_inc = 0
            if ro_min<0:
                epoch_after_suc = 0
            epoch_after_suc = epoch_after_suc + 1
            
            print(ro_mean)


            L_mean[epoch] = ro_mean
            
            cost = - ro_mean
            cost.backward()
            
            # t = t + 1
            # j=0
            # for p in rnn.parameters():
            #     dW = p.grad
            #     if torch.any(torch.isnan(dW)):
            #         print('Error!')
            #     # dW = dW*batch_size/batch_size_eff
            #     m_new[j] = beta1 * m_old[j] + (1-beta1) * dW
            #     v_new[j] = beta2 * v_old[j] + (1-beta2) * dW**2
            #     m_old[j] = m_new[j].clone()
            #     v_old[j] = v_new[j].clone()
            #     p.data += learning_rate * math.sqrt(1-beta2**t) / (1-beta1**t) * m_new[j] / (torch.sqrt(v_new[j]) + epsilon)
            #     j = j+1
            optimizer.step()
            rnn.zero_grad()
            
            if epoch_no_inc > 40 or epoch_after_suc > 40:
                break
            end_epoch = datetime.datetime.now()
            print(end_epoch-start_epoch)
            training_time += end_epoch-start_epoch
    
        L_all = np.concatenate((L_all,L_mean[:epoch+1]))
        nb_iter[iteration] = epoch+1
            
    
    
        plt.plot(np.arange(EPOCH),L_mean)
        plt.grid()
        plt.show()
        end2 = datetime.datetime.now()
        rnn = torch.load('rnn_new'+str(iteration)+'.pkl')

        print('testing...')
        Ro_mean[iteration], suc_rate[iteration], nb_collision[iteration], nb_out_of_bound[iteration] = rnn_test_function(iteration,T,formula1,dt,Di)
        print(end2-start)
        print('Training time: ', training_time)

        


plt.plot(np.arange(L_all.size),L_all)
for i in range(Iter-1):
    plt.axvline(nb_iter[:i+1].sum(),c='k',ls='--',zorder=0)
plt.grid()
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Average robustness', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('robust.png',dpi=300,pad_inches=0.01,bbox_inches='tight')
plt.show()

plt.plot(np.arange(Iter),Ro_mean)
plt.grid()
plt.xlabel('Cycles', fontsize=14)
plt.ylabel('Mean testing robustness', fontsize=14)
plt.show()
plt.plot(np.arange(Iter),suc_rate)
plt.grid()
plt.xlabel('Cycles', fontsize=14)
plt.ylabel('Success rate', fontsize=14)
plt.show()

    
        
