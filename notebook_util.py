import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getHeatMap(mine_net,data,grids=[50,50]):
    Xmin = min(data[:,0])
    Xmax = max(data[:,0])
    Ymin = min(data[:,1])
    Ymax = max(data[:,1])
    x = np.linspace(Xmin, Xmax, grids[0])
    y = np.linspace(Ymin, Ymax, grids[1])
    xs, ys = np.meshgrid(x,y)
    XY = np.array((xs,ys))
    Z=np.array([mine_net(torch.FloatTensor(XY[:,i,j]).to(device)).item() 
                for j in range(ys.shape[0]) 
                for i in range(xs.shape[1])]).reshape(xs.shape[1],ys.shape[0])
    plt.pcolormesh(xs,ys,Z,cmap='RdBu_r')
    
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

import torch.nn as nn
import torch.nn.functional as F

class MineNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()
        sigma = .02
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight,std=sigma)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=sigma)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=sigma)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

from torch import optim

def resample(data,batch_size,replace=False):
    index = np.random.choice(range(data.shape[0]), size=batch_size, replace=replace)
    batch = data[index]
    return batch

#def resample_marginal(data,batch_size,X_ind,Y_ind,replace=False):
#    X_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=replace)
#    marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
#    batch = np.concatenate([data[joint_index][:,resp].reshape(-1,1), data[marginal_index][:,cond].reshape(-1,len(cond))], axis=1)
    

def sample_batch(data, x_index=0, y_index=[1], batch_size=100, sample_mode='marginal'):
    """[summary]
    
    Arguments:
        data {[type]} -- [N X 2]
        resp {[int]} -- [description]
        cond {[list]} -- [1 dimension]
    
    Keyword Arguments:
        batch_size {int} -- [description] (default: {100})
        randomJointIdx {bool} -- [description] (default: {True})
    
    Returns:
        [batch_joint] -- [batch size X 2]
        [batch_mar] -- [batch size X 2]
    """
    if type(cond)==list:
        whole = cond.copy()
        whole.append(resp)
    else:
        raise TypeError("cond should be list")
    if sample_mode == 'joint':
        index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[index]
        batch = batch[:, whole]
    elif sample_mode == 'unif':
        dataMax = data.max(axis=0)[whole]
        dataMin = data.min(axis=0)[whole]
        batch = (dataMax - dataMin)*np.random.random((batch_size,len(cond)+1)) + dataMin
    elif sample_mode == 'marginal':
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = np.concatenate([data[joint_index][:,resp].reshape(-1,1), data[marginal_index][:,cond].reshape(-1,len(cond))], axis=1)
    else:
        raise ValueError('Sample mode: {} not recognized.'.format(sample_mode))
    return batch



import os
import dill

def mi_max_envelope_estimate(mi_estimate,stopping_t, w1=50,w2=50):
    mi_smooth = np.array([max(mi_estimate[:i+1][-w1:]) for i in range(mi_estimate.size)])
    plt.plot(mi_estimate[:stopping_t+1],color='yellow')
    plt.plot(mi_smooth[:stopping_t+1],color='green')
    mi = mi_smooth[:stopping_t+1][-w2:].mean()
    plt.axhline(mi)
    return mi