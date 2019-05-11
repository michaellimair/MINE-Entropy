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

sigma = .02
class MineNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()
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

from minee.model.mine import sample_batch

import os
import dill

