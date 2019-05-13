import numpy as np
import torch

# Use GPU when available
# Need to use Tensor to create the tensor of default type
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

def plot_net(net, Xmin=-5, Xmax=5, Ymin=-5, Ymax=5, Xgrids=100, Ygrids=100):
    """Plot a heat map of a neural network net. net can only have two inputs.
    """
    x,y = np.mgrid[Xmin:Xmax:Xgrids*1j,Ymin:Ymax:Ygrids*1j]
    xy = np.concatenate((x[:,:,None],y[:,:,None]),axis=2)
    z = net(torch.Tensor(xy))[:,:,0].detach().cpu()
    plt.pcolormesh(x, y, z, cmap='RdBu_r')
    
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """The class defining the basic neural network architecture.
    """
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

import os
import dill

def mi_max_envelope_estimate(mi_lb_list, t, w1=50, w2=50):
    """Max envelop estimate"""
    mi_smooth = np.array([max(mi_lb_list[:i+1][-w1:]) for i in range(mi_lb_list.size)])
    plt.plot(mi_lb_list[:t+1],color='yellow')
    plt.plot(mi_smooth[:t+1],color='green')
    mi = mi_smooth[:t+1][-w2:].mean()
    plt.axhline(mi)
    return mi