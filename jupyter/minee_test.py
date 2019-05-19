from data.gaussian import Gaussian
from model.minee import MINEE
import os
from IPython import display
from ipywidgets import interact
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import numpy as np

name = 'G0.9_ssbs400_d2_MINEE_test'  # filename to load/save the results
chkpt_name = name+'.pt'
fig_name = name+'.pdf'

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

np.random.seed(0)
torch.manual_seed(0)

sample_size = 400
rho = 0.9

rep = 1  # number of repeated runs
d = 4
X = np.zeros((rep, sample_size, d))
Y = np.zeros((rep, sample_size, d))
for i in range(rep):
    for j in range(d):
        data = Gaussian(sample_size=sample_size, rho=rho).data
        X[i, :, j] = data[:, 0]
        Y[i, :, j] = data[:, 1]

plt.scatter(X[0, :, 0], Y[0, :, 0], label="data",
            marker="+", color="steelblue")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot of data samples')
plt.show()

batch_size = int(sample_size*1)
lr = 1e-4

minee_list = []
for i in range(rep):
    minee_list.append(MINEE(torch.Tensor(X[i]), torch.Tensor(
        Y[i]), batch_size=batch_size, lr=lr))
dXY_list = np.zeros((rep, 0))
dX_list = np.zeros((rep, 0))
dY_list = np.zeros((rep, 0))

load_available = False
if load_available and os.path.exists(chkpt_name):
    checkpoint = torch.load(
        chkpt_name, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    dXY_list = checkpoint['dXY_list']
    dX_list = checkpoint['dX_list']
    dY_list = checkpoint['dY_list']
    minee_state_list = checkpoint['minee_state_list']
    for i in range(rep):
        minee_list[i].load_state_dict(minee_state_list[i])


for k in range(2):
    for j in range(2):
        dXY_list = np.append(dXY_list, np.zeros((rep, 1)), axis=1)
        dX_list = np.append(dX_list, np.zeros((rep, 1)), axis=1)
        dY_list = np.append(dY_list, np.zeros((rep, 1)), axis=1)
        for i in range(rep):
            minee_list[i].step()
            dXY_list[i, -1], dX_list[i, -1], dY_list[i, -
                                                     1] = minee_list[i].forward()
        # To show intermediate works
    for i in range(rep):
        plt.plot(dXY_list[i, :])
        plt.plot(dX_list[i, :])
        plt.plot(dY_list[i, :])
    display.clear_output(wait=True)
    display.display(plt.gcf())
display.clear_output()


minee_state_list = [minee_list[i].state_dict() for i in range(rep)]
torch.save({
    'dXY_list': dXY_list,
    'dX_list': dX_list,
    'dY_list': dY_list,
    'minee_state_list': minee_state_list
}, chkpt_name)


def plot_net_1(net, input_min=-5, input_max=5, grids=100):
    """Plot a neural network net. net can only has one input.
    """
    x = np.linspace(input_min, input_max, grids)
    y = net(torch.Tensor(x[:, None]))[:, 0].detach().cpu().numpy()
    plt.plot(x, y)


def plot_net_2(net, Xmin=-5, Xmax=5, Ymin=-5, Ymax=5, Xgrids=100, Ygrids=100):
    """Plot a heat map of a neural network net. net can only have two inputs.
    """
    x, y = np.mgrid[Xmin:Xmax:Xgrids * 1j, Ymin:Ymax:Ygrids * 1j]
    xy = np.concatenate((np.concatenate([x[:, :, None]] * 4, axis=2),
                         np.concatenate([y[:, :, None]] * 4, axis=2)), axis=2)
    z = net(torch.Tensor(xy))[:, :, 0].detach().cpu()
    plt.pcolormesh(x, y, z, cmap='RdBu_r')


plot_net_2(minee_list[0].XY_net)
