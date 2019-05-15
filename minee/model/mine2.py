import torch
torch.manual_seed(seed=1)

# Use GPU when available
# Need to use Tensor to create the tensor of default type
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)


import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import copy
import dill
from ..util import plot_util
from ..util import torch_util
from ..util.random_util import resample

# from ..utils import save_train_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MineNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

import torch.optim as optim

class Mine():
    def __init__(self, lr, batch_size, ma_rate, hidden_size=100, snapshot=[], iter_num=int(1e+3), model_name="MINE", log=True, prefix="", ground_truth=0, verbose=False):
        self.lr = lr
        self.batch_size = batch_size
        self.ma_rate = ma_rate
        self.hidden_size = hidden_size
        self.snapshot = snapshot
        self.iter_num = iter_num
        self.model_name = model_name
        self.log = log
        self.prefix = prefix
        self.ground_truth = ground_truth
        self.verbose = verbose

    def predict(self, X, Y, a, b):
        X_ref, Y_ref = np.meshgrid(X, Y)
        X_ref = X_ref.flatten()[:,None]
        Y_ref = Y_ref.flatten()[:,None]

        plt.scatter(X,Y)
        plt.scatter(X,Y,label="data",marker="+",color="steelblue")
        plt.scatter(X_ref,Y_ref,label="ref",marker="_",color="darkorange")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Plot of all data samples and reference samples')
        plt.legend()
        figName = os.path.join(self.prefix, "mix_gaussian_mine_full_batch_ma_data.png")
        plt.savefig(figName)
        plt.close()

        XY_t = torch.Tensor(np.concatenate((X,Y),axis=1))
        XY_ref_t = torch.Tensor(np.concatenate((X_ref,Y_ref),axis=1))

        XY_net = MineNet(input_size=X.shape[1]+Y.shape[1],hidden_size=300)
        XY_optimizer = optim.Adam(XY_net.parameters(),lr=self.lr)
        ma_ef = 1
        dXY_list = []
        for j in range(50):
            for i in range(200):
                XY_optimizer.zero_grad()
                batch_XY = resample(XY_t,batch_size=self.batch_size)
                batch_XY_ref = torch.Tensor(np.concatenate((resample(X,batch_size=self.batch_size),                                                         resample(Y,batch_size=self.batch_size)),axis=1))
                
                fXY = XY_net(batch_XY)
                efXY_ref = torch.exp(XY_net(batch_XY_ref))
                batch_dXY = torch.mean(fXY) - torch.log(torch.mean(efXY_ref))
                ma_ef = (1-self.ma_rate)*ma_ef + self.ma_rate*torch.mean(efXY_ref)
                batch_loss_XY = -(torch.mean(fXY) - (1/ma_ef.mean()).detach()*torch.mean(efXY_ref))
                batch_loss_XY.backward()
                XY_optimizer.step()    
                dXY_list = np.append(dXY_list,(torch.mean(XY_net(XY_t)) - torch.log(torch.mean(torch.exp(XY_net(XY_ref_t))))).cpu().item())
            mi_list = dXY_list
            plt.plot(mi_list)
            plt.title("Plot of MI estimates against number iteractions")
            plt.xlabel("number of iterations")
            plt.ylabel("MI estimate")
            # plt.show()
            figName = os.path.join(self.prefix, "mix_gaussian_mine_full_batch_ma_iter={}.png".format(j))
            plt.savefig(figName)
            plt.close()
        mi_lb = dXY_list[-1]
        return mi_lb
