import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import copy
import dill
from ..util import plot_util
from ..util.random_util import resample, uniform_sample

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

class Minee():
    def __init__(self, lr, batch_size, hidden_size=100, snapshot=[], iter_num=int(1e+3), model_name="MINEE", log=True, prefix="", ground_truth=0, verbose=False, ref_window_scale=1, ref_batch_factor=1, load_dict=False):
        self.lr = lr
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.snapshot = snapshot
        self.iter_num = iter_num
        self.model_name = model_name
        self.log = log
        self.prefix = prefix
        self.ground_truth = ground_truth
        self.verbose = verbose
        self.ref_window_scale = ref_window_scale
        self.ref_batch_factor = ref_batch_factor
        self.load_dict = load_dict

    def fit(self, Train_X, Train_Y, Test_X, Test_Y):
        self.Train_X = Train_X
        self.Train_Y = Train_Y
        self.Test_X = Test_X
        self.Test_Y = Test_Y

        if self.log:
            log_file = os.path.join(self.prefix, "{}_train.log".format(self.model_name))
            log = open(log_file, "w")
            log.write("lr={0}\n".format(self.lr))
            log.write("batch_size={0}\n".format(self.batch_size))
            log.write("hidden_size={0}\n".format(self.hidden_size))
            log.write("snapshot={0}\n".format(self.snapshot))
            log.write("iter_num={0}\n".format(self.iter_num))
            log.write("model_name={0}\n".format(self.model_name))
            log.write("prefix={0}\n".format(self.prefix))
            log.write("ground_truth={0}\n".format(self.ground_truth))
            log.write("verbose={0}\n".format(self.verbose))
            log.write("ref_window_scale={0}\n".format(self.ref_window_scale))
            log.write("ref_batch_factor={0}\n".format(self.ref_batch_factor))
            log.write("load_dict={0}\n".format(self.load_dict))
            log.close()

        self.XY_net = MineNet(input_size=Train_X.shape[1]+Train_Y.shape[1],hidden_size=self.hidden_size)
        self.X_net = MineNet(input_size=Train_X.shape[1],hidden_size=self.hidden_size)
        self.Y_net = MineNet(input_size=Train_Y.shape[1],hidden_size=self.hidden_size)
        self.XY_optimizer = optim.Adam(self.XY_net.parameters(),lr=self.lr)
        self.X_optimizer = optim.Adam(self.X_net.parameters(),lr=self.lr)
        self.Y_optimizer = optim.Adam(self.Y_net.parameters(),lr=self.lr)

        self.Train_dXY_list = []
        self.Train_dX_list = []
        self.Train_dY_list = []
        self.Test_dXY_list = []
        self.Test_dX_list = []
        self.Test_dY_list = []

        snapshot_i = 0
        # set starting iter_num
        start_i = 0
        fname = os.path.join(self.prefix, "cache.pt")
        if self.load_dict and os.path.exists(fname):
            state_dict = torch.load(fname, map_location = "cuda" if torch.cuda.is_available() else "cpu")
            self.load_state_dict(state_dict)
            if self.verbose:
                print('results loaded from '+fname)

        # For MI estimate
        Train_X_ref = uniform_sample(Train_X,batch_size=int(Train_X.shape[0]*self.ref_batch_factor),window_scale=self.ref_window_scale)
        Train_Y_ref = uniform_sample(Train_Y,batch_size=int(Train_Y.shape[0]*self.ref_batch_factor), window_scale=self.ref_window_scale)

        self.log_ref_size = float(np.log(int(Train_X.shape[0]*self.ref_batch_factor)))
        self.log_batch_size = float(np.log(self.batch_size))
        self.XY_ref_t = torch.Tensor(np.concatenate((Train_X_ref,Train_Y_ref),axis=1))
        self.X_ref_t = torch.Tensor(Train_X_ref)
        self.Y_ref_t = torch.Tensor(Train_Y_ref)

        if len(self.Train_dXY_list) > 0:
            start_i = len(self.Train_dXY_list) + 1
            for i in range(len(self.snapshot)):
                if self.snapshot[i] <= start_i:
                    snapshot_i = i+1
        for i in range(start_i, self.iter_num):
            self.update_mine_net(Train_X, Train_Y, self.batch_size)
            Train_dXY, Train_dX, Train_dY = self.get_estimate(Train_X, Train_Y)
            self.Train_dXY_list = np.append(self.Train_dXY_list, Train_dXY)
            self.Train_dX_list = np.append(self.Train_dX_list, Train_dX)
            self.Train_dY_list = np.append(self.Train_dY_list, Train_dY)

            Test_dXY, Test_dX, Test_dY = self.get_estimate(Test_X, Test_Y)
            self.Test_dXY_list = np.append(self.Test_dXY_list, Test_dXY)
            self.Test_dX_list = np.append(self.Test_dX_list, Test_dX)
            self.Test_dY_list = np.append(self.Test_dY_list, Test_dY)

            if len(self.snapshot)>snapshot_i and (i+1)%self.snapshot[snapshot_i]==0:
                self.save_figure(suffix="iter={}".format(self.snapshot[snapshot_i]))
                # XY_net_state_dict = copy.deepcopy(self.XY_net.state_dict())
                # X_net_state_dict = copy.deepcopy(self.X_net.state_dict())
                # Y_net_state_dict = copy.deepcopy(self.Y_net.state_dict())
                # XY_optim_state_dict = copy.deepcopy(self.XY_optimizer.state_dict())
                # X_optim_state_dict = copy.deepcopy(self.X_optimizer.state_dict())
                # Y_optim_state_dict = copy.deepcopy(self.Y_optimizer.state_dict())
                # To save intermediate works, change the condition to True
                fname_i = os.path.join(self.prefix, "cache_iter={}.pt".format(i+1))
                if True:
                    with open(fname_i,'wb') as f:
                        # dill.dump([XY_net_state_dict,X_net_state_dict,Y_net_state_dict,self.Train_dXY_list,self.Train_dX_list,self.Train_dY_list],f)
                        # dill.dump(self.state_dict(),f)
                        torch.save(self.state_dict(),f)
                        if self.verbose:
                            print('results saved: '+str(snapshot_i))
                snapshot_i += 1

        # To save new results to a db file using the following code, delete the existing db file.
        fname = os.path.join(self.prefix, "cache_iter={}.pt".format(self.iter_num))
        if not os.path.exists(fname):
            with open(fname,'wb') as f:
                torch.save(self.state_dict(),f)
                if self.verbose:
                    print('results saved to '+fname)



    def update_mine_net(self, Train_X, Train_Y, batch_size):
        XY_t = torch.Tensor(np.concatenate((Train_X,Train_Y),axis=1))
        X_t = torch.Tensor(Train_X)
        Y_t = torch.Tensor(Train_Y)
        self.XY_optimizer.zero_grad()
        self.X_optimizer.zero_grad()
        self.Y_optimizer.zero_grad()
        batch_XY = resample(XY_t,batch_size=batch_size)
        batch_X = resample(X_t, batch_size=batch_size)
        batch_Y = resample(Y_t,batch_size=batch_size)
        batch_X_ref = uniform_sample(Train_X,batch_size=int(self.ref_batch_factor*batch_size), window_scale=self.ref_window_scale)
        batch_Y_ref = uniform_sample(Train_Y,batch_size=int(self.ref_batch_factor*batch_size), window_scale=self.ref_window_scale)
        batch_XY_ref = torch.Tensor(np.concatenate((batch_X_ref, batch_Y_ref
        ),axis=1))
        batch_X_ref = batch_XY_ref[:,0:Train_X.shape[1]]
        batch_Y_ref = batch_XY_ref[:,-Train_Y.shape[1]:]

        fXY = self.XY_net(batch_XY)
        # efXY_ref = torch.exp(self.XY_net(batch_XY_ref))
        # batch_dXY = torch.mean(fXY) - torch.log(torch.mean(efXY_ref))
        batch_mar_XY = torch.logsumexp(self.XY_net(batch_XY_ref), 0) - self.log_batch_size
        batch_dXY = torch.mean(fXY) - batch_mar_XY
        batch_loss_XY = -batch_dXY
        batch_loss_XY.backward()
        self.XY_optimizer.step()    

        fX = self.X_net(batch_X)
        # efX_ref = torch.exp(self.X_net(batch_X_ref))
        # batch_dX = torch.mean(fX) - torch.log(torch.mean(efX_ref))
        batch_mar_X = torch.logsumexp(self.X_net(batch_X_ref), 0) - self.log_batch_size
        batch_dX = torch.mean(fX) - batch_mar_X
        batch_loss_X = -batch_dX
        batch_loss_X.backward()
        self.X_optimizer.step()    
        
        fY = self.Y_net(batch_Y)
        # efY_ref = torch.exp(self.Y_net(batch_Y_ref))
        # batch_dY = torch.mean(fY) - torch.log(torch.mean(efY_ref))
        batch_mar_Y = torch.logsumexp(self.Y_net(batch_Y_ref), 0) - self.log_batch_size
        batch_dY = torch.mean(fY) - batch_mar_Y
        batch_loss_Y = -batch_dY
        batch_loss_Y.backward()
        self.Y_optimizer.step()    

    def get_estimate(self, X, Y):
        XY_t = torch.Tensor(np.concatenate((X,Y),axis=1))
        X_t = torch.Tensor(X)
        Y_t = torch.Tensor(Y)

        # dXY = torch.mean(self.XY_net(XY_t)) - torch.log(torch.mean(torch.exp(self.XY_net(self.XY_ref_t))))
        # dX = torch.mean(self.X_net(X_t)) - torch.log(torch.mean(torch.exp(self.X_net(self.X_ref_t))))
        # dY = torch.mean(self.Y_net(Y_t)) - torch.log(torch.mean(torch.exp(self.Y_net(self.Y_ref_t))))
        dXY = torch.mean(self.XY_net(XY_t)) - (torch.logsumexp(self.XY_net(self.XY_ref_t), 0) - self.log_ref_size)
        dX = torch.mean(self.X_net(X_t)) - (torch.logsumexp(self.X_net(self.X_ref_t), 0) - self.log_ref_size)
        dY = torch.mean(self.Y_net(Y_t)) - (torch.logsumexp(self.Y_net(self.Y_ref_t), 0) - self.log_ref_size)
        return dXY.cpu().item(), dX.cpu().item(), dY.cpu().item()

    def predict(self, Train_X, Train_Y, Test_X, Test_Y):
        Train_X, Train_Y = np.array(Train_X), np.array(Train_Y)
        Test_X, Test_Y = np.array(Test_X), np.array(Test_Y)
        self.fit(Train_X,Train_Y, Test_X, Test_Y)

        mi_lb = self.Train_dXY_list[-1] - self.Train_dY_list[-1] - self.Train_dX_list[-1]

        if self.log:
            self.save_figure(suffix="iter={}".format(self.iter_num))
        self.Train_X = []
        self.Train_Y = []
        self.Test_X = []
        self.Test_Y = []

        self.XY_ref_t = []
        self.X_ref_t = []
        self.Y_ref_t = []
        self.XY_net = []
        self.X_net = []
        self.Y_net = []
        self.XY_optimizer = []
        self.X_optimizer = []
        self.Y_optimizer = []

        self.Train_dXY_list = []
        self.Train_dX_list = []
        self.Train_dY_list = []
        self.Test_dXY_list = []
        self.Test_dX_list = []
        self.Test_dY_list = []
        return mi_lb

    def save_figure(self, suffix=""):
        fig, ax = plt.subplots(2,4, figsize=(90, 30))
        #plot Data
        axCur = ax[0,0]
        axCur.plot(self.Train_dXY_list, label='XY')
        axCur.plot(self.Train_dX_list, label='X')
        axCur.plot(self.Train_dY_list, label='Y')
        axCur.legend()
        axCur.set_xlabel("number of iterations")
        axCur.set_ylabel('divergence estimates')
        axCur.set_title('divergence estimates of training data')

        #plot training curve
        axCur = ax[0,1]
        axCur.plot(self.Test_dXY_list, label='XY')
        axCur.plot(self.Test_dX_list, label='X')
        axCur.plot(self.Test_dY_list, label='Y')
        axCur.legend()
        axCur.set_xlabel("number of iterations")
        axCur.set_ylabel('divergence estimates')
        axCur.set_title('divergence estimates of testing data')

        #plot mi_lb curve
        axCur = ax[0,2]
        Train_mi_lb = self.Train_dXY_list-self.Train_dX_list-self.Train_dY_list
        axCur = plot_util.getTrainCurve(Train_mi_lb , [], axCur, show_min=False, ground_truth=self.ground_truth)
        axCur.set_title('curve of training data mutual information')

        #plot mi_lb curve
        axCur = ax[0,3]
        Test_mi_lb = self.Test_dXY_list-self.Test_dX_list-self.Test_dY_list
        axCur = plot_util.getTrainCurve(Test_mi_lb , [], axCur, show_min=False, ground_truth=self.ground_truth)
        axCur.set_title('curve of testing data mutual information')

        # Trained Function contour plot
        if self.Train_X.shape[1] == 1 and self.Train_Y.shape[1] == 1:
            Xmax = self.Train_X.max()
            Xmin = self.Train_X.min()
            Ymax = self.Train_Y.max()
            Ymin = self.Train_Y.min()
            x = np.linspace(Xmin, Xmax, 300)
            y = np.linspace(Ymin, Ymax, 300)
            xs, ys = np.meshgrid(x,y)
            # mesh = torch.FloatTensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))
            mesh = torch.FloatTensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))
            fxy = self.XY_net(mesh)
            fx = self.X_net(mesh[:,[0]])
            fy = self.Y_net(mesh[:,[1]])
            ixy = (fxy - fx - fy).detach().numpy()
            ixy = ixy.reshape(xs.shape[1], ys.shape[0])

            axCur = ax[1,0]
            axCur, c = plot_util.getHeatMap(axCur, xs, ys, ixy)
            fig.colorbar(c, ax=axCur)
            axCur.set_title('heatmap of i(x,y)')

            fxy = fxy.detach().numpy().reshape(xs.shape[1], ys.shape[0])
            axCur = ax[1,1]
            axCur, c = plot_util.getHeatMap(axCur, xs, ys, fxy)
            fig.colorbar(c, ax=axCur)
            axCur.set_title('heatmap T(X,Y) for learning H(X,Y)')

            axCur = ax[1,2]
            axCur.scatter(self.Train_X, self.Train_Y, color='red', marker='o', label='train')
            axCur.scatter(self.Test_X, self.Test_Y, color='green', marker='x', label='test')
            axCur.set_title('Plot of all train data samples and test data samples')
            axCur.legend()


        figName = os.path.join(self.prefix, "{}_{}".format(self.model_name, suffix))
        fig.savefig(figName, bbox_inches='tight')
        plt.close()

    def state_dict(self):
        return {
            'XY_net': self.XY_net.state_dict(),
            'XY_optimizer': self.XY_optimizer.state_dict(),
            'X_net': self.X_net.state_dict(),
            'X_optimizer': self.X_optimizer.state_dict(),
            'Y_net': self.Y_net.state_dict(),
            'Y_optimizer': self.Y_optimizer.state_dict(),
            'Train_X': self.Train_X,
            'Train_Y': self.Train_Y,
            'Test_X': self.Test_X,
            'Test_Y': self.Test_Y,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'ref_batch_factor': self.ref_batch_factor,
            'ref_window_scale': self.ref_window_scale,
            'Train_dXY_list' :self.Train_dXY_list,
            'Train_dX_list' :self.Train_dX_list,
            'Train_dY_list' :self.Train_dY_list,
            'Test_dXY_list' :self.Test_dXY_list,
            'Test_dX_list' :self.Test_dX_list,
            'Test_dY_list' :self.Test_dY_list
        }

    def load_state_dict(self, state_dict):
        self.XY_net.load_state_dict(state_dict['XY_net'])
        self.XY_optimizer.load_state_dict(state_dict['XY_optimizer'])
        self.X_net.load_state_dict(state_dict['X_net'])
        self.X_optimizer.load_state_dict(state_dict['X_optimizer'])
        self.Y_net.load_state_dict(state_dict['Y_net'])
        self.Y_optimizer.load_state_dict(state_dict['Y_optimizer'])
        self.Train_X = state_dict['Train_X']
        self.Train_Y = state_dict['Train_Y']
        self.Test_X = state_dict['Test_X']
        self.Test_Y = state_dict['Test_Y']
        if 'lr' in state_dict:
            self.lr = state_dict['lr']
        if 'batch_size' in state_dict:
            self.batch_size = state_dict['batch_size']
        if 'ref_batch_factor' in state_dict:
            self.ref_batch_factor = state_dict['ref_batch_factor']
        if 'ref_window_scale' in state_dict:
            self.ref_window_scale = state_dict['ref_window_scale']
        self.Train_dXY_list = state_dict['Train_dXY_list']
        self.Train_dX_list = state_dict['Train_dX_list']
        self.Train_dY_list = state_dict['Train_dY_list']
        self.Test_dXY_list = state_dict['Test_dXY_list']
        self.Test_dX_list = state_dict['Test_dX_list']
        self.Test_dY_list = state_dict['Test_dY_list']
