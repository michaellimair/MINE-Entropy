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
    def __init__(self, lr, batch_size, hidden_size=100, snapshot=[], iter_num=int(1e+3), model_name="MINEE", log=True, prefix="", ground_truth=0, verbose=False):
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

    def fit(self, Train_X, Train_Y, Test_X, Test_Y):
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
            log.write("verbose={}\n".format(self.verbose))
            log.close()

        self.Train_X = Train_X
        self.Train_Y = Train_Y
        self.Test_X = Test_X
        self.Test_Y = Test_Y
        # For MI estimate
        Train_X_ref = uniform_sample(Train_X.min(axis=0),Train_X.max(axis=0),Train_X.shape[0])
        Train_Y_ref = uniform_sample(Train_Y.min(axis=0),Train_Y.max(axis=0),Train_Y.shape[0])

        self.XY_ref_t = torch.Tensor(np.concatenate((Train_X_ref,Train_Y_ref),axis=1))
        self.X_ref_t = torch.Tensor(Train_X_ref)
        self.Y_ref_t = torch.Tensor(Train_Y_ref)

        self.XY_net = MineNet(input_size=Train_X.shape[1]+Train_Y.shape[1],hidden_size=self.hidden_size)
        self.X_net = MineNet(input_size=Train_X.shape[1],hidden_size=self.hidden_size)
        self.Y_net = MineNet(input_size=Train_Y.shape[1],hidden_size=self.hidden_size)
        self.XY_optimizer = optim.Adam(self.XY_net.parameters(),lr=self.lr)
        self.X_optimizer = optim.Adam(self.X_net.parameters(),lr=self.lr)
        self.Y_optimizer = optim.Adam(self.Y_net.parameters(),lr=self.lr)

        XY_net_list = []
        X_net_list = []
        Y_net_list = []

        self.Train_dXY_list = []
        self.Train_dX_list = []
        self.Train_dY_list = []
        self.Test_dXY_list = []
        self.Test_dX_list = []
        self.Test_dY_list = []

        fname = os.path.join(self.prefix, "cache_iter={}.pt".format(self.iter_num))
        if os.path.exists(fname):
            with open(fname,'rb') as f:
                checkpoint = torch.load(fname,map_location = "cuda" if torch.cuda.is_available() else "cpu")
                XY_net_list = checkpoint['XY_net_list']
                X_net_list = checkpoint['X_net_list']
                Y_net_list = checkpoint['Y_net_list']
                self.Train_dXY_list = checkpoint['Train_dXY_list']
                self.Train_dX_list = checkpoint['Train_dX_list']
                self.Train_dY_list = checkpoint['Train_dY_list']
                self.XY_net.load_state_dict(XY_net_list[-1])
                self.X_net.load_state_dict(X_net_list[-1])
                self.Y_net.load_state_dict(Y_net_list[-1])
                if self.verbose:
                    print('results loaded from '+fname)
        else:
            snapshot_i = 0
            for i in range(self.iter_num):
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
                    XY_net_list = np.append(XY_net_list,copy.deepcopy(self.XY_net.state_dict()))
                    X_net_list = np.append(X_net_list,copy.deepcopy(self.X_net.state_dict()))
                    Y_net_list = np.append(Y_net_list,copy.deepcopy(self.Y_net.state_dict()))
                    # To save intermediate works, change the condition to True
                    fname_i = os.path.join(self.prefix, "cache_iter={}.pt".format(i+1))
                    if True:
                        with open(fname_i,'wb') as f:
                            dill.dump([XY_net_list,X_net_list,Y_net_list,self.Train_dXY_list,self.Train_dX_list,self.Train_dY_list],f)
                            if self.verbose:
                                print('results saved: '+str(snapshot_i))
                    snapshot_i += 1

        # To save new results to a db file using the following code, delete the existing db file.
        if not os.path.exists(fname):
            with open(fname,'wb') as f:
                torch.save({
                    'Train_dXY_list' : self.Train_dXY_list,
                    'Train_dX_list' : self.Train_dX_list,
                    'Train_dY_list' : self.Train_dY_list,
                    'XY_net_list' : XY_net_list,
                    'X_net_list' : X_net_list,
                    'Y_net_list' : Y_net_list
                },f)
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
        batch_XY_ref = torch.Tensor(np.concatenate((uniform_sample(Train_X.min(axis=0),Train_X.max(axis=0),batch_size), uniform_sample(Train_Y.min(axis=0),Train_Y.max(axis=0),batch_size)),axis=1))
        batch_X_ref = batch_XY_ref[:,0:Train_X.shape[1]]
        batch_Y_ref = batch_XY_ref[:,-Train_Y.shape[1]:]

        fXY = self.XY_net(batch_XY)
        efXY_ref = torch.exp(self.XY_net(batch_XY_ref))
        batch_dXY = torch.mean(fXY) - torch.log(torch.mean(efXY_ref))
        batch_loss_XY = -batch_dXY
        batch_loss_XY.backward()
        self.XY_optimizer.step()    

        fX = self.X_net(batch_X)
        efX_ref = torch.exp(self.X_net(batch_X_ref))
        batch_dX = torch.mean(fX) - torch.log(torch.mean(efX_ref))
        batch_loss_X = -batch_dX
        batch_loss_X.backward()
        self.X_optimizer.step()    
        
        fY = self.Y_net(batch_Y)
        efY_ref = torch.exp(self.Y_net(batch_Y_ref))
        batch_dY = torch.mean(fY) - torch.log(torch.mean(efY_ref))
        batch_loss_Y = -batch_dY
        batch_loss_Y.backward()
        self.Y_optimizer.step()    

    def get_estimate(self, X, Y):
        XY_t = torch.Tensor(np.concatenate((X,Y),axis=1))
        X_t = torch.Tensor(X)
        Y_t = torch.Tensor(Y)

        dXY = torch.mean(self.XY_net(XY_t)) - torch.log(torch.mean(torch.exp(self.XY_net(self.XY_ref_t))))
        dX = torch.mean(self.X_net(X_t)) - torch.log(torch.mean(torch.exp(self.X_net(self.X_ref_t))))
        dY = torch.mean(self.Y_net(Y_t)) - torch.log(torch.mean(torch.exp(self.Y_net(self.Y_ref_t))))
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