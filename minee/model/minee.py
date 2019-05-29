
# random_seed = 1
import numpy as np
# np.random.seed(seed=random_seed)
import torch
# torch.manual_seed(seed=random_seed)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    def __init__(self, lr, batch_size, hidden_size=100, snapshot=[], iter_num=int(1e+3), model_name="MINEE", log=True, prefix="", ground_truth=0, verbose=False, ref_window_scale=1, ref_batch_factor=1, load_dict=False, rep=1, fix_ref_est=False):
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
        self.rep = rep
        self.fix_ref_est = fix_ref_est

    # def fit(self, Train_X, Train_Y, Test_X, Test_Y):
    #     self.Train_X = Train_X
    #     self.Train_Y = Train_Y
    #     self.Test_X = Test_X
    #     self.Test_Y = Test_Y
    def fit(self, data_model):
        data_train = data_model.data
        data_test = data_model.data
        if data_train.shape[1]%2 == 1 or data_test.shape[1]%2 == 1:
            raise ValueError("dim of data should be even")
        self.sample_size = data_train.shape[0]
        self.dim = data_train.shape[1]//2
        self.Trainlist_X = []
        self.Trainlist_Y = []
        self.Testlist_X = []
        self.Testlist_Y = []
        for i in range(self.rep):
            if i > 0:
                data_train = data_model.data
                data_test = data_model.data
            self.Trainlist_X.append(data_train[:,0:data_train.shape[1]//2].copy())
            self.Trainlist_Y.append(data_train[:,-data_train.shape[1]//2:].copy())
            self.Testlist_X.append(data_test[:,0:data_test.shape[1]//2].copy())
            self.Testlist_Y.append(data_test[:,-data_test.shape[1]//2:].copy())

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
            log.write("rep={0}\n".format(self.rep))
            log.write("fix_ref_est={0}\n".format(self.fix_ref_est))
            log.write("dim={0}\n".format(self.dim))
            log.write("sample_size={0}\n".format(self.sample_size))
            log.close()

        self.XYlist_net = []
        self.Xlist_net = []
        self.Ylist_net = []
        self.XYlist_optimizer = []
        self.Xlist_optimizer = []
        self.Ylist_optimizer = []

        for i in range(self.rep):
            self.XYlist_net.append(MineNet(input_size=self.dim*2,hidden_size=self.hidden_size))
            self.Xlist_net.append(MineNet(input_size=self.dim,hidden_size=self.hidden_size))
            self.Ylist_net.append(MineNet(input_size=self.dim,hidden_size=self.hidden_size))
            self.XYlist_optimizer.append(optim.Adam(self.XYlist_net[i].parameters(),lr=self.lr))
            self.Xlist_optimizer.append(optim.Adam(self.Xlist_net[i].parameters(),lr=self.lr))
            self.Ylist_optimizer.append(optim.Adam(self.Ylist_net[i].parameters(),lr=self.lr))

        self.Train_dXY_list = np.zeros((self.rep, 0))
        self.Train_dX_list = np.zeros((self.rep, 0))
        self.Train_dY_list = np.zeros((self.rep, 0))
        self.Test_dXY_list = np.zeros((self.rep, 0))
        self.Test_dX_list = np.zeros((self.rep, 0))
        self.Test_dY_list = np.zeros((self.rep, 0))

        snapshot_i = 0
        # set starting iter_num
        start_i = 0
        fname = os.path.join(self.prefix, "cache.pt")
        if self.load_dict and os.path.exists(fname):
            state_dict = torch.load(fname, map_location = "cuda" if torch.cuda.is_available() else "cpu")
            self.load_state_dict(state_dict)
            if self.verbose:
                print('results loaded from '+fname)

        self.log_ref_size = float(np.log(int(self.sample_size*self.ref_batch_factor)))
        self.log_batch_size = float(np.log(self.batch_size*self.ref_batch_factor))

        # For MI estimate
        self.XYlist_ref_t = []
        self.Xlist_ref_t = []
        self.Ylist_ref_t = []
        if self.fix_ref_est:
            for i in range(self.rep):
                Train_X_ref = uniform_sample(self.Trainlist_X[i],batch_size=int(self.sample_size*self.ref_batch_factor),window_scale=self.ref_window_scale)
                Train_Y_ref = uniform_sample(self.Trainlist_Y[i],batch_size=int(self.sample_size*self.ref_batch_factor), window_scale=self.ref_window_scale)

                self.XYlist_ref_t.append(torch.Tensor(np.concatenate((Train_X_ref,Train_Y_ref),axis=1)))
                self.Xlist_ref_t.append(torch.Tensor(Train_X_ref))
                self.Ylist_ref_t.append(torch.Tensor(Train_Y_ref))

        if type(self.Train_dXY_list)==np.ndarray and self.Train_dXY_list.ndim == 2 and len(self.Train_dXY_list[0,:]) > 0:
            start_i = len(self.Train_dXY_list[0,:]) + 1
            for i in range(len(self.snapshot)):
                if self.snapshot[i] <= start_i:
                    snapshot_i = i+1
        for i in range(start_i, self.iter_num):
            self.update_mine_net(self.Trainlist_X, self.Trainlist_Y, self.batch_size)
            Train_dXY, Train_dX, Train_dY = self.get_estimate(self.Trainlist_X, self.Trainlist_Y)
            self.Train_dXY_list = np.append(self.Train_dXY_list, Train_dXY, axis=1)
            self.Train_dX_list = np.append(self.Train_dX_list, Train_dX, axis=1)
            self.Train_dY_list = np.append(self.Train_dY_list, Train_dY, axis=1)

            Test_dXY, Test_dX, Test_dY = self.get_estimate(self.Testlist_X, self.Testlist_Y)
            self.Test_dXY_list = np.append(self.Test_dXY_list, Test_dXY, axis=1)
            self.Test_dX_list = np.append(self.Test_dX_list, Test_dX, axis=1)
            self.Test_dY_list = np.append(self.Test_dY_list, Test_dY, axis=1)

            if len(self.snapshot)>snapshot_i and (i+1)%self.snapshot[snapshot_i]==0:
                self.save_figure(suffix="iter={}".format(self.snapshot[snapshot_i]))
                # To save intermediate works, change the condition to True
                fname_i = os.path.join(self.prefix, "cache_iter={}.pt".format(i+1))
                if True:
                    with open(fname_i,'wb') as f:
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



    def update_mine_net(self, X, Y, batch_size):
        for i in range(self.rep):
            XY_t = torch.Tensor(np.concatenate((X[i],Y[i]),axis=1))
            X_t = torch.Tensor(X[i])
            Y_t = torch.Tensor(Y[i])
            batch_XY = resample(XY_t,batch_size=batch_size)
            batch_X = resample(X_t, batch_size=batch_size)
            batch_Y = resample(Y_t,batch_size=batch_size)
            batch_X_ref = uniform_sample(X[i],batch_size=int(self.ref_batch_factor*batch_size), window_scale=self.ref_window_scale)
            batch_Y_ref = uniform_sample(Y[i],batch_size=int(self.ref_batch_factor*batch_size), window_scale=self.ref_window_scale)
            batch_XY_ref = torch.Tensor(np.concatenate((batch_X_ref, batch_Y_ref),axis=1))
            batch_X_ref = batch_XY_ref[:,0:self.dim]
            batch_Y_ref = batch_XY_ref[:,-self.dim:]
            self.XYlist_optimizer[i].zero_grad()
            self.Xlist_optimizer[i].zero_grad()
            self.Ylist_optimizer[i].zero_grad()

            fXY = self.XYlist_net[i](batch_XY)
            batch_mar_XY = torch.logsumexp(self.XYlist_net[i](batch_XY_ref), 0) - self.log_batch_size
            batch_dXY = torch.mean(fXY) - batch_mar_XY
            batch_loss_XY = -batch_dXY
            batch_loss_XY.backward()
            self.XYlist_optimizer[i].step()    

            fX = self.Xlist_net[i](batch_X)
            batch_mar_X = torch.logsumexp(self.Xlist_net[i](batch_X_ref), 0) - self.log_batch_size
            batch_dX = torch.mean(fX) - batch_mar_X
            batch_loss_X = -batch_dX
            batch_loss_X.backward()
            self.Xlist_optimizer[i].step()    
            
            fY = self.Ylist_net[i](batch_Y)
            batch_mar_Y = torch.logsumexp(self.Ylist_net[i](batch_Y_ref), 0) - self.log_batch_size
            batch_dY = torch.mean(fY) - batch_mar_Y
            batch_loss_Y = -batch_dY
            batch_loss_Y.backward()
            self.Ylist_optimizer[i].step()    

    def get_estimate(self, X, Y):

        dXY_list = np.zeros((self.rep, 1))
        dY_list = np.zeros((self.rep, 1))
        dX_list = np.zeros((self.rep, 1))
        for i in range(self.rep):
            XY_t = torch.Tensor(np.concatenate((X[i],Y[i]),axis=1))
            X_t = torch.Tensor(X[i])
            Y_t = torch.Tensor(Y[i])
            if self.fix_ref_est:
                XY_ref_t = self.XYlist_ref_t[i]
                X_ref_t = self.Xlist_ref_t[i]
                Y_ref_t = self.Ylist_ref_t[i]
            else:
                Train_X_ref = uniform_sample(X[i],batch_size=int(self.sample_size*self.ref_batch_factor),window_scale=self.ref_window_scale)
                Train_Y_ref = uniform_sample(Y[i],batch_size=int(self.sample_size*self.ref_batch_factor), window_scale=self.ref_window_scale)

                XY_ref_t = torch.Tensor(np.concatenate((Train_X_ref,Train_Y_ref),axis=1))
                Y_ref_t = torch.Tensor(Train_Y_ref)
                X_ref_t = torch.Tensor(Train_X_ref)
            dXY = torch.mean(self.XYlist_net[i](XY_t)) - (torch.logsumexp(self.XYlist_net[i](XY_ref_t), 0) - self.log_ref_size)
            dX = torch.mean(self.Xlist_net[i](X_t)) - (torch.logsumexp(self.Xlist_net[i](X_ref_t), 0) - self.log_ref_size)
            dY = torch.mean(self.Ylist_net[i](Y_t)) - (torch.logsumexp(self.Ylist_net[i](Y_ref_t), 0) - self.log_ref_size)

            dXY_list[i, 0] = dXY.cpu().item()
            dY_list[i, 0] = dY.cpu().item()
            dX_list[i, 0] = dX.cpu().item()

        return dXY_list, dX_list, dY_list

    # def predict(self, Train_X, Train_Y, Test_X, Test_Y):
    #     Train_X, Train_Y = np.array(Train_X), np.array(Train_Y)
    #     Test_X, Test_Y = np.array(Test_X), np.array(Test_Y)
    #     self.fit(Train_X,Train_Y, Test_X, Test_Y)
    def predict(self, data_model):
        self.fit(data_model)

        mi_lb = np.average(self.Train_dXY_list[:,-1]) - np.average(self.Train_dY_list[:,-1]) - np.average(self.Train_dX_list[:,-1])

        if self.log:
            self.save_figure(suffix="iter={}".format(self.iter_num))
        self.Trainlist_X = []
        self.Trainlist_Y = []
        self.Testlist_X = []
        self.Testlist_Y = []

        self.XYlist_ref_t = []
        self.Xlist_ref_t = []
        self.Ylist_ref_t = []
        self.XYlist_net = []
        self.Xlist_net = []
        self.Ylist_net = []
        self.XYlist_optimizer = []
        self.Xlist_optimizer = []
        self.Ylist_optimizer = []

        self.Train_dXY_list = np.zeros((self.rep, 0))
        self.Train_dX_list = np.zeros((self.rep, 0))
        self.Train_dY_list = np.zeros((self.rep, 0))
        self.Test_dXY_list = np.zeros((self.rep, 0))
        self.Test_dX_list = np.zeros((self.rep, 0))
        self.Test_dY_list = np.zeros((self.rep, 0))
        return mi_lb

    def save_figure(self, suffix=""):
        if len(self.XYlist_net) == 1:
            fig, ax = plt.subplots(2,4, figsize=(90, 30))
            #plot Data
            axCur = ax[0,0]
            axCur.plot(self.Train_dXY_list[0,:], label='XY')
            axCur.plot(self.Train_dX_list[0,:], label='X')
            axCur.plot(self.Train_dY_list[0,:], label='Y')
            axCur.legend()
            axCur.set_xlabel("number of iterations")
            axCur.set_ylabel('divergence estimates')
            axCur.set_title('divergence estimates of training data')

            #plot training curve
            axCur = ax[0,1]
            axCur.plot(self.Test_dXY_list[0,:], label='XY')
            axCur.plot(self.Test_dX_list[0,:], label='X')
            axCur.plot(self.Test_dY_list[0,:], label='Y')
            axCur.legend()
            axCur.set_xlabel("number of iterations")
            axCur.set_ylabel('divergence estimates')
            axCur.set_title('divergence estimates of testing data')

            #plot mi_lb curve
            axCur = ax[0,2]
            Train_mi_lb = self.Train_dXY_list[0,:]-self.Train_dX_list[0,:]-self.Train_dY_list[0,:]
            axCur = plot_util.getTrainCurve(Train_mi_lb , [], axCur, show_min=False, ground_truth=self.ground_truth)
            axCur.set_title('curve of training data mutual information')

            #plot mi_lb curve
            axCur = ax[0,3]
            Test_mi_lb = self.Test_dXY_list[0,:]-self.Test_dX_list[0,:]-self.Test_dY_list[0,:]
            axCur = plot_util.getTrainCurve(Test_mi_lb , [], axCur, show_min=False, ground_truth=self.ground_truth)
            axCur.set_title('curve of testing data mutual information')

            # Trained Function contour plot
            if len(self.Trainlist_X) == 1 and self.Trainlist_X[0].shape[1] == 1 and self.Trainlist_Y[0].shape[1] == 1:
                Xmax = self.Trainlist_X[0].max()
                Xmin = self.Trainlist_X[0].min()
                Ymax = self.Trainlist_Y[0].max()
                Ymin = self.Trainlist_Y[0].min()
                x = np.linspace(Xmin, Xmax, 300)
                y = np.linspace(Ymin, Ymax, 300)
                xs, ys = np.meshgrid(x,y)
                # mesh = torch.FloatTensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))
                mesh = torch.FloatTensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))
                fxy = self.XYlist_net[0](mesh)
                fx = self.Xlist_net[0](mesh[:,[0]])
                fy = self.Ylist_net[0](mesh[:,[1]])
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
                axCur.scatter(self.Trainlist_X[0], self.Trainlist_Y[0], color='red', marker='o', label='train')
                axCur.scatter(self.Testlist_X[0], self.Testlist_Y[0], color='green', marker='x', label='test')
                axCur.set_title('Plot of all train data samples and test data samples')
                axCur.legend()
        else:
            fig, ax = plt.subplots(3,4, figsize=(90, 45))
            axCur = ax[0,0]
            for i in range(self.rep):
                axCur.plot(self.Train_dXY_list[i,:], label="XY_{}".format(i))
            axCur.legend()
            axCur.set_xlabel("number of iterations")
            axCur.set_ylabel('divergence estimates')
            axCur.set_title('XY divergence estimates of training data')

            axCur = ax[0,1]
            for i in range(self.rep):
                axCur.plot(self.Test_dXY_list[i,:], label="XY_{}".format(i))
            axCur.legend()
            axCur.set_xlabel("number of iterations")
            axCur.set_ylabel('divergence estimates')
            axCur.set_title('XY divergence estimates of testing data')
            
            axCur = ax[0,2]
            for i in range(self.rep):
                axCur.plot(self.Train_dX_list[i,:], label="X_{}".format(i))
            axCur.legend()
            axCur.set_xlabel("number of iterations")
            axCur.set_ylabel('divergence estimates')
            axCur.set_title('X divergence estimates of training data')

            axCur = ax[0,3]
            for i in range(self.rep):
                axCur.plot(self.Test_dX_list[i,:], label="X_{}".format(i))
            axCur.legend()
            axCur.set_xlabel("number of iterations")
            axCur.set_ylabel('divergence estimates')
            axCur.set_title('X divergence estimates of testing data')
            
            axCur = ax[1,0]
            for i in range(self.rep):
                axCur.plot(self.Train_dY_list[i,:], label="Y_{}".format(i))
            axCur.legend()
            axCur.set_xlabel("number of iterations")
            axCur.set_ylabel('divergence estimates')
            axCur.set_title('Y divergence estimates of training data')

            axCur = ax[1,1]
            for i in range(self.rep):
                axCur.plot(self.Test_dY_list[i,:], label="Y_{}".format(i))
            axCur.legend()
            axCur.set_xlabel("number of iterations")
            axCur.set_ylabel('divergence estimates')
            axCur.set_title('Y divergence estimates of testing data')

            #plot mi_lb curve
            axCur = ax[1,2]
            Train_mi_lb = self.Train_dXY_list.copy()
            Train_mi_lb = Train_mi_lb-self.Train_dX_list-self.Train_dY_list
            axCur = plot_util.getTrainCurve(Train_mi_lb , [], axCur, show_min=False, ground_truth=self.ground_truth)
            axCur.set_title('curve of training data mutual information')

            #plot mi_lb curve
            axCur = ax[1,3]
            Test_mi_lb = self.Test_dXY_list.copy()
            Test_mi_lb = Test_mi_lb-self.Test_dX_list-self.Test_dY_list
            axCur = plot_util.getTrainCurve(Test_mi_lb , [], axCur, show_min=False, ground_truth=self.ground_truth)
            axCur.set_title('curve of testing data mutual information')

            #plot mi_lb curve
            axCur = ax[2,2]
            Train_mi_lb = self.Train_dXY_list.copy()
            Train_mi_lb = Train_mi_lb-self.Train_dX_list-self.Train_dY_list
            axCur = plot_util.getTrainCurve(Train_mi_lb , [], axCur, show_min=False, ground_truth=self.ground_truth, ma_rate=0.01)
            axCur.set_title('curve of training data mutual information with 0.01 ma')

            #plot mi_lb curve
            axCur = ax[2,3]
            Test_mi_lb = self.Test_dXY_list.copy()
            Test_mi_lb = Test_mi_lb-self.Test_dX_list-self.Test_dY_list
            axCur = plot_util.getTrainCurve(Test_mi_lb , [], axCur, show_min=False, ground_truth=self.ground_truth, ma_rate=0.01)
            axCur.set_title('curve of testing data mutual information with 0.01 ma')


        figName = os.path.join(self.prefix, "{}_{}".format(self.model_name, suffix))
        fig.savefig(figName, bbox_inches='tight')
        plt.close()

    def state_dict(self):
        return {
            'XYlist_net': self.XYlist_net,
            'XYlist_optimizer': self.XYlist_optimizer,
            'Xlist_net': self.Xlist_net,
            'Xlist_optimizer': self.Xlist_optimizer,
            'Ylist_net': self.Ylist_net,
            'Ylist_optimizer': self.Ylist_optimizer,
            'Trainlist_X': self.Trainlist_X,
            'Trainlist_Y': self.Trainlist_Y,
            'Testlist_X': self.Testlist_X,
            'Testlist_Y': self.Testlist_Y,
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
        self.XYlist_net = state_dict['XYlist_net']
        self.XYlist_optimizer = state_dict['XYlist_optimizer']
        self.Xlist_net = state_dict['Xlist_net']
        self.Xlist_optimizer = state_dict['Xlist_optimizer']
        self.Ylist_net = state_dict['Ylist_net']
        self.Ylist_optimizer = state_dict['Ylist_optimizer']
        self.Trainlist_X = state_dict['Trainlist_X']
        self.Trainlist_Y = state_dict['Trainlist_Y']
        self.Testlist_X = state_dict['Testlist_X']
        self.Testlist_Y = state_dict['Testlist_Y']
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
