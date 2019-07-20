# random_seed = 1
import numpy as np
# np.random.seed(seed=random_seed)
import torch
# torch.manual_seed(seed=random_seed)

# Use GPU when available
# Need to use Tensor to create the tensor of default type
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import numpy as np
import os
import copy
import dill
from ..util import plot_util
from ..util.random_util import resample

# from ..utils import save_train_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import collections

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

class Mine():
    def __init__(self, lr, batch_size, ma_rate, hidden_size=100, snapshot=[], iter_num=int(1e+3), model_name="MINE", log=True, prefix="", ground_truth=0, verbose=False, full_ref=False, load_dict=False, ref_factor=1, rep=1, fix_ref_est=False, archive_length=0, full_batch_ref=False, estimate_rate=1, video_rate=0, infinite_sample=False):
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
        self.full_ref = full_ref
        self.load_dict = load_dict
        self.ref_factor = ref_factor
        self.rep = rep
        self.fix_ref_est = fix_ref_est
        self.archive_length = archive_length
        self.full_batch_ref = full_batch_ref
        self.estimate_rate = estimate_rate
        self.video_rate = video_rate
        self.infinite_sample = infinite_sample

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
            log.write("ma_rate={0}\n".format(self.ma_rate))
            log.write("hidden_size={0}\n".format(self.hidden_size))
            log.write("snapshot={0}\n".format(self.snapshot))
            log.write("iter_num={0}\n".format(self.iter_num))
            log.write("model_name={0}\n".format(self.model_name))
            log.write("prefix={0}\n".format(self.prefix))
            log.write("ground_truth={0}\n".format(self.ground_truth))
            log.write("verbose={0}\n".format(self.verbose))
            log.write("full_ref={0}\n".format(self.full_ref))
            log.write("load_dict={0}\n".format(self.load_dict))
            log.write("ref_factor={0}\n".format(self.ref_factor))
            log.write("rep={0}\n".format(self.rep))
            log.write("fix_ref_est={0}\n".format(self.fix_ref_est))
            log.write("dim={0}\n".format(self.dim))
            log.write("sample_size={0}\n".format(self.sample_size))
            log.write("archive_length={0}\n".format(self.archive_length))
            log.write("full_batch_ref={0}\n".format(self.full_batch_ref))
            log.write("estimate_rate={0}\n".format(self.estimate_rate))
            log.write("video_rate={0}\n".format(self.video_rate))
            log.write("infinite_sample={0}\n".format(self.infinite_sample))
            log.close()

        self.ixy_list = []
        if self.video_rate>0:
            Xmax = self.Trainlist_X[0].max()
            Xmin = self.Trainlist_X[0].min()
            Ymax = self.Trainlist_Y[0].max()
            Ymin = self.Trainlist_Y[0].min()
            x = np.linspace(Xmin, Xmax, 300)
            y = np.linspace(Ymin, Ymax, 300)
            xs, ys = np.meshgrid(x,y)
            # mesh = torch.FloatTensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))
            mesh = torch.FloatTensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))
            self.ixy_list_shape = np.append(np.array(xs.shape), 0).tolist()
            # ixy_list = np.zeros(self.ixy_list_shape)
            for _ in range(self.rep):
                self.ixy_list.append(np.zeros(self.ixy_list_shape))

        self.XYlist_net = []
        self.XYlist_optimizer = []
        for i in range(self.rep):
            self.XYlist_net.append(MineNet(input_size=self.dim*2,hidden_size=self.hidden_size))
            self.XYlist_optimizer.append(optim.Adam(self.XYlist_net[i].parameters(),lr=self.lr))

        self.Train_dXY_list = np.zeros((self.rep, 0))
        self.Test_dXY_list = np.zeros((self.rep, 0))

        self.ma_ef = 1
        snapshot_i = 0
        # set starting iter_num
        start_i = 0
        self.array_start = 0
        self.Train_start_ma = 0
        self.Test_start_ma = 0
        fname = self.get_latest_cache_name()
        if self.load_dict and os.path.exists(fname):
            state_dict = torch.load(fname, map_location = "cuda" if torch.cuda.is_available() else "cpu")
            self.load_state_dict(state_dict)
            if self.verbose:
                print('results loaded from '+fname)

        self.XY_ref_t_log_size = float(np.log(self.sample_size*self.ref_factor))
        if self.full_ref:
            self.XY_ref_t_log_size = float(2*np.log(self.sample_size))
        self.XYlist_t = []
        self.XYlist_ref_t = []
        for i in range(self.rep):
            self.XYlist_t.append(torch.Tensor(np.concatenate((self.Trainlist_X[i],self.Trainlist_Y[i]),axis=1)))

            if self.full_ref:
                Train_X_ref, Train_Y_ref = np.meshgrid(self.Trainlist_X[i], self.Trainlist_Y[i].T)
                if self.dim==1:
                    Train_X_ref = Train_X_ref.flatten()[:,None]
                    Train_Y_ref = Train_Y_ref.flatten()[:,None]
                else:
                    Train_X_ref = Train_X_ref[:self.sample_size,:].reshape((self.sample_size**2), self.dim)
                    Train_Y_ref = Train_Y_ref[:,:self.sample_size].reshape(self.dim, (self.sample_size**2)).T
                self.XYlist_ref_t.append(torch.Tensor(np.concatenate((Train_X_ref,Train_Y_ref),axis=1)))
            elif self.fix_ref_est:
                if self.ref_factor > 1:
                    Train_X_ref = resample(self.Trainlist_X[i],batch_size=int(self.sample_size*self.ref_factor), replace=True)
                    Train_Y_ref = resample(self.Trainlist_Y[i],batch_size=int(self.sample_size*self.ref_factor), replace=True)
                else:
                    Train_X_ref = resample(self.Trainlist_X[i],batch_size=int(self.sample_size*self.ref_factor), replace=False)
                    Train_Y_ref = resample(self.Trainlist_Y[i],batch_size=int(self.sample_size*self.ref_factor), replace=False)
                self.XYlist_ref_t.append(torch.Tensor(np.concatenate((Train_X_ref,Train_Y_ref),axis=1)))

        if type(self.Train_dXY_list)==np.ndarray and self.Train_dXY_list.ndim == 2:
            start_i = len(self.Train_dXY_list[0,:]) + self.array_start
            for i in range(len(self.snapshot)):
                if self.snapshot[i] <= start_i:
                    snapshot_i = i+1
        for i in range(start_i, self.iter_num):
            if self.infinite_sample and i > 0:
                for _ in range(self.rep):
                    data_train = data_model.data
                    data_test = data_model.data
                    self.Trainlist_X.append(data_train[:,0:data_train.shape[1]//2].copy())
                    self.Trainlist_Y.append(data_train[:,-data_train.shape[1]//2:].copy())
                    self.Testlist_X.append(data_test[:,0:data_test.shape[1]//2].copy())
                    self.Testlist_Y.append(data_test[:,-data_test.shape[1]//2:].copy())
            self.update_mine_net(self.Trainlist_X, self.Trainlist_Y, self.batch_size, self.ma_rate)

            if (i+1)%self.estimate_rate==0:
                Train_dXY = self.get_estimate(self.Trainlist_X, self.Trainlist_Y)
                self.Train_dXY_list = np.append(self.Train_dXY_list, Train_dXY, axis=1)

                Test_dXY = self.get_estimate(self.Testlist_X, self.Testlist_Y)
                self.Test_dXY_list = np.append(self.Test_dXY_list, Test_dXY, axis=1)

            if self.video_rate>0 and (i+1)%self.video_rate==0:
                for j in range(self.rep):
                    ixy = self.XYlist_net[j](mesh).detach().numpy()
                    ixy = ixy.reshape(xs.shape[1], ys.shape[0])
                    self.ixy_list[j] = np.append(self.ixy_list[j], ixy[...,None], axis=2)

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

            if self.archive_length>0 and (i+1)%self.archive_length==0:
                self.save_array()

        if self.log:
            self.save_figure(suffix="iter={}".format(self.iter_num))
        # To save new results to a db file using the following code, delete the existing db file.
        fname = os.path.join(self.prefix, "cache_iter={}.pt".format(self.iter_num))
        if not os.path.exists(fname):
            with open(fname,'wb') as f:
                torch.save(self.state_dict(),f)
                if self.verbose:
                    print('results saved to '+fname)

    def update_mine_net(self, X, Y, batch_size, ma_rate):
        for i in range(self.rep):
            # XY_t = torch.Tensor(np.concatenate((X[i],Y[i]),axis=1))
            XY_t = np.concatenate((X[i],Y[i]),axis=1)
            batch_XY = resample(XY_t,batch_size=batch_size)

            if self.full_batch_ref:
                batch_X_ref, batch_Y_ref = np.meshgrid(batch_XY[:,0:self.dim], batch_XY[:,-self.dim:].T)
                if self.dim==1:
                    batch_X_ref = batch_X_ref.flatten()[:,None]
                    batch_Y_ref = batch_Y_ref.flatten()[:,None]
                else:
                    batch_X_ref = batch_X_ref[:batch_size,:].reshape((batch_size**2), self.dim)
                    batch_Y_ref = batch_Y_ref[:,:batch_size].reshape(self.dim, (batch_size**2)).T
                batch_XY_ref = torch.Tensor(np.concatenate((batch_X_ref,batch_Y_ref),axis=1))
            else:
                batch_XY_ref = torch.Tensor(np.concatenate((resample(X[i],batch_size=batch_size),                                                         resample(Y[i],batch_size=batch_size)),axis=1))

            batch_XY = torch.Tensor(batch_XY)
            self.XYlist_optimizer[i].zero_grad()
            fXY = self.XYlist_net[i](batch_XY)
            efXY_ref = torch.exp(self.XYlist_net[i](batch_XY_ref))
            self.ma_ef = (1-ma_rate)*self.ma_ef + ma_rate*torch.mean(efXY_ref)
            batch_dXY = -(torch.mean(fXY) - (1/self.ma_ef.mean()).detach()*torch.mean(efXY_ref))
            batch_dXY.backward()
            self.XYlist_optimizer[i].step()

    def get_estimate(self, X, Y):
        dXY_list = np.zeros((self.rep, 1))
        for i in range(self.rep):
            if self.full_ref or self.fix_ref_est:
                XY_ref_t = self.XYlist_ref_t[i]
            else:
                if self.ref_factor > 1:
                    X_ref = resample(X[i],batch_size=int(self.sample_size*self.ref_factor), replace=True)
                    Y_ref = resample(Y[i],batch_size=int(self.sample_size*self.ref_factor), replace=True)
                else:
                    X_ref = resample(X[i],batch_size=int(self.sample_size*self.ref_factor), replace=False)
                    Y_ref = resample(Y[i],batch_size=int(self.sample_size*self.ref_factor), replace=False)
                XY_ref_t = torch.Tensor(np.concatenate((X_ref,Y_ref),axis=1))

            XY_t = torch.Tensor(np.concatenate((X[i],Y[i]),axis=1))

            dXY = torch.mean(self.XYlist_net[i](XY_t)) - (torch.logsumexp(self.XYlist_net[i](XY_ref_t), 0) - self.XY_ref_t_log_size)
            dXY_list[i, 0] = dXY.cpu().item()
        return dXY_list

    def save_array(self):
        array_end = self.array_start + self.Train_dXY_list.shape[1]
        fpath = os.path.join(self.prefix, "archive")
        if not os.path.exists(fpath):
            os.mkdir(fpath)
        if self.video_rate>0:
            self.save_video(array_end)
        fname = os.path.join(fpath, "[{}-{}).pt".format(self.array_start, array_end))
        with open(fname, 'wb') as f:
            torch.save(self.array_state_dict(),f)
            if self.verbose:
                print("array archived")
            Train_ma = plot_util.Moving_average(self.Train_dXY_list, ma_rate=0.01, start=self.Train_start_ma)
            self.Train_start_ma = Train_ma[:,-1]
            Test_ma = plot_util.Moving_average(self.Test_dXY_list, ma_rate=0.01, start=self.Test_start_ma)
            self.Test_start_ma = Test_ma[:,-1]
            self.array_start = array_end
            self.Train_dXY_list = np.zeros((self.rep, 0))
            self.Test_dXY_list = np.zeros((self.rep, 0))
            if self.video_rate>0:
                for j in range(self.rep):
                    self.ixy_list[j] = np.zeros(self.ixy_list_shape)

    def load_all_array(self):
        fname = self.get_latest_cache_name()
        if self.load_dict and os.path.exists(fname):
            state_dict = torch.load(fname, map_location = "cuda" if torch.cuda.is_available() else "cpu")
            self.load_state_dict(state_dict)
        start = 0
        end = self.archive_length + start
        cache_array_start = self.array_start
        fname = os.path.join(self.prefix, "archive", "[{}-{}).pt".format(start, end))
        Train_dXY_list = np.zeros((self.rep, 0))
        Test_dXY_list = np.zeros((self.rep, 0))
        # if self.video_rate>0:
        #     ixy_list = []
        #     for _ in range(self.rep):
        #         ixy_list.append(np.zeros(self.ixy_list_shape))
        while(os.path.exists(fname) and end <= cache_array_start):
            state_dict = torch.load(fname, map_location = "cuda" if torch.cuda.is_available() else "cpu")
            start = self.archive_length + start
            end = self.archive_length + start
            fname = os.path.join(self.prefix, "archive", "[{}-{}).pt".format(start, end))
            Train_dXY_list = np.append(Train_dXY_list, state_dict['Train_dXY_list'], axis=1)
            Test_dXY_list = np.append(Test_dXY_list, state_dict['Test_dXY_list'], axis=1)
            # if self.video_rate>0 and 'ixy_list' in state_dict:
            #     for i in range(self.rep):
            #         ixy_list[i] = np.append(ixy_list[i], state_dict['ixy_list'][i], axis=2)

        # if self.video_rate>0 and ixy_list[0].shape[2]>=0:
        #     for i in range(self.rep):
        #         self.ixy_list[i] = np.append(ixy_list[i], self.ixy_list[i], axis=2)
        
        if self.Train_dXY_list.shape[1]>=0 and self.array_start==start:
            self.Train_dXY_list = np.append(Train_dXY_list, self.Train_dXY_list, axis=1)
        
        if self.Test_dXY_list.shape[1]>=0 and self.array_start==start:
            self.Test_dXY_list = np.append(Test_dXY_list, self.Test_dXY_list, axis=1)
        self.array_start = 0

    def predict(self, data_model):
        self.fit(data_model)

        if self.archive_length>0 and self.Train_dXY_list.shape[1]==0:
            mi_lb = np.average(self.Train_start_ma)
        else:
            mi_lb = np.average(self.Train_dXY_list[:,-1])

        self.Trainlist_X = []
        self.Trainlist_Y = []
        self.Testlist_X = []
        self.Testlist_Y = []

        self.XYlist_net = []
        self.XYlist_optimizer = []
        self.XYlist_ref_t = []

        self.Train_dXY_list = np.zeros((self.rep, 0))
        self.Test_dXY_list = np.zeros((self.rep, 0))
        return mi_lb

    def save_figure(self, suffix=""):
        fig, ax = plt.subplots(2,4, figsize=(90, 30))

        #plot mi_lb curve
        axCur = ax[0, 0]
        Train_mi_lb = self.Train_dXY_list
        axCur = plot_util.getTrainCurve(Train_mi_lb , [], axCur, show_min=False, ground_truth=self.ground_truth, start=self.array_start)
        axCur.set_title('curve of training data mutual information')

        #plot mi_lb curve
        axCur = ax[0, 1]
        Test_mi_lb = self.Test_dXY_list
        axCur = plot_util.getTrainCurve(Test_mi_lb , [], axCur, show_min=False, ground_truth=self.ground_truth, start=self.array_start)
        axCur.set_title('curve of testing data mutual information')

        # Trained Function contour plot
        if len(self.XYlist_net) == 1 and len(self.Trainlist_X) == 1 and self.Trainlist_X[0].shape[1] == 1 and self.Trainlist_Y[0].shape[1] == 1:
            Xmax = self.Trainlist_X[0].max()
            Xmin = self.Trainlist_X[0].min()
            Ymax = self.Trainlist_Y[0].max()
            Ymin = self.Trainlist_Y[0].min()
            x = np.linspace(Xmin, Xmax, 300)
            y = np.linspace(Ymin, Ymax, 300)
            xs, ys = np.meshgrid(x,y)
            # mesh = torch.FloatTensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))
            mesh = torch.FloatTensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))
            ixy = self.XYlist_net[0](mesh).detach().numpy()
            ixy = ixy.reshape(xs.shape[1], ys.shape[0])

            axCur = ax[0, 2]
            axCur, c = plot_util.getHeatMap(axCur, xs, ys, ixy)
            fig.colorbar(c, ax=axCur)
            axCur.set_title('heatmap of i(x,y)')

            axCur = ax[0, 3]
            axCur.scatter(self.Trainlist_X[0], self.Trainlist_Y[0], color='red', marker='o', label='train')
            axCur.scatter(self.Testlist_X[0], self.Testlist_Y[0], color='green', marker='x', label='test')
            axCur.set_title('Plot of all train data samples and test data samples')
            axCur.legend()
        else:
            #plot mi_lb curve
            axCur = ax[0, 2]
            Train_mi_lb = plot_util.Moving_average(self.Train_dXY_list, ma_rate=0.01, start=self.Train_start_ma)
            
            axCur = plot_util.getTrainCurve(Train_mi_lb , [], axCur, show_min=False, ground_truth=self.ground_truth, start=self.array_start)
            axCur.set_title('curve of training data mutual information with 0.01 ma')

            #plot mi_lb curve
            axCur = ax[0, 3]
            Test_mi_lb = plot_util.Moving_average(self.Test_dXY_list, ma_rate=0.01, start=self.Test_start_ma)
            axCur = plot_util.getTrainCurve(Test_mi_lb , [], axCur, show_min=False, ground_truth=self.ground_truth, start=self.array_start)
            axCur.set_title('curve of testing data mutual information with 0.01 ma')

            #plot mi_lb bias curve
            axCur = ax[1, 0]
            Train_bias = np.mean(Train_mi_lb, axis=0) - self.ground_truth
            Test_bias = np.mean(Test_mi_lb, axis=0) - self.ground_truth
            axCur = plot_util.getTrainCurve(Train_bias , Test_bias, axCur, show_min=False, start=self.array_start)
            axCur.set_ylabel("bias of mutual information estimate")
            axCur.set_xlabel("number of iteration step")
            axCur.set_title('curve of mutual information estimation bias')

            #plot mi_lb standard deviation curve
            axCur = ax[1, 1]
            Train_std = np.std(Train_mi_lb, axis=0)
            Test_std = np.std(Test_mi_lb, axis=0)
            axCur = plot_util.getTrainCurve(Train_std, Test_std, axCur, show_min=False, start=self.array_start)
            axCur.set_ylabel("std of mutual information estimate")
            axCur.set_xlabel("number of iteration step")
            axCur.set_title('curve of mutual information estimation standard deviation')


        figName = os.path.join(self.prefix, "{}_{}".format(self.model_name, suffix))
        fig.savefig(figName, bbox_inches='tight')
        plt.close()

    def state_dict(self):
        return {
            'Train_dXY_list' : self.Train_dXY_list,
            'Test_dXY_list' : self.Test_dXY_list,
            'XYlist_net': [XY_net.state_dict() for XY_net in self.XYlist_net],
            'XYlist_optimizer': [XY_optim.state_dict() for XY_optim in self.XYlist_optimizer],
            # 'XYlist_net': self.XYlist_net.state_dict(),
            # 'XYlist_optimizer': self.XYlist_optimizer.state_dict(),
            'Trainlist_X': self.Trainlist_X,
            'Trainlist_Y': self.Trainlist_Y,
            'Testlist_X': self.Testlist_X,
            'Testlist_Y': self.Testlist_Y,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'ma_rate': self.ma_rate,
            'ma_ef': self.ma_ef,
            'array_start': self.array_start,
            'Train_start_ma': self.Train_start_ma,
            'Test_start_ma': self.Test_start_ma,
            'infinite_sample': self.infinite_sample,
            # 'ixy_list': self.ixy_list
        }

    def load_state_dict(self, state_dict):
        if 'XYlist_net' in state_dict:
            if collections.OrderedDict == type(state_dict['XYlist_net'][0]):
                for k in range(len(state_dict['XYlist_net'])):
                    self.XYlist_net[k].load_state_dict(state_dict['XYlist_net'][k])
            else:
                self.XYlist_net = state_dict['XYlist_net']
        if 'XYlist_optimizer' in state_dict:
            if collections.OrderedDict == type(state_dict['XYlist_optimizer'][0]):
                for k in range(len(state_dict['XYlist_optimizer'])):
                    self.XYlist_optimizer[k].load_state_dict(state_dict['XYlist_optimizer'][k])
            else:
                self.XYlist_optimizer = state_dict['XYlist_optimizer']
        # self.XYlist_net = state_dict['XYlist_net']
        # self.XYlist_optimizer = state_dict['XYlist_optimizer']
        self.Trainlist_X = state_dict['Trainlist_X']
        self.Trainlist_Y = state_dict['Trainlist_Y']
        self.Testlist_X = state_dict['Testlist_X']
        self.Testlist_Y = state_dict['Testlist_Y']
        self.lr = state_dict['lr']
        self.batch_size = state_dict['batch_size']
        self.ma_rate = state_dict['ma_rate']
        self.ma_ef = state_dict['ma_ef']
        self.Test_dXY_list = state_dict['Test_dXY_list']
        self.Train_dXY_list = state_dict['Train_dXY_list']
        if 'array_start' in state_dict:
            self.array_start = state_dict['array_start']
        if 'Test_start_ma' in state_dict:
            self.Test_start_ma = state_dict['Test_start_ma']
        if 'Train_start_ma' in state_dict:
            self.Train_start_ma = state_dict['Train_start_ma']
        # if 'ixy_list' in state_dict:
        #     self.ixy_list = state_dict['ixy_list']
        if 'video_rate' in state_dict:
            self.video_rate = state_dict['video_rate']
        if 'infinite_sample' in state_dict:
            self.infinite_sample = state_dict['infinite_sample']

    def array_state_dict(self):
        return {
            'Train_dXY_list' : self.Train_dXY_list,
            'Test_dXY_list' : self.Test_dXY_list,
            'Train_start_ma': self.Train_start_ma,
            'Test_start_ma': self.Test_start_ma,
            # 'ixy_list': self.ixy_list
        }

    def get_latest_cache_name(self):
        fname = os.path.join(self.prefix, "cache_iter={}.pt".format(self.iter_num))
        if not os.path.exists(fname):
            for i in range(len(self.snapshot)):
                cur_snapshot = self.snapshot[-i-1]
                fname = os.path.join(self.prefix, "cache_iter={}.pt".format(cur_snapshot))
                if os.path.exists(fname):
                    break
                if i == len(self.snapshot)-1:
                    fname=os.path.join(self.prefix, "cache.pt")
        return fname

    def save_video(self, iter):
        # if (i+1)%self.video_rate==0:
        Xmax = self.Trainlist_X[0].max()
        Xmin = self.Trainlist_X[0].min()
        Ymax = self.Trainlist_Y[0].max()
        Ymin = self.Trainlist_Y[0].min()
        x = np.linspace(Xmin, Xmax, 300)
        y = np.linspace(Ymin, Ymax, 300)
        xs, ys = np.meshgrid(x,y)
        for j in range(self.rep):
            heatmap_animation_fig, heatmap_animation_ax = plt.subplots(1, 1)
            axCur = heatmap_animation_ax
            cax = axCur.pcolormesh(xs, ys, self.ixy_list[j][:-1,:-1,0], cmap='RdBu', vmin=self.ixy_list[j][:-1,:-1,0].min(), vmax=self.ixy_list[j][:-1,:-1,0].max())
            heatmap_animation_fig.colorbar(cax)

            def animate(i):
                cax.set_array(self.ixy_list[j][:-1,:-1,i].flatten())
                cax.autoscale()

            writer = animation.writers['ffmpeg'](fps=1, bitrate=1800)
            heatmap_animation = animation.FuncAnimation(heatmap_animation_fig, animate, interval=200, blit=False, frames=self.ixy_list[j].shape[2])
            heatmap_animation.save(os.path.join(self.prefix, "archive", "heatmap_net={}_iter={}.mp4".format(j, iter)), writer=writer)
            plt.close()