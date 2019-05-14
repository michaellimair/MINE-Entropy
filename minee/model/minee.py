import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from .pytorchtools import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
# from .DiscreteCondEnt import subset
import os
from ..util import plot_util
from ..util import torch_util

# from ..utils import save_train_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def sample_batch(data, resp=[1], cond=[0], batch_size=100, sample_mode='marginal', ref_bound_factor=1.0):
    """[summary]
    
    Arguments:
        data {[type]} -- [N X 2]
        resp {[list]} -- [description]
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
        if type(resp)==list:
            for i in resp:
                whole.append(i)
        else:
            raise TypeError("resp should be list")
    else:
        raise TypeError("cond should be list")
    if sample_mode == 'joint':
        index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[index]
        batch = batch[:, whole]
    elif sample_mode == 'unif':
        dataMax = data.max(axis=0)[whole]
        dataMin = data.min(axis=0)[whole]
        if ref_bound_factor > 1.0:
            data_rad = (dataMax - dataMin)/2
            data_mean = (dataMax + dataMin)/2
            dataMax = data_mean + ref_bound_factor * data_rad
            dataMin = data_mean - ref_bound_factor * data_rad
        batch = (dataMax - dataMin)*np.random.random((batch_size,len(whole))) + dataMin
    elif sample_mode == 'marginal':
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        if type(resp)==list:
            data_resp = data[marginal_index][:,resp].reshape(-1,len(resp))
        batch = np.concatenate([data[joint_index][:,cond].reshape(-1,len(cond)), data_resp], axis=1)
    else:
        raise ValueError('Sample mode: {} not recognized.'.format(sample_mode))
    return batch

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
    def __init__(self, lr, batch_size, patience=int(20), iter_num=int(1e+3), log_freq=int(100), avg_freq=int(10), ma_rate=0.01, verbose=True, resp=[1], cond=[0], log=True, sample_mode='unif', y_label="", earlyStop=True, iter_snapshot=[], hidden_size=100, ref_bound_factor=1.0, video_frames=int(1e3), ref_ratio=1):
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience  # 20
        self.iter_num = iter_num  # 1e+3
        self.log_freq = int(log_freq)  # int(1e+2)
        self.avg_freq = avg_freq  # int(1e+1)
        self.ma_rate = ma_rate  # 0.01
        self.prefix = ''
        self.verbose = verbose
        self.resp = resp
        self.cond = cond
        self.log = log
        self.sample_mode = sample_mode 
        self.model_name = ""
        self.ground_truth = None
        self.paramName = None
        if sample_mode == "marginal":
            self.y_label = "I(X^Y)"
        elif sample_mode == "unif":
            self.y_label = "HXY"
        else:
            self.y_label = y_label
        self.heatmap_frames = []  # for plotting heatmap animation
        self.mine_net_XY = MineNet(input_size=len(self.cond)+len(self.resp),hidden_size=hidden_size)
        self.mine_net_optim_XY = optim.Adam(self.mine_net_XY.parameters(), lr=self.lr)
        self.mine_net_X = MineNet(input_size=len(self.cond),hidden_size=hidden_size)
        self.mine_net_optim_X = optim.Adam(self.mine_net_X.parameters(), lr=self.lr)
        self.mine_net_Y = MineNet(input_size=len(self.resp),hidden_size=hidden_size)
        self.mine_net_optim_Y = optim.Adam(self.mine_net_Y.parameters(), lr=self.lr)
        self.earlyStop = earlyStop
        self.iter_snapshot = iter_snapshot
        self.ref_bound_factor = ref_bound_factor
        self.ref_size = int(batch_size*ref_ratio)
        self.video_frames = video_frames

    def fit(self, train_data, val_data):
        self.Xmin = min(train_data[:,0])
        self.Xmax = max(train_data[:,0])
        self.Ymin = min(train_data[:,1])
        self.Ymax = max(train_data[:,1])
        if self.ref_bound_factor > 1.0:
            X_mean = (self.Xmax + self.Xmin)/2
            X_rad = (self.Xmax - self.Xmin)/2
            self.Xmin = X_mean - self.ref_bound_factor*X_rad
            self.Xmax = X_mean + self.ref_bound_factor*X_rad
            Y_mean = (self.Ymax + self.Ymin)/2
            Y_rad = (self.Ymax - self.Ymin)/2
            self.Ymin = Y_mean - self.ref_bound_factor*Y_rad
            self.Ymax = Y_mean + self.ref_bound_factor*Y_rad
    
        if self.log:
            log_file = os.path.join(self.prefix, "MINE_train.log")
            log = open(log_file, "w")
            log.write("batch_size={0}\n".format(self.batch_size))
            log.write("iter_num={0}\n".format(self.iter_num))
            log.write("log_freq={0}\n".format(self.log_freq))
            log.write("avg_freq={0}\n".format(self.avg_freq))
            log.write("patience={0}\n".format(self.patience))
            log.write("iter_snapshot={0}\n".format(self.iter_snapshot))
            log.write("lr={0}\n".format(self.lr))
            # log.write("self.mine_net={0}\n".format(type(self.mine_net)))
            log.write("ma_rate={0}\n".format(self.ma_rate))
            log.write("video_frames={0}\n".format(self.video_frames))
            log.close()
            heatmap_animation_fig, heatmap_animation_ax = plt.subplots(1, 1)
        # data is x or y
        self.ma_efx = 1.  # exponential of mi estimation on marginal data
        self.ma_efy = 1. 
        self.ma_efxy = 1. 
        
        #Early Stopping
        train_mi_lb = []
        self.avg_train_mi_lb = []

        train_loss_XY = []
        self.avg_train_loss_XY = []
        train_loss_X = []
        self.avg_train_loss_X = []
        train_loss_Y = []
        self.avg_train_loss_Y = []

        self.reference_dist = sample_batch(train_data, resp= self.resp, cond= self.cond, batch_size=train_data.shape[0], sample_mode=self.sample_mode, ref_bound_factor=self.ref_bound_factor)
        j = 0
        for i in range(self.iter_num):
            #get train data
            # batchTrain = sample_batch(train_data, resp= self.resp, cond= self.cond, batch_size=self.batch_size, sample_mode='joint'), \
            #              sample_batch(train_data, resp= self.resp, cond= self.cond, batch_size=self.ref_size, sample_mode=self.sample_mode, ref_bound_factor=self.ref_bound_factor)
            self.update_mine_net(train_data, ma_rate=self.ma_rate)

            mi_lbTrain, lossTrain_XY, lossTrain_X, lossTrain_Y = self.forward_pass(self.X_train)
            train_loss_XY.append(lossTrain_XY)
            train_loss_X.append(lossTrain_X)
            train_loss_Y.append(lossTrain_Y)
            train_mi_lb.append(mi_lbTrain)
            
            if self.avg_freq==1:
                self.avg_train_loss_XY.append(lossTrain_XY)
                self.avg_train_loss_X.append(lossTrain_X)
                self.avg_train_loss_Y.append(lossTrain_Y)
                self.avg_train_mi_lb.append(mi_lbTrain)
            elif (i+1)%(self.avg_freq)==0:
                self.avg_train_loss_XY.append(np.average(train_loss_XY))
                self.avg_train_loss_X.append(np.average(train_loss_X))
                self.avg_train_loss_Y.append(np.average(train_loss_Y))
                self.avg_train_mi_lb.append(np.average(train_mi_lb))
                train_loss_XY= []
                train_loss_X= []
                train_loss_Y= []
                train_mi_lb = []

            if len(self.iter_snapshot)>j and (i+1)%self.iter_snapshot[j]==0:
                self.savefig(suffix="_iter={}".format(self.iter_snapshot[j]))
                
                ch = "checkpoint_XY_iter={}.pt".format(self.iter_snapshot[j])
                ch = os.path.join(self.prefix, ch)
                torch.save(self.mine_net_XY.state_dict(), ch)
                ch = "checkpoint_X_iter={}.pt".format(self.iter_snapshot[j])
                ch = os.path.join(self.prefix, ch)
                torch.save(self.mine_net_X.state_dict(), ch)
                ch = "checkpoint_Y_iter={}.pt".format(self.iter_snapshot[j])
                ch = os.path.join(self.prefix, ch)
                torch.save(self.mine_net_Y.state_dict(), ch)
                j += 1
            if self.video_frames>0:
                x = np.linspace(self.Xmin, self.Xmax, 300)
                y = np.linspace(self.Ymin, self.Ymax, 300)
                xs, ys = np.meshgrid(x,y)
                # mesh = torch.FloatTensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))
                mesh = torch.FloatTensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))
                fxy = self.mine_net_XY(mesh)
                fx = self.mine_net_X(mesh[:,self.cond])
                fy = self.mine_net_Y(mesh[:,self.resp])
                ixy = (fxy - fx - fy).detach().numpy()
                ixy = ixy.reshape(xs.shape[1], ys.shape[0])
                # ixy = t - np.log(self.ma_et.mean().detach().numpy())
                heatmap_animation_ax, c = plot_util.getHeatMap(heatmap_animation_ax, xs, ys, ixy)
                self.heatmap_frames.append((c,))
                if (i+1)%self.video_frames==0:
                    writer = animation.writers['ffmpeg'](fps=1, bitrate=1800)
                    heatmap_animation = animation.ArtistAnimation(heatmap_animation_fig, self.heatmap_frames, interval=200, blit=False)
                    heatmap_animation.save(os.path.join(self.prefix, "heatmap_less_than_{}.mp4".format(i+1)), writer=writer)
                    self.heatmap_frames=[]
    
        if self.log:
            #Save result to files
            avg_train_loss_XY = np.array(self.avg_train_loss_XY )
            np.savetxt(os.path.join(self.prefix, "avg_train_loss_XY.txt"), avg_train_loss_XY )
            avg_train_loss_X = np.array(self.avg_train_loss_X )
            np.savetxt(os.path.join(self.prefix, "avg_train_loss_X.txt"), avg_train_loss_X )
            avg_train_loss_Y = np.array(self.avg_train_loss_Y )
            np.savetxt(os.path.join(self.prefix, "avg_train_loss_Y.txt"), avg_train_loss_Y )
            avg_train_mi_lb = np.array(self.avg_train_mi_lb)
            np.savetxt(os.path.join(self.prefix, "avg_train_mi_lb.txt"), avg_train_mi_lb)

            # train_batch_mi_lb = np.array(train_batch_mi_lb)
            # np.savetxt(os.path.join(self.prefix, "train_batch_mi_lb.txt"), train_batch_mi_lb)
            # train_batch_mi_lb = np.array(train_batch_mi_lb)
            # np.savetxt(os.path.join(self.prefix, "train_batch_mi_lb.txt"), train_batch_mi_lb)

    
    def update_mine_net(self, train_data, ma_rate=0.01):
        """[summary]
        
        Arguments:
            batch {[type]} -- ([batch_size X 2], [batch_size X 2])
            ma_rate {float} -- [moving average rate] (default: {0.01})
        
        Keyword Arguments:
            mi_lb {} -- []
        """
        joint = sample_batch(train_data, resp= self.resp, cond= self.cond, batch_size=self.batch_size, sample_mode='joint')
        # reference = sample_batch(train_data, resp= self.resp, cond= self.cond, batch_size=self.ref_size, sample_mode=self.sample_mode, ref_bound_factor=self.ref_bound_factor)

        joint_x = sample_batch(train_data, resp= [], cond= self.cond, batch_size=self.batch_size, sample_mode='joint')
        reference_x = sample_batch(train_data, resp= [], cond= self.cond, batch_size=self.ref_size, sample_mode=self.sample_mode, ref_bound_factor=self.ref_bound_factor)
        # reference_x = reference[:,self.cond]

        joint_y = sample_batch(train_data, resp= self.resp, cond= [], batch_size=self.batch_size, sample_mode='joint')
        reference_y = sample_batch(train_data, resp= self.resp, cond= [], batch_size=self.ref_size, sample_mode=self.sample_mode, ref_bound_factor=self.ref_bound_factor)
        # reference_y = reference[:,self.resp]

        reference = np.concatenate((reference_x, reference_y), axis=1)
        self.data_joint, self.data_mar = joint, reference
        # batch is a tuple of (joint, marginal)
        # joint , reference = batch
        # joint = torch.autograd.Variable(torch.FloatTensor(joint))
        # reference = torch.autograd.Variable(torch.FloatTensor(reference))
        joint = torch.FloatTensor(joint)
        reference = torch.FloatTensor(reference)
        joint_x = torch.FloatTensor(joint_x)
        reference_x = torch.FloatTensor(reference_x)
        joint_y = torch.FloatTensor(joint_y)
        reference_y = torch.FloatTensor(reference_y)
        # mi_lb, fx, fy, fxy, efx, efy, efxy = self.mutual_information(joint, reference)
        fxy = self.mine_net_XY(joint)
        # fx = self.mine_net_X(joint[:,self.cond])
        # fy = self.mine_net_Y(joint[:,self.resp])
        fx = self.mine_net_X(joint_x)
        fy = self.mine_net_Y(joint_y)
        fxy_ref = self.mine_net_XY(reference)
        # fx_ref = self.mine_net_X(reference[:,self.cond])
        # fy_ref = self.mine_net_Y(reference[:,self.resp])
        fx_ref = self.mine_net_X(reference_x)
        fy_ref = self.mine_net_Y(reference_y)
        efx, efy, efxy = torch.exp(fx_ref), torch.exp(fy_ref), torch.exp(fxy_ref)

        # self.ma_efx = ((1-ma_rate)*self.ma_efx + ma_rate*torch.mean(efx)).item()
        # self.ma_efy = ((1-ma_rate)*self.ma_efy + ma_rate*torch.mean(efy)).item()
        # self.ma_efxy = ((1-ma_rate)*self.ma_efxy + ma_rate*torch.mean(efxy)).item()
        
        # unbiasing use moving average
        # loss_X = -(torch.mean(fx) - (1/self.ma_efx)*torch.mean(efx))
        # loss_Y = -(torch.mean(fy) - (1/self.ma_efy)*torch.mean(efy))
        # loss_XY = -(torch.mean(fxy) - (1/self.ma_efxy)*torch.mean(efxy))
        loss_X = -(torch.mean(fx) - torch.log(torch.mean(efx)))
        loss_Y = -(torch.mean(fy) - torch.log(torch.mean(efy)))
        loss_XY = -(torch.mean(fxy) - torch.log(torch.mean(efxy)))
        self.mine_net_optim_XY.zero_grad()
        # autograd.backward(loss_XY)
        loss_XY.backward()
        self.mine_net_optim_XY.step()

        self.mine_net_optim_X.zero_grad()
        # autograd.backward(loss_X)
        loss_X.backward()
        self.mine_net_optim_X.step()

        self.mine_net_optim_Y.zero_grad()
        # autograd.backward(loss_Y)
        loss_Y.backward()
        self.mine_net_optim_Y.step()

        # data_xy = torch.FloatTensor(train_data)
        # d_x = - (torch.mean(data_xy[:,self.cond]) - torch.log(torch.mean(efx)))
        # d_y = - (torch.mean(data_xy[:,self.resp]) - torch.log(torch.mean(efy)))
        # d_xy = - (torch.mean(data_xy) - torch.log(torch.mean(efxy)))
        # mi_lb = d_x + d_y - d_xy
        # return mi_lb.item(), d_xy.item(), d_x.item(), d_y.item()

    def mutual_information(self, joint, reference):
        fxy = self.mine_net_XY(joint)
        fx = self.mine_net_X(joint[:,self.cond])
        fy = self.mine_net_Y(joint[:,self.resp])
        fxy_ref = self.mine_net_XY(reference)
        fx_ref = self.mine_net_X(reference[:,self.cond])
        fy_ref = self.mine_net_Y(reference[:,self.resp])
        efx_ref, efy_ref, efxy_ref = torch.exp(fx_ref), torch.exp(fy_ref), torch.exp(fxy_ref)
        if self.sample_mode == 'unif':
            d_x = - (torch.mean(fx) - torch.log(torch.mean(efx_ref)))
            d_y = - (torch.mean(fy) - torch.log(torch.mean(efy_ref)))
            d_xy = - (torch.mean(fxy) - torch.log(torch.mean(efxy_ref)))
            mi_lb = d_x + d_y - d_xy
        else:
            raise ValueError('sample mode: {} not supported yet.'.format(self.sample_mode))
        return mi_lb, fx, fy, fxy, efx_ref, efy_ref, efxy_ref

    def forward_pass(self, X):
        # joint = sample_batch(X, resp= self.resp, cond= self.cond, batch_size=X.shape[0], sample_mode='joint')
        joint = X
        # reference = sample_batch(X, resp= self.resp, cond= self.cond, batch_size=self.ref_size, sample_mode=self.sample_mode, ref_bound_factor=self.ref_bound_factor)
        reference = self.reference_dist
        # joint = torch.autograd.Variable(torch.FloatTensor(joint))
        # reference = torch.autograd.Variable(torch.FloatTensor(reference))
        joint = torch.FloatTensor(joint)
        reference = torch.FloatTensor(reference)
        # mi_lb, fx, fy, fxy, efx, efy, efxy = self.mutual_information(joint, reference)
        fxy = self.mine_net_XY(joint)
        fx = self.mine_net_X(joint[:,self.cond])
        fy = self.mine_net_Y(joint[:,self.resp])
        fxy_ref = self.mine_net_XY(reference)
        fx_ref = self.mine_net_X(reference[:,self.cond])
        fy_ref = self.mine_net_Y(reference[:,self.resp])
        efx, efy, efxy = torch.exp(fx_ref), torch.exp(fy_ref), torch.exp(fxy_ref)
        if self.sample_mode == 'unif':
            d_x = (torch.mean(fx) - torch.log(torch.mean(efx)))
            d_y = (torch.mean(fy) - torch.log(torch.mean(efy)))
            d_xy = (torch.mean(fxy) - torch.log(torch.mean(efxy)))
            mi_lb = d_xy - d_y - d_x

        # loss_X = -(torch.mean(fx) - (1/self.ma_efx)*torch.mean(efx))
        # loss_Y = -(torch.mean(fy) - (1/self.ma_efy)*torch.mean(efy))
        # loss_XY = -(torch.mean(fxy) - (1/self.ma_efxy)*torch.mean(efxy))

        return mi_lb.item(), d_xy.item(), d_x.item(), d_y.item()

    def predict(self, X_train, X_test):
        """[summary]
        
        Arguments:
            X {[numpy array]} -- [N X 2]

        Return:
            mutual information estimate
        """
        self.X_train, self.X_test = X_train, X_test
        X_train = np.array(self.X_train)
        np.savetxt(os.path.join(self.prefix, "X_train.txt"), X_train)
        X_test = np.array(self.X_test)
        np.savetxt(os.path.join(self.prefix, "X_test.txt"), X_test)
        self.fit(self.X_train, self.X_test)
    
        mi_lb, _, _, _ = self.forward_pass(self.X_test)

        if self.log:
            self.savefig(suffix="_iter={}".format(self.iter_num))
            ch = "checkpoint_XY_iter={}.pt".format(self.iter_num)
            ch = os.path.join(self.prefix, ch)
            torch.save(self.mine_net_XY.state_dict(), ch)
            ch = "checkpoint_X_iter={}.pt".format(self.iter_num)
            ch = os.path.join(self.prefix, ch)
            torch.save(self.mine_net_X.state_dict(), ch)
            ch = "checkpoint_Y_iter={}.pt".format(self.iter_num)
            ch = os.path.join(self.prefix, ch)
            torch.save(self.mine_net_Y.state_dict(), ch)

        #release memory
        self.heatmap_frames=[]
        self.avg_train_loss=[]
        self.avg_train_mi_lb=[]
        # self.avg_valid_loss=[]
        # self.avg_valid_mi_lb=[]
        self.X_test=[]
        self.X_train=[]
        return mi_lb

    def savefig(self, suffix=""):
        if len(self.cond) > 1:
            raise ValueError("Only support 2-dim or 1-dim")
        # fig, ax = plt.subplots(3,4, figsize=(100, 45))
        fig, ax = plt.subplots(2,4, figsize=(90, 30))
        #plot Data
        axCur = ax[0,0]
        # axCur.scatter(self.X_train[:,self.resp], self.X_train[:,self.cond], color='red', marker='o', label='train')
        # axCur.scatter(self.X_test[:,self.resp], self.X_test[:,self.cond], color='green', marker='x', label='test')
        # axCur.legend()
        # axCur.set_title('scatter plot of data')
        axCur.scatter(self.data_joint[:,self.cond], self.data_joint[:,self.resp], color='red', marker='o', label='joint')
        axCur.scatter(self.data_mar[:,self.cond], self.data_mar[:,self.resp], color='green', marker='x', label='marginal')
        axCur.set_title('Plot of batch data samples and reference samples')
        axCur.legend()

        #plot training curve
        axCur = ax[0,1]
        # axCur = plot_util.getTrainCurve(self.avg_train_loss , self.avg_valid_loss, axCur, show_min=False)
        # axCur = plot_util.getTrainCurve(self.avg_train_loss , [], axCur, show_min=False, ground_truth=self.ground_truth)
        # axCur.set_title('train curve of total loss')
        axCur.plot(self.avg_train_loss_XY, label='XY')
        axCur.plot(self.avg_train_loss_X, label='X')
        axCur.plot(self.avg_train_loss_Y, label='Y')
        axCur.legend()
        axCur.set_xlabel("number of iterations")
        axCur.set_ylabel('divergence estimates')
        axCur.set_title('divergence estimates')

        # Trained Function contour plot
        x = np.linspace(self.Xmin, self.Xmax, 300)
        y = np.linspace(self.Ymin, self.Ymax, 300)
        xs, ys = np.meshgrid(x,y)
        # mesh = torch.FloatTensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))
        mesh = torch.FloatTensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))
        fxy = self.mine_net_XY(mesh)
        fx = self.mine_net_X(mesh[:,self.cond])
        fy = self.mine_net_Y(mesh[:,self.resp])
        ixy = (fxy - fx - fy).detach().numpy()
        ixy = ixy.reshape(xs.shape[1], ys.shape[0])

        axCur = ax[0,2]
        axCur, c = plot_util.getHeatMap(axCur, xs, ys, ixy)
        fig.colorbar(c, ax=axCur)
        axCur.set_title('heatmap of i(x,y)')

        fxy = fxy.detach().numpy().reshape(xs.shape[1], ys.shape[0])
        axCur = ax[0,3]
        axCur, c = plot_util.getHeatMap(axCur, xs, ys, fxy)
        fig.colorbar(c, ax=axCur)
        axCur.set_title('heatmap T(X,Y) for learning H(X,Y)')

        # axCur = ax[0,3]
        # axCur, _, c = super(Mine_ent, self).getHeatMap(axCur, xs, ys, Z=HXY)
        # fig.colorbar(c, ax=axCur)
        # axCur.set_title('heatmap H(X,Y)')

        # axCur = ax[1,2]
        # fx = self.Mine_resp.mine_net(torch.FloatTensor(x[:,None])).detach().numpy().flatten()
        # axCur = plot_util.getResultPlot(axCur, x, fx)
        # axCur.set_title('plot of T(X)')

        # axCur = ax[1,3]
        # axCur, _ = self.Mine_resp.getResultPlot(axCur, x, Z=HX)
        # axCur.set_title('plot of H(X)')

        # axCur = ax[2,2]
        # fy = self.Mine_cond.mine_net(torch.FloatTensor(y[:,None])).detach().numpy().flatten()
        # axCur = plot_util.getResultPlot(axCur, y, fy)
        # axCur.set_title('plot of T(Y)')

        # axCur = ax[2,3]
        # axCur, _ = self.Mine_resp.getResultPlot(axCur, y, Z=HY)
        # axCur.set_title('plot of H(Y)')
        # axCur = ax[1,0]
        # fx = fx[:-1]
        # fy = fy[:-1]
        # i_xy = [fxy[i,j]-fx[i]-fy[j] for i in range(fx.shape[0]) for j in range(fy.shape[0])]
        # i_xy = np.array(i_xy).reshape(fx.shape[0], fy.shape[0])
        # axCur, c = plot_util.getHeatMap(axCur, xs, ys, i_xy)
        # fig.colorbar(c, ax=axCur)
        # axCur.set_title('heatmap of i_xy')


        # Plot result with ground truth
        axCur = ax[1,0]
        ml_lb_train, _, _, _ = self.forward_pass(self.X_train)
        axCur.scatter(0, self.ground_truth, edgecolors='red', facecolors='none', label='Ground Truth')
        # axCur.scatter(0, ml_lb_estimate, edgecolors='green', facecolors='none', label="{}_Test".format(self.model_name))
        axCur.scatter(0, ml_lb_train, edgecolors='blue', facecolors='none', label="{}_Train".format(self.model_name))
        # axCur.scatter(0, ml_lb_estimate, edgecolors='green', facecolors='none', label="MINE_{0}".format(self.model_name))
        axCur.set_xlabel(self.paramName)
        axCur.set_ylabel(self.y_label)
        axCur.legend()
        axCur.set_title('MI of XY')

        #plot mi_lb curve
        axCur = ax[1,1]
        # axCur = plot_util.getTrainCurve(self.avg_train_mi_lb , self.avg_valid_mi_lb, axCur, show_min=False)
        axCur = plot_util.getTrainCurve(self.avg_train_mi_lb , [], axCur, show_min=False, ground_truth=self.ground_truth)
        axCur.set_title('training curve of mutual information')

        axCur = ax[1,2]
        # axCur.scatter(self.X_train[:,self.resp], self.X_train[:,self.cond], color='red', marker='o', label='train')
        # axCur.scatter(self.X_test[:,self.resp], self.X_test[:,self.cond], color='green', marker='x', label='test')
        # axCur.legend()
        # axCur.set_title('scatter plot of data')
        axCur.scatter(self.X_train[:,self.cond], self.X_train[:,self.resp], color='red', marker='o', label='data')
        axCur.scatter(self.reference_dist[:,self.cond], self.reference_dist[:,self.resp], color='green', marker='x', label='ref')
        axCur.set_title('Plot of all data samples and reference samples')
        axCur.legend()

        figName = os.path.join(self.prefix, "MINE{}".format(suffix))
        fig.savefig(figName, bbox_inches='tight')
        plt.close()

