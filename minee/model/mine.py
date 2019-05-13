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

# from ..utils import save_train_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def save_train_curve(train_loss, valid_loss, figName):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(figName, bbox_inches='tight')
    plt.close()


def sample_batch(data, resp=1, cond=[0], batch_size=100, sample_mode='marginal'):
    """[summary]
    
    Arguments:
        data {[type]} -- [N X 2]
        resp {[int or list]} -- [1 dimension]
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
        elif type(resp)==int:
            whole.append(resp)
        else:
            raise TypeError("resp should be list or int")
    else:
        raise TypeError("cond should be list")
    if sample_mode == 'joint':
        index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[index]
        batch = batch[:, whole]
    elif sample_mode == 'unif':
        dataMax = data.max(axis=0)[whole]
        dataMin = data.min(axis=0)[whole]
        batch = (dataMax - dataMin)*np.random.random((batch_size,len(whole))) + dataMin
    elif sample_mode == 'marginal':
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        if type(resp)==list:
            data_resp = data[marginal_index][:,resp].reshape(-1,len(resp))
        elif type(resp)==int:
            data_resp = data[marginal_index][:,resp].reshape(-1,1)
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

class Mine():
    def __init__(self, lr, batch_size, patience=int(20), iter_num=int(1e+3), log_freq=int(100), avg_freq=int(10), ma_rate=0.01, verbose=True, resp=1, cond=[0], log=True, sample_mode='marginal', y_label="", earlyStop=True, iter_snapshot=[], hidden_size=100, video_frames=int(1e3)):
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
        self.mine_net = MineNet(input_size=len(self.cond)+1, hidden_size=hidden_size)
        self.mine_net_optim = optim.Adam(self.mine_net.parameters(), lr=self.lr)
        self.earlyStop = earlyStop
        self.iter_snapshot = iter_snapshot
        self.video_frames = video_frames

    def fit(self, train_data, val_data):
        self.Xmin = min(train_data[:,0])
        self.Xmax = max(train_data[:,0])
        self.Ymin = min(train_data[:,1])
        self.Ymax = max(train_data[:,1])
    
        if self.log:
            log_file = os.path.join(self.prefix, "MINE_train.log")
            log = open(log_file, "w")
            log.write("batch_size={0}\n".format(self.batch_size))
            log.write("iter_num={0}\n".format(self.iter_num))
            log.write("log_freq={0}\n".format(self.log_freq))
            log.write("avg_freq={0}\n".format(self.avg_freq))
            log.write("patience={0}\n".format(self.patience))
            log.write("iter_snapshot={0}\n".format(self.iter_snapshot))
            log.write("ma_rate={0}\n".format(self.ma_rate))
            log.write("video_frames={0}\n".format(self.video_frames))
            log.close()
            heatmap_animation_fig, heatmap_animation_ax = plt.subplots(1, 1)
        # data is x or y
        self.ma_et = 1.  # exponential of mi estimation on marginal data
        
        #Early Stopping
        train_mi_lb = []
        valid_mi_lb = []
        self.avg_train_mi_lb = []
        self.avg_valid_mi_lb = []
        train_batch_mi_lb = []
        train_batch_loss = []
        
        if self.earlyStop:
            earlyStop = EarlyStopping(patience=self.patience, verbose=self.verbose, prefix=self.prefix)
        j = 0
        batchTrain = sample_batch(train_data, resp= self.resp, cond= self.cond, batch_size=self.batch_size, sample_mode='joint'), \
                        sample_batch(train_data, resp= self.resp, cond= self.cond, batch_size=self.batch_size, sample_mode=self.sample_mode)
        self.data_joint, self.data_mar = batchTrain
        for i in range(self.iter_num):
            #get train data
            # batchTrain = sample_batch(train_data, resp= self.resp, cond= self.cond, batch_size=self.batch_size, sample_mode='joint'), \
            #              sample_batch(train_data, resp= self.resp, cond= self.cond, batch_size=self.batch_size, sample_mode=self.sample_mode)
            batch_mi_lb, batch_loss = self.update_mine_net(batchTrain, self.mine_net_optim, ma_rate=self.ma_rate)
            train_batch_loss.append(batch_loss)
            train_batch_mi_lb.append(batch_mi_lb)

            mi_lb = self.forward_pass(self.X_train)
            train_mi_lb.append(mi_lb)
            
            mi_lb_valid = self.forward_pass(val_data)
            valid_mi_lb.append(mi_lb_valid)
            
            if self.avg_freq==1:
                self.avg_train_mi_lb.append(mi_lb)
                self.avg_valid_mi_lb.append(mi_lb_valid)
            elif (i+1)%(self.avg_freq)==0:
                train_loss = - np.average(train_mi_lb)
                valid_loss = - np.average(valid_mi_lb)
                self.avg_train_mi_lb.append(train_loss)
                self.avg_valid_mi_lb.append(valid_loss)

                if self.verbose:
                    print_msg = "[{0}/{1}] train_loss: {2} valid_loss: {3}".format(i, self.iter_num, train_loss, valid_loss)
                    print (print_msg)

                train_mi_lb = []
                valid_mi_lb = []

                if self.earlyStop:
                    earlyStop(valid_loss, self.mine_net)
                    if (earlyStop.early_stop):
                        if self.verbose:
                            print("Early stopping")
                        break
            if len(self.iter_snapshot)>j and (i+1)%self.iter_snapshot[j]==0:
                mi_lb_ = self.forward_pass(val_data)
                self.savefig(mi_lb_, suffix="_iter={}".format(self.iter_snapshot[j]))
                ch = "checkpoint_iter={}.pt".format(self.iter_snapshot[j])
                ch = os.path.join(self.prefix, ch)
                torch.save(self.mine_net.state_dict(), ch)
                j += 1
            if self.video_frames>0:
                x = np.linspace(self.Xmin, self.Xmax, 300)
                y = np.linspace(self.Ymin, self.Ymax, 300)
                xs, ys = np.meshgrid(x,y)
                # t = self.mine_net(torch.FloatTensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))).detach().numpy()
                t = self.mine_net(torch.Tensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))).detach().numpy()
                t = t.reshape(xs.shape[1], ys.shape[0])
                # ixy = t - np.log(self.ma_et.mean().detach().numpy())
                heatmap_animation_ax, c = plot_util.getHeatMap(heatmap_animation_ax, xs, ys, t)
                self.heatmap_frames.append((c,))
                if (i+1)%self.video_frames==0:
                    writer = animation.writers['ffmpeg'](fps=1, bitrate=1800)
                    heatmap_animation = animation.ArtistAnimation(heatmap_animation_fig, self.heatmap_frames, interval=200, blit=False)
                    heatmap_animation.save(os.path.join(self.prefix, "heatmap_less_than_{}.mp4".format(i+1)), writer=writer)
                    self.heatmap_frames=[]
    
        if self.log:
            # Save result to files
            avg_train_mi_lb = np.array(self.avg_train_mi_lb)
            np.savetxt(os.path.join(self.prefix, "avg_train_mi_lb.txt"), avg_train_mi_lb)
            avg_valid_mi_lb = np.array(self.avg_valid_mi_lb)
            np.savetxt(os.path.join(self.prefix, "avg_valid_mi_lb.txt"), avg_valid_mi_lb)

            train_batch_mi_lb = np.array(train_batch_mi_lb)
            np.savetxt(os.path.join(self.prefix, "train_batch_mi_lb.txt"), train_batch_mi_lb)
            train_batch_mi_lb = np.array(train_batch_mi_lb)
            np.savetxt(os.path.join(self.prefix, "train_batch_mi_lb.txt"), train_batch_mi_lb)

        if self.earlyStop:
            ch = os.path.join(self.prefix, "checkpoint.pt")
            self.mine_net.load_state_dict(torch.load(ch))#'checkpoint.pt'))

    
    def update_mine_net(self, batch, mine_net_optim, ma_rate=0.01):
        """[summary]
        
        Arguments:
            batch {[type]} -- ([batch_size X 2], [batch_size X 2])
            mine_net_optim {[type]} -- [description]
            ma_rate {float} -- [moving average rate] (default: {0.01})
        
        Keyword Arguments:
            mi_lb {} -- []
        """

        # batch is a tuple of (joint, marginal)
        joint , marginal = batch
        joint = torch.autograd.Variable(torch.FloatTensor(joint))
        marginal = torch.autograd.Variable(torch.FloatTensor(marginal))
        # joint = torch.autograd.Variable(torch.Tensor(joint))
        # marginal = torch.autograd.Variable(torch.Tensor(marginal))
        mi_lb , t, et = self.mutual_information(joint, marginal)
        self.ma_et = (1-ma_rate)*self.ma_et + ma_rate*torch.mean(et)
        
        # unbiasing use moving average
        loss = -(torch.mean(t) - (1/self.ma_et.mean()).detach()*torch.mean(et))
        # use biased estimator
    #     loss = - mi_lb
        lossTrain = loss
        mine_net_optim.zero_grad()
        autograd.backward(loss)
        mine_net_optim.step()
        return mi_lb.item(), lossTrain.item()

    
    def mutual_information(self, joint, marginal):
        t = self.mine_net(joint)
        et = torch.exp(self.mine_net(marginal))
        mi_lb = torch.mean(t) - torch.log(torch.mean(et))
        return mi_lb, t, et

    def forward_pass(self, X):
        joint = sample_batch(X, resp= self.resp, cond= self.cond, batch_size=X.shape[0], sample_mode='joint')
        marginal = sample_batch(X, resp= self.resp, cond= self.cond, batch_size=X.shape[0], sample_mode=self.sample_mode)
        joint = torch.autograd.Variable(torch.FloatTensor(joint))
        marginal = torch.autograd.Variable(torch.FloatTensor(marginal))
        # joint = torch.autograd.Variable(torch.Tensor(joint))
        # marginal = torch.autograd.Variable(torch.Tensor(marginal))
        mi_lb , _, _ = self.mutual_information(joint, marginal)
        return mi_lb.item()

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
    
        mi_lb = self.forward_pass(self.X_test)

        if self.log:
            self.savefig(mi_lb, suffix="_iter={}".format(self.iter_num))
            ch = "checkpoint_iter={}.pt".format(self.iter_num)
            ch = os.path.join(self.prefix, ch)
            torch.save(self.mine_net.state_dict(), ch)
        if self.sample_mode == 'unif':
            if 0 == len(self.cond):
                X_max, X_min = X_train[:,self.resp].max(axis=0), X_train[:,self.resp].min(axis=0)
                cross = np.log(X_max-X_min)
            else:
                X_max, X_min = X_train.max(axis=0), X_train.min(axis=0)
                cross = sum(np.log(X_max-X_min))
            return cross - mi_lb

        #release memory
        self.avg_train_mi_lb=[]
        self.avg_valid_mi_lb=[]
        self.X_test=[]
        self.X_train=[]
        self.heatmap_frames=[]
        return mi_lb


    def savefig(self, ml_lb_estimate, suffix=""):
        if len(self.cond) > 1:
            raise ValueError("Only support 2-dim or 1-dim")
        fig, ax = plt.subplots(1,4, figsize=(90, 15))
        #plot Data
        # ax[0].scatter(self.X_train[:,self.resp], self.X_train[:,self.cond], color='red', marker='o', label='train')
        # ax[0].scatter(self.X_test[:,self.resp], self.X_test[:,self.cond], color='green', marker='x', label='test')
        ax[0].scatter(self.data_joint[:,self.cond], self.data_joint[:,self.resp], color='red', marker='o', label='joint')
        ax[0].scatter(self.data_mar[:,self.cond], self.data_mar[:,self.resp], color='green', marker='x', label='marginal')
        ax[0].legend()

        #plot training curve
        # ax[1] = plot_util.getTrainCurve(self.avg_train_mi_lb, self.avg_valid_mi_lb, ax[1], show_min=self.earlyStop)
        ax[1] = plot_util.getTrainCurve(self.avg_train_mi_lb, [], ax[1], show_min=self.earlyStop, ground_truth=self.ground_truth)
        ax[1].set_title('train curve of total loss')

        # Trained Function contour plot
        x = np.linspace(self.Xmin, self.Xmax, 300)
        y = np.linspace(self.Ymin, self.Ymax, 300)
        xs, ys = np.meshgrid(x,y)
        # z = self.mine_net(torch.FloatTensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))).detach().numpy()
        z = self.mine_net(torch.Tensor(np.hstack((xs.flatten()[:,None],ys.flatten()[:,None])))).detach().numpy()
        z = z.reshape(xs.shape[1], ys.shape[0])
        ax[2], c = plot_util.getHeatMap(ax[2], xs, ys, z)

        fig.colorbar(c, ax=ax[2])
        ax[2].set_title('heatmap')

        # Plot result with ground truth
        ml_lb_train = self.forward_pass(self.X_train)
        ax[3].scatter(0, self.ground_truth, edgecolors='red', facecolors='none', label='Ground Truth')
        ax[3].scatter(0, ml_lb_estimate, edgecolors='green', facecolors='none', label="{}_Test".format(self.model_name))
        ax[3].scatter(0, ml_lb_train, edgecolors='blue', facecolors='none', label="{}_Train".format(self.model_name))
        ax[3].set_xlabel(self.paramName)
        ax[3].set_ylabel(self.y_label)
        ax[3].legend()
        figName = os.path.join(self.prefix, "MINE{}".format(suffix))
        fig.savefig(figName, bbox_inches='tight')
        plt.close()
        




