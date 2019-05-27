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
    def __init__(self, lr, batch_size, ma_rate, hidden_size=100, snapshot=[], iter_num=int(1e+3), model_name="MINE", log=True, prefix="", ground_truth=0, verbose=False, full_ref=False, load_dict=False, ref_factor=1):
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
            log.close()



        self.XY_net = MineNet(input_size=self.Train_X.shape[1]+self.Train_Y.shape[1],hidden_size=self.hidden_size)
        self.XY_optimizer = optim.Adam(self.XY_net.parameters(),lr=self.lr)

        self.Train_dXY_list = []
        self.Test_dXY_list = []

        self.ma_ef = 1
        snapshot_i = 0
        # set starting iter_num
        start_i = 0
        fname = os.path.join(self.prefix, "cache.pt")
        if self.load_dict and os.path.exists(fname):
            state_dict = torch.load(fname, map_location = "cuda" if torch.cuda.is_available() else "cpu")
            self.load_state_dict(state_dict)
            if self.verbose:
                print('results loaded from '+fname)

        self.XY_t = torch.Tensor(np.concatenate((self.Train_X,self.Train_Y),axis=1))

        if self.full_ref:
            Train_X_ref, Train_Y_ref = np.meshgrid(self.Train_X, self.Train_Y.T)
            if len(self.Train_X.shape)==1:
                Train_X_ref = Train_X_ref.flatten()[:,None]
                Train_Y_ref = Train_Y_ref.flatten()[:,None]
            elif len(self.Train_X.shape)==2:
                Train_X_ref = Train_X_ref[:self.Train_X.shape[0],:].reshape((self.Train_X.shape[0]**2), self.Train_X.shape[1])
                Train_Y_ref = Train_Y_ref[:,:self.Train_X.shape[0]].reshape(self.Train_X.shape[1], (self.Train_X.shape[0]**2)).T
        elif self.ref_factor > 1:
            Train_X_ref = resample(self.Train_X,batch_size=int(self.Train_X.shape[0]*self.ref_factor), replace=True)
            Train_Y_ref = resample(self.Train_Y,batch_size=int(self.Train_Y.shape[0]*self.ref_factor), replace=True)
        else:
            Train_X_ref = resample(self.Train_X,batch_size=int(self.Train_X.shape[0]*self.ref_factor), replace=False)
            Train_Y_ref = resample(self.Train_Y,batch_size=int(self.Train_Y.shape[0]*self.ref_factor), replace=False)
        self.XY_ref_t = torch.Tensor(np.concatenate((Train_X_ref,Train_Y_ref),axis=1))
        self.XY_ref_t_log_size = float(np.log(Train_X_ref.shape[0]))

        # Plot data and ref for MI estimate
        plt.scatter(self.Train_X,self.Train_Y)
        plt.scatter(Train_X_ref,Train_Y_ref,label="ref",marker="_",color="darkorange")
        plt.scatter(self.Train_X,self.Train_Y,label="data",marker="+",color="steelblue")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Plot of all data samples and reference samples')
        plt.legend()
        figName = os.path.join(self.prefix, "Data and Reference in MI estimate.png")
        plt.savefig(figName)
        plt.close()

        if len(self.Train_dXY_list) > 0:
            start_i = len(self.Train_dXY_list) + 1
            for i in range(len(self.snapshot)):
                if self.snapshot[i] <= start_i:
                    snapshot_i = i+1
        for i in range(start_i, self.iter_num):
            self.update_mine_net(self.Train_X, self.Train_Y, self.batch_size, self.ma_rate)
            Train_dXY = self.get_estimate(self.Train_X, self.Train_Y)
            self.Train_dXY_list = np.append(self.Train_dXY_list, Train_dXY)

            Test_dXY = self.get_estimate(self.Test_X, self.Test_Y)
            self.Test_dXY_list = np.append(self.Test_dXY_list, Test_dXY)

            if len(self.snapshot)>snapshot_i and (i+1)%self.snapshot[snapshot_i]==0:
                self.save_figure(suffix="iter={}".format(self.snapshot[snapshot_i]))
                # To save intermediate works, change the condition to True
                fname_i = os.path.join(self.prefix, "cache_iter={}.pt".format(i+1))
                if True:
                    with open(fname_i,'wb') as f:
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



    def update_mine_net(self, Train_X, Train_Y, batch_size, ma_rate):
        self.XY_optimizer.zero_grad()
        batch_XY = resample(self.XY_t,batch_size=batch_size)
        batch_XY_ref = torch.Tensor(np.concatenate((resample(Train_X,batch_size=batch_size),                                                         resample(Train_Y,batch_size=batch_size)),axis=1))

        fXY = self.XY_net(batch_XY)
        efXY_ref = torch.exp(self.XY_net(batch_XY_ref))
        self.ma_ef = (1-ma_rate)*self.ma_ef + ma_rate*torch.mean(efXY_ref)
        batch_dXY = -(torch.mean(fXY) - (1/self.ma_ef.mean()).detach()*torch.mean(efXY_ref))
        batch_dXY.backward()
        # batch_loss_XY = -batch_dXY
        # batch_loss_XY.backward()
        self.XY_optimizer.step()

    def get_estimate(self, X, Y):
        XY_t = torch.Tensor(np.concatenate((X,Y),axis=1))

        # dXY = torch.mean(self.XY_net(XY_t)) - torch.log(torch.mean(torch.exp(self.XY_net(self.XY_ref_t))))
        dXY = torch.mean(self.XY_net(XY_t)) - (torch.logsumexp(self.XY_net(self.XY_ref_t), 0) - self.XY_ref_t_log_size)
        return dXY.cpu().item()

    def predict(self, Train_X, Train_Y, Test_X, Test_Y):
        Train_X, Train_Y = np.array(Train_X), np.array(Train_Y)
        Test_X, Test_Y = np.array(Test_X), np.array(Test_Y)
        self.fit(Train_X,Train_Y, Test_X, Test_Y)

        mi_lb = self.Train_dXY_list[-1]

        if self.log:
            self.save_figure(suffix="iter={}".format(self.iter_num))
        self.Train_X = []
        self.Train_Y = []
        self.Test_X = []
        self.Test_Y = []

        self.XY_ref_t = []
        self.XY_net = []
        self.XY_optimizer = []

        self.Train_dXY_list = []
        self.Test_dXY_list = []
        return mi_lb

    def save_figure(self, suffix=""):
        fig, ax = plt.subplots(1,4, figsize=(90, 15))

        #plot mi_lb curve
        axCur = ax[0]
        Train_mi_lb = self.Train_dXY_list
        axCur = plot_util.getTrainCurve(Train_mi_lb , [], axCur, show_min=False, ground_truth=self.ground_truth)
        axCur.set_title('curve of training data mutual information')

        #plot mi_lb curve
        axCur = ax[1]
        Test_mi_lb = self.Test_dXY_list
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
            ixy = self.XY_net(mesh).detach().numpy()
            ixy = ixy.reshape(xs.shape[1], ys.shape[0])

            axCur = ax[2]
            axCur, c = plot_util.getHeatMap(axCur, xs, ys, ixy)
            fig.colorbar(c, ax=axCur)
            axCur.set_title('heatmap of i(x,y)')

            axCur = ax[3]
            axCur.scatter(self.Train_X, self.Train_Y, color='red', marker='o', label='train')
            axCur.scatter(self.Test_X, self.Test_Y, color='green', marker='x', label='test')
            axCur.set_title('Plot of all train data samples and test data samples')
            axCur.legend()


        figName = os.path.join(self.prefix, "{}_{}".format(self.model_name, suffix))
        fig.savefig(figName, bbox_inches='tight')
        plt.close()

    def state_dict(self):
        return {
            'Train_dXY_list' : self.Train_dXY_list,
            'Test_dXY_list' : self.Test_dXY_list,
            'XY_net': self.XY_net.state_dict(),
            'XY_optimizer': self.XY_optimizer.state_dict(),
            'Train_X': self.Train_X,
            'Train_Y': self.Train_Y,
            'Test_X': self.Test_X,
            'Test_Y': self.Test_Y,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'ma_rate': self.ma_rate,
            'ma_ef': self.ma_ef
        }

    def load_state_dict(self, state_dict):
        self.XY_net.load_state_dict(state_dict['XY_net'])
        self.XY_optimizer.load_state_dict(state_dict['XY_optimizer'])
        self.Train_X = state_dict['Train_X']
        self.Train_Y = state_dict['Train_Y']
        self.Test_X = state_dict['Test_X']
        self.Test_Y = state_dict['Test_Y']
        self.lr = state_dict['lr']
        self.batch_size = state_dict['batch_size']
        self.ma_rate = state_dict['ma_rate']
        self.ma_ef = state_dict['ma_ef']
        self.Test_dXY_list = state_dict['Test_dXY_list']
        self.Train_dXY_list = state_dict['Train_dXY_list']

