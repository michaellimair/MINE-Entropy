

from notebook_util_mine import *
# get_ipython().run_line_magic('matplotlib', 'inline')


import torch
random_seed = 1
torch.manual_seed(seed=random_seed)
from minee.data.mix_gaussian import MixedGaussian


# ## Data


np.random.seed(0)
sample_size = 200
rho1 = 0.9
data = MixedGaussian(sample_size=sample_size,rho1=rho1).data


# Separate X data and Y data for the estimation. This should be a preprocessing done before instead of after feeding the data to the model.


X = data[:,[0]]
Y = data[:,[1]]





# Generate the reference samples by resampling.


X_ref = resample(X,batch_size=sample_size)
Y_ref = resample(Y,batch_size=sample_size)
Y_ref = resample(Y_ref,batch_size=sample_size)
import os
experiment_name = input('Please enter the experiment name: ')
experiment_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), experiment_name)
while True:
    if os.path.exists(experiment_path):
        break
    else:
        os.makedirs(experiment_path)
        print('Output will be saved into {}'.format(experiment_path))


plt.scatter(X,Y)

plt.scatter(X,Y,label="data",marker="+",color="steelblue")
plt.scatter(X_ref,Y_ref,label="ref",marker="_",color="darkorange")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot of all data samples and reference samples')
plt.legend()
figName = os.path.join(experiment_path,"mix_gaussian_mine_full_batch_ma_data.png")
plt.savefig(figName)
plt.close()


# ## MI estimation

# ### Choice of parameters


batch_size = int(sample_size*1)
lr = 1e-3
ma_rate = 0.1
fname = 'mix_gaussian_mine_full_batch_ma.pt' # file to load/save the results


# ### Initialization


XY_t = torch.Tensor(np.concatenate((X,Y),axis=1))
XY_ref_t = torch.Tensor(np.concatenate((X_ref,Y_ref),axis=1))

XY_net = Net(input_size=X.shape[1]+Y.shape[1],hidden_size=300)
XY_optimizer = optim.Adam(XY_net.parameters(),lr=lr)

ma_ef = 1 # for moving average

# For storing the results

XY_net_list = []
dXY_list = []


# ### Training
# Automatically load previous results from db file if exists


if os.path.exists(fname):
    with open(fname,'rb') as f:
        checkpoint = torch.load(fname,map_location = "cuda" if torch.cuda.is_available() else "cpu")
        XY_net_list = checkpoint['XY_net_list']
        dXY_list = checkpoint['dXY_list']
        XY_net.load_state_dict(XY_net_list[-1])
        print('results loaded from '+fname)
else:
    for j in range(50):
        for i in range(200):
            XY_optimizer.zero_grad()
            batch_XY = resample(XY_t,batch_size=batch_size)
            batch_XY_ref = torch.Tensor(np.concatenate((resample(X,batch_size=batch_size),                                                         resample(Y,batch_size=batch_size)),axis=1))
            
            fXY = XY_net(batch_XY)
            efXY_ref = torch.exp(XY_net(batch_XY_ref))
            batch_dXY = torch.mean(fXY) - torch.log(torch.mean(efXY_ref))
            ma_ef = (1-ma_rate)*ma_ef + ma_rate*torch.mean(efXY_ref)
            batch_loss_XY = -(torch.mean(fXY) - (1/ma_ef.mean()).detach()*torch.mean(efXY_ref))
            batch_loss_XY.backward()
            XY_optimizer.step()    
            dXY_list = np.append(dXY_list,                                     (torch.mean(XY_net(XY_t))                                      - torch.log(torch.mean(torch.exp(XY_net(XY_ref_t))))).cpu().item())
        mi_list = dXY_list
        plt.plot(mi_list)
        plt.title("Plot of MI estimates against number iteractions")
        plt.xlabel("number of iterations")
        plt.ylabel("MI estimate")
        # plt.show()
        figName = os.path.join(experiment_path,"mix_gaussian_mine_full_batch_ma_iter={}.png".format(j))
        plt.savefig(figName)
        plt.close()
            
        XY_net_list = np.append(XY_net_list,copy.deepcopy(XY_net.state_dict()))

        # To save intermediate works, change the condition to True
        if False:
            with open(str(j)+fname,'wb') as f:
                dill.dump([XY_net_list,dXY_list],f)
                print('results saved: '+str(j))


# To save new results to a db file using the following code, delete the existing db file.


# if not os.path.exists(fname):
#     with open(fname,'wb') as f:
#         torch.save({
#             'dXY_list' : dXY_list,
#             'XY_net_list' : XY_net_list
#         },f)
#         #dill.dump([XY_net_list,X_net_list,Y_net_list,dXY_list,dX_list,dY_list],f)
#         print('results saved to '+fname)


# ## Analysis


mi_list = dXY_list
plt.plot(mi_list)
plt.title("Plot of MI estimates against number iteractions")
plt.xlabel("number of iterations")
plt.ylabel("MI estimate")
plt.show()
plt.savefig("mix_gaussian_mine_full_batch_ma.png")
plt.close()


# XY_net_ = copy.deepcopy(XY_net)



# T = mi_list.size # total number of iteractions
# dt = T // XY_net_list.shape[0] # period for each snapshot of the NN
# @interact(t=(dt,T,dt))
# def f(t=T):
#     plt.subplot(121)
#     plt.plot(mi_list)
#     plt.axvline(t)
#     plt.subplot(122)
#     XY_net_.load_state_dict(XY_net_list[(t // dt) - 1])
#     plot_net_2(XY_net_)
#     plt.show()



# @interact(t=(100,T,100))
# def f(t=T):
#     w = 50
#     plt.plot(mi_list,color='yellow')
#     mi = mi_list[:t+1][-w:].mean()
#     plt.axhline(mi)
#     plt.axvline(t)
#     print(mi)


