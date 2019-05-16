
#%%
from notebook_util import *
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
from minee.data.mix_gaussian import MixedGaussian

#%% [markdown]
# - It is better to use `sample_size` instead of `n_sample`.
# - Use the default `tensor` instead of `FloatTensor`. 

#%%
np.random.seed(0)
sample_size = 200
rho1 = 0.9
data = MixedGaussian(sample_size=sample_size,rho1=rho1).data
data_t = torch.Tensor(data)

#%% [markdown]
# Separate X data and Y data for the estimation. This should be a preprocessing done before instead of after feeding the data to the model.

#%%
X = data[:,[0]]
Y = data[:,[1]]


#%%
plt.scatter(X,Y)

#%% [markdown]
# Generate the reference samples by resampling.

#%%
X_ref = resample(X,batch_size=sample_size)
Y_ref = resample(Y,batch_size=sample_size)


#%%
plt.scatter(X,Y,label="data")
plt.scatter(X_ref,Y_ref,label="ref")
plt.legend()


#%%
XY_t = torch.Tensor(np.concatenate((X,Y),axis=1))


#%%
XY_ref_t = torch.Tensor(np.concatenate((X_ref,Y_ref),axis=1))


#%%
batch_size = int(sample_size*0.5)
net_list = []
mi_lb_list = []
batch_mi_lb_list = []
batch_loss_list = []
ma_rate = 1
ma_ef = 1
lr = 1e-3
XY_net = Net(input_size=X.shape[1]+Y.shape[1])
optimizer = optim.Adam(XY_net.parameters(),lr=lr)

#%% [markdown]
# Automatically load previous results from db file if exists

#%%
fname = 'mix_gaussian_small_batch_no_ma.db'
if os.path.exists(fname):
    with open(fname,'rb') as f:
        net_list,mi_lb_list,batch_loss_list,batch_mi_lb_list,ma_ef = dill.load(f)
        XY_net.load_state_dict(net_list[-1])
        print('results loaded from '+fname)

#%% [markdown]
# Repeately run the following to continue to train

#%%
for j in range(50):
    for i in range(1000):
        optimizer.zero_grad()
        batch_XY = resample(XY_t,batch_size=batch_size)
        #batch_XY_ref = resample(XY_ref_t,batch_size=batch_size)
        batch_XY_ref = torch.Tensor(np.concatenate((resample(X,batch_size=batch_size),                                                     resample(Y,batch_size=batch_size)),axis=1))
        fXY = XY_net(batch_XY)
        efXY_ref = torch.exp(XY_net(batch_XY_ref))
        batch_mi_lb = torch.mean(fXY) - torch.log(torch.mean(efXY_ref))
        #loss = -batch_mi_lb
        ma_ef = (1-ma_rate)*ma_ef + ma_rate*torch.mean(efXY_ref)
        batch_loss = -(torch.mean(fXY) - (1/ma_ef.mean()).detach()*torch.mean(efXY_ref))
        batch_loss.backward()
        optimizer.step()    
        mi_lb_list = np.append(mi_lb_list,                                 (torch.mean(XY_net(XY_t))                                  - torch.log(torch.mean(torch.exp(XY_net(XY_ref_t))))).cpu().item())
        batch_mi_lb_list = np.append(batch_mi_lb_list,batch_mi_lb.cpu().item())
        batch_loss_list = np.append(batch_loss_list,batch_loss.cpu().item())
    net_list = np.append(net_list,XY_net.state_dict())
    # save existing work
    with open(str(j)+fname,'wb') as f:
        dill.dump([net_list,mi_lb_list,batch_loss_list,batch_mi_lb_list,ma_ef],f)
        print('results saved: '+str(j))


#%%
plt.plot(mi_lb_list)


#%%
plot_net(XY_net)

#%% [markdown]
# To save new results to a db file using the following code, delete the existing db file.

#%%
if not os.path.exists(fname):
    with open(fname,'wb') as f:
        dill.dump([net_list,mi_lb_list,batch_loss_list,batch_mi_lb_list,ma_ef],f)
        print('results saved to '+fname)


#%%
mi_max_envelope_estimate(mi_lb_list,2000,w1=10,w2=50)


#%%



