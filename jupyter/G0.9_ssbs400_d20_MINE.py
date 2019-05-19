
# coding: utf-8

# In[90]:


name = 'G0.9_ssbs400_d20_MINE' # filename to load/save the results
script_name = name+'.ipynb'
chkpt_name = name+'.pt'
fig_name = name+'.pdf'


# In[91]:


get_ipython().system('jupyter nbconvert --to script $script_name')


# In[70]:


import numpy as np
import torch
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    
import matplotlib as mpl
import matplotlib.pyplot as plt

from ipywidgets import interact

from IPython import display

import os

get_ipython().run_line_magic('matplotlib', 'auto')


# In[71]:


from model.mine import Mine
from data.gaussian import Gaussian


# In[72]:


np.random.seed(0)
torch.manual_seed(0)


# ## Data

# In[73]:


sample_size = 400
rho = 0.9


# In[74]:


rep = 5 # number of repeated runs
d = 20
X = np.zeros((rep,sample_size,d))
Y = np.zeros((rep,sample_size,d))
for i in range(rep):
    for j in range(d):
        data = Gaussian(sample_size=sample_size,rho=rho).data
        X[i,:,j] = data[:,0]
        Y[i,:,j] = data[:,1]


# Generate the reference samples by resampling.

# In[75]:


plt.scatter(X[0,:,0],Y[0,:,0],label="data",marker="+",color="steelblue")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot of data samples')


# ## MI estimation

# ### Choice of parameters

# In[76]:


batch_size = int(sample_size*1)
lr = 1e-3
ma_rate = 0.1


# ### Initialization

# In[77]:


mine_list = []
for i in range(rep):
    mine_list.append(Mine(torch.Tensor(X[i]),torch.Tensor(Y[i]),batch_size=batch_size,lr=lr,ma_rate=ma_rate))
dXY_list = np.zeros((rep,0))


# In[78]:


load_available = True
if load_available and os.path.exists(chkpt_name):
    checkpoint = torch.load(chkpt_name,map_location = 'cuda' if torch.cuda.is_available() else 'cpu')
    dXY_list = checkpoint['dXY_list']
    mine_state_list = checkpoint['mine_state_list']
    for i in range(rep):
        mine_list[i].load_state_dict(mine_state_list[i])


# ### Training

# In[79]:


for k in range(20):
    for j in range(200):
        dXY_list = np.append(dXY_list,np.zeros((rep,1)),axis=1)
        for i in range(rep):
            mine_list[i].step()
            dXY_list[i,-1] = mine_list[i].forward()
        # To show intermediate works
    for i in range(rep):
        plt.plot(dXY_list[i,:])
    display.clear_output(wait=True)
    display.display(plt.gcf())
display.clear_output()


# In[80]:


mine_state_list = [mine_list[i].state_dict() for i in range(rep)]


# In[81]:


torch.save({
    'dXY_list' : dXY_list,
    'mine_state_list' : mine_state_list
},chkpt_name)


# ## Analysis

# Ground truth mutual information

# In[82]:


mi = - 0.5 * np.log(1-rho **2) * d
print(mi)


# Apply moving average to smooth out the mutual information estimate.

# In[87]:


ma_rate = 1
mi_list = dXY_list.copy()
for i in range(1,dXY_list.shape[1]):
    mi_list[:,i] = (1-ma_rate) * mi_list[:,i-1] + ma_rate * dXY_list[:,i]
for i in range(rep):
    plt.plot(mi_list[i,:])
plt.axhline(mi)
plt.title("Plot of MI estimates against number of iteractions")
plt.xlabel("number of iterations")
plt.ylabel("MI estimate")
plt.savefig(fig_name)

