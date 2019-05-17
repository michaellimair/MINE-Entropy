import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def resample(data,batch_size,replace=False):
    index = np.random.choice(range(data.shape[0]), size=batch_size, replace=replace)
    batch = data[index]
    return batch
    
class Mine():
    class Net(nn.Module):       
        def __init__(self, input_size=2, hidden_size=100, sigma = 0.02):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, 1)
            nn.init.normal_(self.fc1.weight,std=sigma)
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.normal_(self.fc2.weight,std=sigma)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.normal_(self.fc3.weight,std=sigma)
            nn.init.constant_(self.fc3.bias, 0)

        def forward(self, input):
            output = F.elu(self.fc1(input))
            output = F.elu(self.fc2(output))
            output = self.fc3(output)
            return output
    
    def __init__(self, X, Y, batch_size=32, lr=1e-3, ma_rate=0.1, hidden_size=100, ma_ef = 1):
        self.lr = lr
        self.batch_size = batch_size
        self.ma_rate = ma_rate

        self.X = X
        self.Y = Y
        self.XY = torch.cat((self.X,self.Y),dim=1)
        
        self.X_ref = resample(self.X, batch_size=self.X.shape[0])
        self.Y_ref = resample(self.Y, batch_size=self.Y.shape[0])
        
        self.XY_net = Mine.Net(input_size=X.shape[1]+Y.shape[1],hidden_size=300)
        self.XY_optimizer = optim.Adam(self.XY_net.parameters(),lr=lr)

        self.ma_ef = ma_ef # for moving average

    def step(self):
        self.XY_optimizer.zero_grad()
        batch_XY = resample(self.XY,batch_size=self.batch_size)
        batch_XY_ref = torch.cat((resample(self.X,batch_size=self.batch_size), \
                                  resample(self.Y,batch_size=self.batch_size)),dim=1)
        fXY = self.XY_net(batch_XY)
        efXY_ref = torch.exp(self.XY_net(batch_XY_ref))
        batch_dXY = torch.mean(fXY) - torch.log(torch.mean(efXY_ref))
        self.ma_ef = (1-self.ma_rate)*self.ma_ef + self.ma_rate*torch.mean(efXY_ref)
        batch_loss_XY = -(torch.mean(fXY) - (1/self.ma_ef.mean()).detach()*torch.mean(efXY_ref))
        batch_loss_XY.backward()
        self.XY_optimizer.step()            
    
    def forward(self,X=None,Y=None):
        if X is None or Y is None:
            X, Y = self.X, self.Y
        XY = torch.cat((X,Y),dim=1)
        X_ref = resample(X,batch_size=X.shape[0])
        Y_ref = resample(Y,batch_size=Y.shape[0])
        XY_ref = torch.cat((X_ref,Y_ref),dim=1)
        return (torch.mean(self.XY_net(XY)) \
            - torch.log(torch.mean(torch.exp(self.XY_net(XY_ref))))).cpu().item()

    def state_dict(self):
        return {
            'XY_net' : self.XY_net.state_dict(),
            'X' : self.X,
            'Y' : self.Y,
            'lr' : self.lr,
            'batch_size' : self.batch_size,
            'ma_rate' : self.ma_rate,
            'ma_ef' : self.ma_ef
        }
    
    def load_state_dict(self,state_dict):
        self.XY_net.load_state_dict(state_dict['XY_net'])
        self.X = state_dict['X']
        self.Y = state_dict['Y']
        self.lr = state_dict['lr']
        self.batch_size = state_dict['batch_size']
        self.ma_rate = state_dict['ma_rate']
        self.ma_ef = state_dict['ma_ef']
