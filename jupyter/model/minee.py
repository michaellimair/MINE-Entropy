import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def resample(data, batch_size, replace=False):
    index = np.random.choice(
        range(data.shape[0]), size=batch_size, replace=replace)
    batch = data[index]
    return batch


def uniform_sample(data, batch_size, margin=9):
    data_min = data.min(dim=0)[0] - margin
    data_max = data.max(dim=0)[0] + margin
    return (data_max - data_min) * torch.rand((batch_size, data_min.shape[0])) + data_min


class MINEE():
    class Net(nn.Module):
        def __init__(self, input_size=2, hidden_size=100, sigma=0.02):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, 1)
            nn.init.normal_(self.fc1.weight, std=sigma)
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.normal_(self.fc2.weight, std=sigma)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.normal_(self.fc3.weight, std=sigma)
            nn.init.constant_(self.fc3.bias, 0)

        def forward(self, input):
            output = F.elu(self.fc1(input))
            output = F.elu(self.fc2(output))
            output = self.fc3(output)
            return output

    def __init__(self, X, Y, batch_size=32, ref_batch_factor=1, ref_margin=0, lr=1e-3, hidden_size=100):
        self.lr = lr
        self.batch_size = batch_size
        self.ref_batch_factor = ref_batch_factor
        self.ref_margin = ref_margin
        self.X = X
        self.Y = Y
        self.XY = torch.cat((self.X, self.Y), dim=1)

        self.X_ref = uniform_sample(X, batch_size=int(
            self.ref_batch_factor * X.shape[0]), margin=self.ref_margin)
        self.Y_ref = uniform_sample(Y, batch_size=int(
            self.ref_batch_factor * Y.shape[0]), margin=self.ref_margin)

        self.XY_net = MINEE.Net(
            input_size=X.shape[1]+Y.shape[1], hidden_size=100)
        self.X_net = MINEE.Net(input_size=X.shape[1], hidden_size=100)
        self.Y_net = MINEE.Net(input_size=Y.shape[1], hidden_size=100)
        self.XY_optimizer = optim.Adam(self.XY_net.parameters(), lr=lr)
        self.X_optimizer = optim.Adam(self.X_net.parameters(), lr=lr)
        self.Y_optimizer = optim.Adam(self.Y_net.parameters(), lr=lr)

    def step(self, iter=1):
        for i in range(iter):
            self.XY_optimizer.zero_grad()
            batch_XY = resample(self.XY, batch_size=self.batch_size)
            batch_XY_ref = torch.cat((resample(self.X, batch_size=self.batch_size),
                                    resample(self.Y, batch_size=self.batch_size)), dim=1)

            self.XY_optimizer.zero_grad()
            self.X_optimizer.zero_grad()
            self.Y_optimizer.zero_grad()
            batch_XY = resample(self.XY, batch_size=self.batch_size)
            batch_X = resample(self.X, batch_size=self.batch_size)
            batch_Y = resample(self.Y, batch_size=self.batch_size)
            batch_X_ref = uniform_sample(self.X, batch_size=int(
                self.ref_batch_factor * self.batch_size), margin=self.ref_margin)
            batch_Y_ref = uniform_sample(self.Y, batch_size=int(
                self.ref_batch_factor * self.batch_size), margin=self.ref_margin)
            batch_XY_ref = torch.cat((batch_X_ref, batch_Y_ref), dim=1)

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

    def forward(self, X=None, Y=None):
        XY = None
        if X is None or Y is None:
            XY, X, Y = self.XY, self.X, self.Y
        else:
            XY = torch.cat((X, Y), dim=1)
        X_ref = uniform_sample(X, batch_size=int(
            self.ref_batch_factor * X.shape[0]), margin=self.ref_margin)
        Y_ref = uniform_sample(Y, batch_size=int(
            self.ref_batch_factor * Y.shape[0]), margin=self.ref_margin)
        XY_ref = torch.cat((X_ref, Y_ref), dim=1)
        dXY = (torch.mean(self.XY_net(XY))
               - torch.log(torch.mean(torch.exp(self.XY_net(XY_ref))))).cpu().item()
        dX = (torch.mean(self.X_net(X))
              - torch.log(torch.mean(torch.exp(self.X_net(X_ref))))).cpu().item()
        dY = (torch.mean(self.Y_net(Y))
              - torch.log(torch.mean(torch.exp(self.Y_net(Y_ref))))).cpu().item()
        return dXY, dX, dY

    def state_dict(self):
        return {
            'XY_net': self.XY_net.state_dict(),
            'XY_optimizer': self.XY_optimizer.state_dict(),
            'X_net': self.X_net.state_dict(),
            'X_optimizer': self.X_optimizer.state_dict(),
            'Y_net': self.Y_net.state_dict(),
            'Y_optimizer': self.Y_optimizer.state_dict(),
            'X': self.X,
            'Y': self.Y,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'ref_batch_factor': self.ref_batch_factor,
            'ref_margin': self.ref_margin
        }

    def load_state_dict(self, state_dict):
        self.XY_net.load_state_dict(state_dict['XY_net'])
        self.XY_optimizer.load_state_dict(state_dict['XY_optimizer'])
        self.X_net.load_state_dict(state_dict['X_net'])
        self.X_optimizer.load_state_dict(state_dict['X_optimizer'])
        self.Y_net.load_state_dict(state_dict['Y_net'])
        self.Y_optimizer.load_state_dict(state_dict['Y_optimizer'])
        self.X = state_dict['X']
        self.Y = state_dict['Y']
        if 'lr' in state_dict:
            self.lr = state_dict['lr']
        if 'batch_size' in state_dict:
            self.batch_size = state_dict['batch_size']
        if 'ref_batch_factor' in state_dict:
            self.ref_batch_factor = state_dict['ref_batch_factor']
        if 'ref_margin' in state_dict:
            self.ref_margin = state_dict['ref_margin']
