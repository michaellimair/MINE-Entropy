import numpy as np
random_seed = 2
np.random.seed(seed=random_seed)
from ..model.kraskov import Kraskov

from scipy.integrate import quad, dblquad
from scipy.special import xlogy

class Dataset():
    def __init__(self, filepath, col_x=0, col_y=1, delim='\",\"', col_nan=1, row_nan=1, index=""):
        self.path = filepath
        self.col_x = col_x
        self.col_y = col_y
        self.delim = delim
        self.col_nan = col_nan
        self.row_nan = row_nan
        self.totalRow = 0
        self.totalCol = 0
        data = np.genfromtxt(self.path, delimiter=self.delim)[self.row_nan:,self.col_nan:]
        self.totalRow, self.totalCol = data.shape
        self.X = data[:,(self.col_x, self.col_y)].copy()
        self.X = self.X[~np.isnan(self.X).any(axis=1)]

    @property
    def data(self):
        """[summary]
        Returns:
            [np array] -- [N by 2 matrix]
        """
        return self.X

    @property
    def ground_truth(self):
        kk = Kraskov('auto', 3)
        return kk.predict(self.X)

    def plot_i(self, ax, xs, ys):
        i_ = [self.I(xs[i,j], ys[i,j]) for j in range(ys.shape[1]) for i in range(xs.shape[0])]
        i_ = np.array(i_).reshape(xs.shape[0], ys.shape[1])
        i_ = i_[:-1, :-1]
        # i_min, i_max = -np.abs(i_).max(), np.abs(i_).max()
        i_min, i_max = i_.min(), i_.max()
        c = ax.pcolormesh(xs, ys, i_, cmap='RdBu', vmin=i_min, vmax=i_max)
        # set the limits of the plot to the limits of the data
        ax.axis([xs.min(), xs.max(), ys.min(), ys.max()])
        return ax, c
    
    def I(self, x,y):
        # cov = np.array(self.rho)
        dim = 2
        covMat, mu = (self.var*np.identity(len(self.mean))+self.rho*np.identity(len(self.mean))[::-1]), np.array(self.mean)
        def fxy(x,y):
            if type(x)==np.float64 or type(x)==float:
                X = np.array([x, y])
            else:
                X = np.concatenate((x,y))
            temp1 = np.matmul(np.matmul(X-mu , np.linalg.inv(covMat)), (X-mu).transpose())
            return np.exp(-.5*temp1) / (((2*np.pi)**(dim))* np.sqrt(np.linalg.det(covMat))) 

        def fx(x):
            if type(x)==np.float64 or type(x)==float:
                return np.exp(-(x-mu[0])**2/(2*covMat[0,0])) / np.sqrt(2*np.pi*covMat[0,0])
            else:
                temp1 = np.matmul(np.matmul(x-mu[0:dim] , np.linalg.inv(covMat[0:dim,0:dim])), (x-mu[0:dim]).transpose())
                return np.exp(-.5*temp1) / (((2*np.pi)**(dim /2))* np.sqrt(np.linalg.det(covMat[0:dim,0:dim])))

        def fy(y):
            if type(y)==np.float64 or type(y)==float:
                return np.exp(-(y-mu[1])**2/(2*covMat[1,1])) / np.sqrt(2*np.pi*covMat[1,1])*dim
            else:
                temp1 = np.matmul(np.matmul(y-mu[-dim:] , np.linalg.inv(covMat[-dim:,-dim:])), (y-mu[-dim:]).transpose())
                return np.exp(-.5*temp1) / (((2*np.pi)**(dim /2))* np.sqrt(np.linalg.det(covMat[-dim:,-dim:])))

        return np.log(fxy(x, y)/(fx(x)*fy(y)))
    
    # def sum_d(self, x,y):
    #     sum_d_res = []
    #     dim = len(self.mean)//2
    #     realmean = self.mean
    #     for dim_no in range(0, dim):
    #         X = x[dim_no]
    #         Y = y[dim_no]
    #         self.mean = np.zeros(2).tolist() #temporarily modify mean for dimensional pair
    #         sum_d_res.append(self.I(X, Y)) 
    #     self.mean = realmean #save real mean again
    #     # def fxy(x,y, covMat, mu):
    #     #     if type(x)==np.float64 or type(x)==float:
    #     #         X = np.array([x, y])
    #     #     else:
    #     #         raise ValueError("x should be float")
    #     #     temp1 = np.matmul(np.matmul(X-mu , np.linalg.inv(covMat)), (X-mu).transpose())
    #     #     return np.exp(-.5*temp1) / (2*np.pi * np.sqrt(1-self.rho**2)) 

    #     # def fx(x, covMat, mu):
    #     #     if type(x)==np.float64 or type(x)==float:
    #     #         return np.exp(-(x-mu[0])**2/(2*covMat[0,0])) / np.sqrt(2*np.pi*covMat[0,0])
    #     #     else:
    #     #         raise ValueError("x should be float")

    #     # def fy(y, covMat, mu):
    #     #     if type(y)==np.float64 or type(y)==float:
    #     #         return np.exp(-(y-mu[1])**2/(2*covMat[1,1])) / np.sqrt(2*np.pi*covMat[1,1])*dim
    #     #     else:
    #     #         raise ValueError("y should be float")
    #     # sum_d_res = []
    #     # dim = len(self.mean)//2
    #     # for dim_no in range(0, dim):
    #     #     X = x[dim_no]
    #     #     Y = y[-dim_no-1]
    #     #     mu = np.array([self.mean[dim_no], self.mean[-dim_no-1]])
    #     #     covMat = (np.identity(2)+self.rho*np.identity(2)[::-1])
    #     #     fxy_ = fxy(X, Y, covMat, mu)
    #     #     fx_ = fx(X, covMat, mu)
    #     #     fy_ = fy(Y, covMat, mu)
    #     #     mi = np.log(fxy_/(fx_*fy_))
    #     #     sum_d_res.append(mi) 
    #     return np.sum(sum_d_res)

# class Gaussian():
#     def __init__(self, sample_size, rho, mean=[0, 0], var=1):
#         self.sample_size = sample_size
#         self.mean = mean
#         self.rho = rho
#         self.var = var

#     @property
#     def data(self):
#         """[summary]
#         Returns:
#             [np array] -- [N by 2 matrix]
#         """
#         if len(self.mean)%2 == 1:
#             raise ValueError("length of mean array is assummed to be even")
#         cov = (self.var*np.identity(len(self.mean))+self.rho*np.identity(len(self.mean))[::-1]).tolist()
#         return np.random.multivariate_normal(
#             mean=self.mean,
#             cov=cov, 
#             size=self.sample_size)

#     @property
#     def ground_truth(self):
#         # cov = np.array(self.rho)
#         if len(self.mean)%2 == 1:
#             raise ValueError("length of mean array is assummed to be even")
#         dim = len(self.mean)//2
#         return -0.5*np.log(self.var**2-self.rho**2)*dim

#     def plot_i(self, ax, xs, ys):
#         if len(self.mean)!=2:
#             raise ValueError("Only 2-dimension gaussian can be plotted")
#         i_ = [self.I(xs[i,j], ys[i,j]) for j in range(ys.shape[1]) for i in range(xs.shape[0])]
#         i_ = np.array(i_).reshape(xs.shape[0], ys.shape[1])
#         i_ = i_[:-1, :-1]
#         # i_min, i_max = -np.abs(i_).max(), np.abs(i_).max()
#         i_min, i_max = i_.min(), i_.max()
#         c = ax.pcolormesh(xs, ys, i_, cmap='RdBu', vmin=i_min, vmax=i_max)
#         # set the limits of the plot to the limits of the data
#         ax.axis([xs.min(), xs.max(), ys.min(), ys.max()])
#         return ax, c
    
#     def I(self, x,y):
#         # cov = np.array(self.rho)
#         if len(self.mean)%2 == 1:
#             raise ValueError("length of mean array is assummed to be even")
#         dim = len(self.mean)//2
#         covMat, mu = (self.var*np.identity(len(self.mean))+self.rho*np.identity(len(self.mean))[::-1]), np.array(self.mean)
#         def fxy(x,y):
#             if type(x)==np.float64 or type(x)==float:
#                 X = np.array([x, y])
#             else:
#                 X = np.concatenate((x,y))
#             temp1 = np.matmul(np.matmul(X-mu , np.linalg.inv(covMat)), (X-mu).transpose())
#             return np.exp(-.5*temp1) / (((2*np.pi)**(dim))* np.sqrt(np.linalg.det(covMat))) 

#         def fx(x):
#             if type(x)==np.float64 or type(x)==float:
#                 return np.exp(-(x-mu[0])**2/(2*covMat[0,0])) / np.sqrt(2*np.pi*covMat[0,0])
#             else:
#                 temp1 = np.matmul(np.matmul(x-mu[0:dim] , np.linalg.inv(covMat[0:dim,0:dim])), (x-mu[0:dim]).transpose())
#                 return np.exp(-.5*temp1) / (((2*np.pi)**(dim /2))* np.sqrt(np.linalg.det(covMat[0:dim,0:dim])))

#         def fy(y):
#             if type(y)==np.float64 or type(y)==float:
#                 return np.exp(-(y-mu[1])**2/(2*covMat[1,1])) / np.sqrt(2*np.pi*covMat[1,1])*dim
#             else:
#                 temp1 = np.matmul(np.matmul(y-mu[-dim:] , np.linalg.inv(covMat[-dim:,-dim:])), (y-mu[-dim:]).transpose())
#                 return np.exp(-.5*temp1) / (((2*np.pi)**(dim /2))* np.sqrt(np.linalg.det(covMat[-dim:,-dim:])))

#         return np.log(fxy(x, y)/(fx(x)*fy(y)))
    
#     # def sum_d(self, x,y):
#     #     sum_d_res = []
#     #     dim = len(self.mean)//2
#     #     realmean = self.mean
#     #     for dim_no in range(0, dim):
#     #         X = x[dim_no]
#     #         Y = y[dim_no]
#     #         self.mean = np.zeros(2).tolist() #temporarily modify mean for dimensional pair
#     #         sum_d_res.append(self.I(X, Y)) 
#     #     self.mean = realmean #save real mean again
#     #     # def fxy(x,y, covMat, mu):
#     #     #     if type(x)==np.float64 or type(x)==float:
#     #     #         X = np.array([x, y])
#     #     #     else:
#     #     #         raise ValueError("x should be float")
#     #     #     temp1 = np.matmul(np.matmul(X-mu , np.linalg.inv(covMat)), (X-mu).transpose())
#     #     #     return np.exp(-.5*temp1) / (2*np.pi * np.sqrt(1-self.rho**2)) 

#     #     # def fx(x, covMat, mu):
#     #     #     if type(x)==np.float64 or type(x)==float:
#     #     #         return np.exp(-(x-mu[0])**2/(2*covMat[0,0])) / np.sqrt(2*np.pi*covMat[0,0])
#     #     #     else:
#     #     #         raise ValueError("x should be float")

#     #     # def fy(y, covMat, mu):
#     #     #     if type(y)==np.float64 or type(y)==float:
#     #     #         return np.exp(-(y-mu[1])**2/(2*covMat[1,1])) / np.sqrt(2*np.pi*covMat[1,1])*dim
#     #     #     else:
#     #     #         raise ValueError("y should be float")
#     #     # sum_d_res = []
#     #     # dim = len(self.mean)//2
#     #     # for dim_no in range(0, dim):
#     #     #     X = x[dim_no]
#     #     #     Y = y[-dim_no-1]
#     #     #     mu = np.array([self.mean[dim_no], self.mean[-dim_no-1]])
#     #     #     covMat = (np.identity(2)+self.rho*np.identity(2)[::-1])
#     #     #     fxy_ = fxy(X, Y, covMat, mu)
#     #     #     fx_ = fx(X, covMat, mu)
#     #     #     fy_ = fy(Y, covMat, mu)
#     #     #     mi = np.log(fxy_/(fx_*fy_))
#     #     #     sum_d_res.append(mi) 
#     #     return np.sum(sum_d_res)




if __name__ == '__main__':
    # rho = 1-(1e-3)
    # gaus=Gaussian(200, 0, 1, rho)
    # # data = gaus.data
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    eq = []
    integral = []
    rhos = []
    log_rhos = []
    for i in range(12):
        rho = 1 - 1/(10**i)
        rhos.append(rho)
        log_rhos.append(-i)
        # print(rho)
        gaus=Gaussian(200,rho,[0,0],rho)
        integral.append(gaus.ground_truth)
        eq.append(-0.5*np.log(1-rho*rho))
    fig, axs = plt.subplots(1, 2, figsize=(16*2,9))
    ax = axs[0]
    ax.plot(rhos, eq, color='r', marker='o', label='equation')
    ax.plot(rhos, integral, color='g', marker='x', label='integral')
    ax.legend()
    ax.set_xlabel('rho')
    ax.set_ylabel('mutual information')
    ax.set_title("mi ground truth comparison")

    ax = axs[1]
    ax.plot(log_rhos, eq, color='r', marker='o', label='equation')
    ax.plot(log_rhos, integral, color='g', marker='x', label='integral')
    ax.legend()
    ax.set_xlabel('log(1-rho)')
    ax.set_ylabel('mutual information')
    ax.set_title("mi ground truth comparison")

    figName = os.path.join("minee/experiments", "mi_ground_truth_compare_Test.png")
    fig.savefig(figName, bbox_inches='tight')
    plt.close()