import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import xlogy

class Additive_noise:
    def __init__(self, n_sample, shape, noise_ratio_amplitude):
        self.n_sample = n_sample
        self.shape = shape
        self.noise_ratio_amplitude = noise_ratio_amplitude

    @property
    def data(self):
        """[summary]
        Returns:
            [np array] -- [N by 2 matrix]
        """
        data_ = None
        nr = self.noise_ratio_amplitude
        if "linear_sym" == self.shape:
            x = np.random.normal(0, 1, self.n_sample)
            epsilon = np.random.normal(0, 1, self.n_sample)
            y = 2*x/3.0 + nr*epsilon
            data_ = np.array([x,y]).T
        elif "linear_asym" == self.shape:
            x = np.random.normal(0, 1, self.n_sample)
            epsilon = np.random.normal(0, 1, self.n_sample)**2-1
            y = 0.05*x + nr*epsilon
            data_ = np.array([x,y]).T
        elif "quad" == self.shape:
            x = np.random.normal(0, 1, self.n_sample)
            epsilon = np.random.normal(0, 1, self.n_sample)
            y = 2*(x**2)/3.0 + nr*epsilon
            data_ = np.array([x,y]).T
        # elif "circle" == self.shape:
        #     theta = np.random
        #     xi = 
        #     epsilon = 
        # elif "spiral" == self.shape:
        # elif "cloud" == self.shape:
        # elif "sine" == self.shape:
        # elif "diamond" == self.shape:
        # elif "x-para" == self.shape:
        # elif "step" == self.shape:
        
        return data_

    @property
    def ground_truth(self):
        nr = self.noise_ratio_amplitude
        lim = np.inf 
        if "linear_sym" == self.shape:
            def fx(x, m=0, v=1):
                return np.exp(-(x-m)**2/(2*v)) / np.sqrt(2*np.pi*v)
            def fy(y):
                return quad(lambda t: fx(t,0,2/3)*fx(y-t,0,nr),-lim,lim)
            def fxy(x, y):
                return fx(x)*fx(y,2*x/3,nr)
        elif "linear_asym" == self.shape:
            def fx(x, m=0, v=1):
                return np.exp(-(x-m)**2/(2*v)) / np.sqrt(2*np.pi*v)
            from scipy.special import gamma
            def chi_sq_pdf(x, k=1):
                if x > 0:
                    return x**(k/2-1)*np.exp(-x/2)/(2**(k/2)*gamma(k/2))
                else:
                    return 0
            def fy(y):
                fy_ = quad(lambda t: fx(t,0,0.05)*chi_sq_pdf((y-t)/nr),-lim,lim)
                return fy_[0]
            def fxy(x, y):
                return fx(x)*chi_sq_pdf((y-0.05*x)/nr)
        elif "quad" == self.shape:
            def fx(x, m=0, v=1):
                return np.exp(-(x-m)**2/(2*v)) / np.sqrt(2*np.pi*v)
            from scipy.special import gamma
            def chi_sq_pdf(x, k=1):
                if x > 0:
                    return x**(k/2-1)*np.exp(-x/2)/(2**(k/2)*gamma(k/2))
                else:
                    return 0
            def fy(y):
                fy_ = quad(lambda t: fx(t,0,0.05)*chi_sq_pdf((y-t)/nr),-lim,lim)
                return fy_[0]
            def fxy(x, y):
                return chi_sq_pdf((3/2)**(.5)*x)*fx(y, 2*(x**2)/3, nr)
        # elif "circle" == self.shape:
        # elif "spiral" == self.shape:
        # elif "cloud" == self.shape:
        # elif "sine" == self.shape:
        # elif "diamond" == self.shape:
        # elif "x-para" == self.shape:
        # elif "step" == self.shape:
        hx = quad(lambda x: -xlogy(fx(x),fx(x)), -lim, lim)
        isReliable = hx[1]
        # print(isReliable)
        hy = quad(lambda y: -xlogy(fy(y),fy(y)), -lim, lim)
        isReliable = np.maximum(isReliable,hy[1])
        # print(isReliable)
        hxy = dblquad(lambda x, y: -xlogy(fxy(x,y),fxy(x,y)), -lim, lim, lambda x:-lim, lambda x:lim)
        isReliable = np.maximum(isReliable,hxy[1])
        return hx[0] + hy[0] - hxy[0]

    def plot_i(self, ax, xs, ys):
        nr = self.noise_ratio_amplitude
        lim = np.inf 
        if "linear_sym" == self.shape:
            def fx(x, m=0, v=1):
                return np.exp(-(x-m)**2/(2*v)) / np.sqrt(2*np.pi*v)
            def fy(y):
                fy_ = quad(lambda t: fx(t,0,2/3)*fx(y-t,0,nr),-lim,lim)
                return fy_[0]
            def fxy(x, y):
                return fx(x)*fx(y,2*x/3,nr)
        elif "linear_asym" == self.shape:
            def fx(x, m=0, v=1):
                return np.exp(-(x-m)**2/(2*v)) / np.sqrt(2*np.pi*v)
            from scipy.special import gamma
            def guassian_sq_pdf(x, k=1):
                if x > 0:
                    return np.exp(-x/2)/((2*np.pi*x)**.5)
                    # return (x**(k/2-1))*np.exp(-x/2)/(2**(k/2)*gamma(k/2))
                else:
                    return 0
            def fy(y):
                fy_ = quad(lambda t: fx(t,0,0.05)*guassian_sq_pdf((y-t)/nr),-lim,lim)
                return fy_[0]
            def fxy(x, y):
                return fx(x)*guassian_sq_pdf((y-x/0.05)/nr)
        elif "quad" == self.shape:
            def fx(x, m=0, v=1):
                return np.exp(-(x-m)**2/(2*v)) / np.sqrt(2*np.pi*v)
            from scipy.special import gamma
            def guassian_sq_pdf(x, k=1):
                if x > 0:
                    return (x**(k/2-1))*np.exp(-x/2)/(2**(k/2)*gamma(k/2))
                else:
                    return 0
            def fy(y):
                fy_ = quad(lambda t: fx(t,0,0.05)*guassian_sq_pdf((y-t)/nr),-lim,lim)
                return fy_[0]
            def fxy(x, y):
                return guassian_sq_pdf((3/2)**(.5)*x)*fx(y, 2*(x**2)/3, nr)
        # elif "circle" == self.shape:
        #     def fx(x):
        #     def fy(y):
        #     def fxy(x, y):
        # elif "spiral" == self.shape:
        #     def fx(x):
        #     def fy(y):
        #     def fxy(x, y):
        # elif "cloud" == self.shape:
        #     def fx(x):
        #     def fy(y):
        #     def fxy(x, y):
        # elif "sine" == self.shape:
        #     def fx(x):
        #     def fy(y):
        #     def fxy(x, y):
        # elif "diamond" == self.shape:
        #     def fx(x):
        #     def fy(y):
        #     def fxy(x, y):
        # elif "x-para" == self.shape:
        #     def fx(x):
        #     def fy(y):
        #     def fxy(x, y):
        # elif "step" == self.shape:
            # def fx(x):
            # def fy(y):
            # def fxy(x, y):

        i = [fxy(xs[i,j], ys[i,j]) for j in range(ys.shape[1]) for i in range(xs.shape[0])]
        # i = [np.log(fxy(xs[i,j], ys[i,j])/(fx(xs[i,j])*fy(ys[i,j]))) for j in range(ys.shape[1]) for i in range(xs.shape[0])]
        i = np.array(i).reshape(xs.shape[0], ys.shape[1])
        i = i[:-1, :-1]
        i_min, i_max = -np.abs(i).max(), np.abs(i).max()
        c = ax.pcolormesh(xs, ys, i, cmap='RdBu', vmin=i_min, vmax=i_max)
        # set the limits of the plot to the limits of the data
        ax.axis([xs.min(), xs.max(), ys.min(), ys.max()])
        return ax, c


if __name__ == '__main__':
    shape = "linear_asym"
    jk_shape=Additive_noise(200, shape, 0.5)
    data = jk_shape.data
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
        
    #Plot Ground Truth MI
    fig, axs = plt.subplots(1, 2, figsize=(45, 15))
    ax = axs[0]
    ax.scatter(data[:,0], data[:,1], color='r', marker='o')

    ax = axs[1]
    Xmax = max(data[:,0])+1
    Xmin = min(data[:,0])-1
    Ymax = max(data[:,1])+1
    Ymin = min(data[:,1])-1
    x = np.linspace(Xmin, Xmax, 300)
    y = np.linspace(Ymin, Ymax, 300)
    xs, ys = np.meshgrid(x,y)
    ax, c = jk_shape.plot_i(ax, xs, ys)
    fig.colorbar(c, ax=ax)
    ax.set_title("i(X;Y)")
    figName = os.path.join("experiments", "{0}.png".format(jk_shape.shape))
    fig.savefig(figName, bbox_inches='tight')
    plt.close()

    # plt.scatter(data[:,0], data[:,1])
    # print(jk_shape.ground_truth)
    # plt.show()