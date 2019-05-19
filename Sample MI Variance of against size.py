from minee.data.mix_gaussian import MixedGaussian
from minee.data.mix_uniform import MixedUniform
from minee.data.gaussian import Gaussian
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
# get_ipython().run_line_magic('matplotlib', 'inline')


random_seeds = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10
]
num_samples = [
    10,
    20,
    30
]
sample_size = [
    200, 
    400, 
    600,
    800, 
    1000,
    1200,
    1400,
    1600,
    1800,
    2000
    ]
dims = [
    1, 
    5, 
    10,
    20
    ]
rhos = [
    0, 
    0.2, 
    0.4, 
    0.6, 
    0.8, 
    0.9, 
    0.95, 
    0.99
    ]
widths = [
    2, 
    4, 
    6, 
    8, 
    10
    ]
mix = 0.5
for seed in tqdm(random_seeds):
    np.random.seed(seed)
    for num_sample in tqdm(num_samples):
        for dim in tqdm(dims):
            mean = np.zeros(dim*2).tolist()
            for rho in tqdm(rhos):
                var = []
                GT = []
                MIs = []
                sample_sizes = []
                for i in range(len(sample_size)):
                    MG = Gaussian(sample_size=sample_size[i], rho=rho, mean=mean)
                    diff = []
                    GT.append(MG.ground_truth)
                    for _ in range(num_sample):
                        data = MG.data
                        sample_sizes.append(sample_size[i])
                        if dim==1:
                            MI = np.average([MG.I(X[0],X[1]) for X in data])
                        else:
                            MI = np.average([MG.sum_d(X[0:dim],X[-dim:]) for X in data])
                        diff.append((MI-GT)**2)
                        MIs.append(MI)
                    var.append(np.average(diff))


                fig, ax = plt.subplots(1,2, figsize=(22, 7))

                axCur = ax[0]
                axCur.scatter(sample_sizes, MIs, label="estimate")
                axCur.scatter(sample_size, GT, label="Ground Truth")
                axCur.legend()
                axCur.set_title("scatter plot of {} estimate with ground truth vs sample size".format(num_sample))
                axCur.set_xlabel("sample size")
                axCur.set_ylabel("mutual information")

                axCur = ax[1]
                axCur.plot(sample_size, var)
                axCur.set_title("mean square difference of sample MI with ground truth")
                axCur.set_xlabel("sample size")
                axCur.set_ylabel("mean square diff with ground truth")
                plt.savefig("{} {}-dim gaussian samples with rho={} mi plot and mean-square-diff with ground truth seed={}.png".format(num_sample, dim, rho, seed))
                # plt.savefig("/public/hphuang/experiments/var/{} {}-dim gaussian samples with rho={} mi plot and mean-square-diff with ground truth seed={}.png".format(num_sample, dim, rho, seed))
                plt.show()
                plt.close()



        for rho in tqdm(rhos):
            var = []
            GT = []
            MIs = []
            sample_sizes = []
            for i in range(len(sample_size)):
                MG = MixedGaussian(sample_size=sample_size[i], rho1=rho)
                diff = []
                GT.append(MG.ground_truth)
                for _ in range(num_sample):
                    data = MG.data
                    sample_sizes.append(sample_size[i])
                    MI = np.average([MG.sum_d(X[0],X[1]) for X in data])
                    diff.append((MI-GT)**2)
                    MIs.append(MI)
                var.append(np.average(diff))



            fig, ax = plt.subplots(1,2, figsize=(22, 7))

            axCur = ax[0]
            axCur.scatter(sample_sizes, MIs, label="estimate")
            axCur.scatter(sample_size, GT, label="Ground Truth")
            axCur.legend()
            axCur.set_title("scatter plot of {} estimate with ground truth vs sample size".format(num_sample))
            axCur.set_xlabel("sample size")
            axCur.set_ylabel("mutual information")

            axCur = ax[1]
            axCur.plot(sample_size, var)
            axCur.set_title("mean square difference of sample MI with ground truth")
            axCur.set_xlabel("sample size")
            axCur.set_ylabel("mean square diff with ground truth")
            plt.savefig("{} mixed gaussian samples with rho={} mi plot and mean-square-diff with ground truth seed={}.png".format(num_sample, rho, seed))
            # plt.savefig("/public/hphuang/experiments/var/{} mixed gaussian samples with rho={} mi plot and mean-square-diff with ground truth seed={}.png".format(num_sample, rho, seed))
            plt.show()
            plt.close()



        for width in tqdm(widths):
            var = []
            GT = []
            MIs = []
            sample_sizes = []
            for i in range(len(sample_size)):
                MG = MixedUniform(mix, width, width, sample_size=sample_size[i])
                diff = []
                GT.append(MG.ground_truth)
                for _ in range(num_sample):
                    data = MG.data
                    sample_sizes.append(sample_size[i])
                    MI = np.average([MG.sum_d(X[0],X[1]) for X in data])
                    diff.append((MI-GT)**2)
                    MIs.append(MI)
                var.append(np.average(diff))



            fig, ax = plt.subplots(1,2, figsize=(22, 7))

            axCur = ax[0]
            axCur.scatter(sample_sizes, MIs, label="estimate")
            axCur.scatter(sample_size, GT, label="Ground Truth")
            axCur.legend()
            axCur.set_title("scatter plot of {} estimate with ground truth vs sample size".format(num_sample))
            axCur.set_xlabel("sample size")
            axCur.set_ylabel("mutual information")

            axCur = ax[1]
            axCur.plot(sample_size, var)
            axCur.set_title("mean square difference of sample MI with ground truth")
            axCur.set_xlabel("sample size")
            axCur.set_ylabel("mean square diff with ground truth")
            plt.savefig("{} mixed uniform samples with width={} mi plot and mean-square-diff with ground truth seed={}.png".format(num_sample, width, seed))
            # plt.savefig("/public/hphuang/experiments/var/{} mixed uniform samples with width={} mi plot and mean-square-diff with ground truth seed={}.png".format(num_sample, width, seed))
            plt.show()
            plt.close()



