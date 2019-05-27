from .model.linear_regression import LinearReg
from .model.mine import Mine
from .model.mine_entropy import Mine_ent
from .model.mine_multitask import MineMultiTask
from .model.kraskov import Kraskov
from .model.cart_regression import cartReg


from .data.mix_gaussian import MixedGaussian
from .data.mix_uniform import MixedUniform
from .data.gaussian import Gaussian
from .data.uniform_mmi import UniformMMI
import math
import os
from datetime import datetime

cpu = 1
batch_size=64
patience=int(250)
lr = 2e-3
moving_average_rate = 1
hidden_size = 100

pop_batch = [
    (200, 200),
    # (32, 32), (32, 8), (32, 2), 
    # (128, 128), (128, 32), (128, 2), (128, 8), 
    # (512, 512), (512, 128), (512, 2), (512, 8), (512, 32), 
    # (2048, 2), (2048, 8), (2048, 32), (2048, 128), (2048, 512), (2048, 2048), 
    # (8192, 2), (8192, 8), (8192, 32), (8192, 128), (8192, 512), (8192, 2048), (8192, 8192)
    ]
# pop_batch = [
#     (32, 2), (32, 8), (32, 32), 
#     (128, 2), (128, 8), (128, 32), (128, 128), 
#     (512, 2), (512, 8), (512, 32), (512, 128), (512, 512), 
#     (2048, 2), (2048, 8), (2048, 32), (2048, 128), (2048, 512), (2048, 2048), 
#     (8192, 2), (8192, 8), (8192, 32), (8192, 128), (8192, 512), (8192, 2048), (8192, 8192)
#     ]

iter_num = int(1e5)
# snapshot = [iter_num//1028, iter_num//512, iter_num//256, iter_num//128, iter_num//64, iter_num//32, iter_num//16, iter_num//8, iter_num//4, iter_num//2]
snapshot = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
video_frames=int(0)
# snapshot = [i for i in range(0, iter_num, 100)]


time_now = datetime.now()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")

# ground truth is plotted in red
model = {
    # 'MINE_direct_hidden_X_2': {
    #     'model': Mine(
    #         lr=lr, 
    #         batch_size=batch_size,  
    #         patience=patience, 
    #         iter_num=iter_num, 
    #         log_freq=int(100), 
    #         avg_freq=int(1), 
    #         ma_rate=moving_average_rate, 
    #         verbose=False,
    #         log=True,
    #         sample_mode='marginal',
    #         earlyStop=False,
    #         hidden_size=hidden_size*2,
    #         iter_snapshot=snapshot,
    #         video_frames=video_frames
    #     ), 
    #     'color': 'magenta'
    # },
    # 'MINE_multi_task': {
    #     'model': MineMultiTask(
    #         lr=lr, 
    #         batch_size=batch_size,  
    #         ref_size=batch_size,
    #         patience=patience, 
    #         iter_num=iter_num, 
    #         log_freq=int(100), 
    #         avg_freq=int(1), 
    #         ma_rate=moving_average_rate, 
    #         verbose=False,
    #         log=True,
    #         sample_mode='unif',
    #         earlyStop=False,
    #         add_mar=True,
    #         hidden_size=hidden_size,
    #         iter_snapshot=snapshot,
    #         video_frames=video_frames
    #     ), 
    #     'color': 'grey'
    # },
    # 'MINE_entropy': {
    #     'model': MineMultiTask(
    #         lr=lr, 
    #         batch_size=batch_size,  
    #         ref_size=batch_size,
    #         patience=patience, 
    #         iter_num=iter_num, 
    #         log_freq=int(100), 
    #         avg_freq=int(1), 
    #         ma_rate=moving_average_rate, 
    #         verbose=False,
    #         log=True,
    #         sample_mode='unif',
    #         earlyStop=False,
    #         add_mar=False,
    #         hidden_size=hidden_size,
    #         iter_snapshot=snapshot,
    #         video_frames=video_frames
    #     ), 
    #     'color': 'purple'
    # },
    'MINE_direct': {
        'model': Mine(
            lr=lr, 
            batch_size=batch_size,  
            patience=patience, 
            iter_num=iter_num, 
            log_freq=int(100), 
            avg_freq=int(1), 
            ma_rate=moving_average_rate, 
            verbose=False,
            log=True,
            sample_mode='marginal',
            earlyStop=False,
            hidden_size=hidden_size,
            iter_snapshot=snapshot,
            video_frames=video_frames
        ), 
        'color': 'orange'
    },
}

# n_samples = 6400
n_samples = batch_size * 20
# rhos = [ 0, 0.2, 0.6 ,0.8, 0.9, 0.99 ]
rhos = [0.9]
widths = list(range(2, 12, 4))


data = {
    'Mixed Gaussian': {
        'model': MixedGaussian,
        'kwargs': [  # list of params
            {
                'n_samples':n_samples, 
                'mean1':0, 
                'mean2':0, 
                'rho1': rho, 
                'rho2': -rho,
            } for rho in rhos
        ], 
        'varying_param_name': 'rho1', # the parameter name which denotes the x-axis of the plot
        'x_axis_name': 'correlation', 
    }, 
    'Gaussian': {
        'model': Gaussian, 
        'kwargs': [
            {
                'n_samples':n_samples, 
                'mean1':0, 
                'mean2':0, 
                'rho': rho,
            } for rho in rhos
        ], 
        'varying_param_name': 'rho', 
        'x_axis_name': 'correlation', 
    },
    'Mixed Uniform': {
        'model': MixedUniform, 
        'kwargs': [
            {
                'n_samples':n_samples, 
                'width_a': width, 
                'width_b': width, 
                'mix': 0.5
            } for width in widths
        ], 
        'varying_param_name': 'width_a', 
        'x_axis_name': 'width'
    }, 
    # {
    #     'name': 'Examples', 
    #     'model': XX(
    #         n_samples=XX
    #         rho=XX
    #     )
    # }, 
}


n_datasets = len(data)
# n_columns = max([len(rhos), len(widths)])
