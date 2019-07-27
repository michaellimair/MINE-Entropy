import numpy as np
random_seed = 2
np.random.seed(seed=random_seed)
import torch
# torch.manual_seed(seed=random_seed)
from .model.mine import Mine
from .model.minee import Minee
from .model.kraskov import Kraskov


from .data.mix_gaussian import MixedGaussian
from .data.mix_uniform import MixedUniform
from .data.gaussian import Gaussian
from .data.uniform_mmi import UniformMMI
import math
import os
from datetime import datetime
import numpy as np

cpu = 20
batch_size=50
lr = 5e-5
moving_average_rate = 0.1
hidden_size = 300

pop_batch = [
    (1000, 1000)
    ]

iter_num = int(1e6)
record_rate = int(100)
snapshot = (record_rate*(2**np.arange(int(np.log2(iter_num//record_rate))))).tolist()
video_frames=int(0)


time_now = datetime.now()
# output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
output_path = os.path.join("/public/hphuang", "experiments")

# ground truth is plotted in red
model = {
    'MINEE_w_Gaussian_ref': {
        'model': Minee(
            lr=lr, 
            batch_size=batch_size,
            hidden_size=hidden_size,
            snapshot=snapshot,
            iter_num=iter_num,
            log=True,
            verbose=False,
            ref_window_scale=1,
            ref_batch_factor=1,
            load_dict=True,
            rep=10,
            fix_ref_est=False,
            archive_length=500,
            estimate_rate=1,
            video_rate=0,
            infinite_sample=True,
            gaussian_ref=True,
            gaussian_ref_var=3
        ), 
        'color': 'purple'
    },
    'MINEE': {
        'model': Minee(
            lr=lr, 
            batch_size=batch_size,
            hidden_size=hidden_size,
            snapshot=snapshot,
            iter_num=iter_num,
            log=True,
            verbose=False,
            ref_window_scale=1,
            ref_batch_factor=1,
            load_dict=True,
            rep=10,
            fix_ref_est=False,
            archive_length=500,
            estimate_rate=1,
            video_rate=0,
            infinite_sample=True,
            gaussian_ref=False,
            gaussian_ref_var=3
        ), 
        'color': 'purple'
    },
    'MINE': {
        'model': Mine(
            lr=lr, 
            batch_size=batch_size,
            ma_rate=moving_average_rate,
            hidden_size=hidden_size,
            snapshot=snapshot,
            iter_num=iter_num,
            log=True,
            verbose=False,
            full_ref=False,
            load_dict=True,
            ref_factor=1,
            rep=10,
            fix_ref_est=False,
            archive_length=500,
            full_batch_ref=False,
            estimate_rate=1,
            video_rate=0,
            infinite_sample=True
        ),
        'color': 'magenta'
    },
}

sample_size = 400
rhos = [ 
    # 0, 
    # 0.2, 
    # 0.4, 
    # 0.6, 
    # 0.8, 
    0.9, 
    # 0.95, 
    # 0.99 
    ]
widths = [
    2,
    4,
    6,
    8,
    10
]


data = {
    '6-Dimension Mixed Gaussian +': {
        'model': MixedGaussian,
        'kwargs': [  # list of params
            {
                'sample_size':sample_size, 
                'mean1':[0, 0], 
                'mean2':[0, 0], 
                'rho1': rho, 
                'rho2': -rho,
                'dim': 6,
                'theta': np.pi/4
            } for rho in rhos
        ], 
        'varying_param_name': 'rho1', # the parameter name which denotes the x-axis of the plot
        'x_axis_name': 'correlation', 
    }, 
    '6-Dimension Mixed Gaussian + separate': {
        'model': MixedGaussian,
        'kwargs': [  # list of params
            {
                'sample_size':sample_size, 
                'mean1':[0, 2], 
                'mean2':[0, -2], 
                'rho1': rho, 
                'rho2': -rho,
                'dim': 6,
                'theta': np.pi/4
            } for rho in rhos
        ], 
        'varying_param_name': 'rho1', # the parameter name which denotes the x-axis of the plot
        'x_axis_name': 'correlation', 
    }, 
    '6-Dimension Mixed Gaussian X': {
        'model': MixedGaussian,
        'kwargs': [  # list of params
            {
                'sample_size':sample_size, 
                'mean1':[0, 0], 
                'mean2':[0, 0], 
                'rho1': rho, 
                'rho2': -rho,
                'dim': 6
            } for rho in rhos
        ], 
        'varying_param_name': 'rho1', # the parameter name which denotes the x-axis of the plot
        'x_axis_name': 'correlation', 
    }, 
    '6-Dimension Mixed Gaussian X separate': {
        'model': MixedGaussian,
        'kwargs': [  # list of params
            {
                'sample_size':sample_size, 
                'mean1':[0, 2], 
                'mean2':[0, -2], 
                'rho1': rho, 
                'rho2': -rho,
                'dim': 6
            } for rho in rhos
        ], 
        'varying_param_name': 'rho1', # the parameter name which denotes the x-axis of the plot
        'x_axis_name': 'correlation', 
    }, 
    '6-Dimension Gaussian': {
        'model': Gaussian, 
        'kwargs': [
            {
                'sample_size':sample_size, 
                'rho': rho,
                'mean':np.zeros(12).tolist(), 
            } for rho in rhos
        ], 
        'varying_param_name': 'rho', 
        'x_axis_name': 'correlation', 
    },
    # '6-Dimension Gaussian': {
    #     'model': Gaussian, 
    #     'kwargs': [
    #         {
    #             'sample_size':sample_size, 
    #             'rho': rho,
    #             'mean':np.zeros(12).tolist(), 
    #         } for rho in rhos
    #     ], 
    #     'varying_param_name': 'rho', 
    #     'x_axis_name': 'correlation', 
    # },
    # {
    #     'name': 'Examples', 
    #     'model': XX(
    #         sample_size=XX
    #         rho=XX
    #     )
    # }, 
}


n_datasets = len(data)
# n_columns = max([len(rhos), len(widths)])
