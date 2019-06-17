import numpy as np
random_seed = 0
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


cpu = 1
batch_size=50
lr = 1e-3
moving_average_rate = 0.1
hidden_size = 100


pop_batch = [
    # (200, 50), 
    # (200, 100), 
    (200, 200),
    
    ]

iter_num = int(1e3)
record_rate = int(250)
# snapshot = [iter_num//1028, iter_num//512, iter_num//256, iter_num//128, iter_num//64, iter_num//32, iter_num//16, iter_num//8, iter_num//4, iter_num//2]
# snapshot = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
snapshot = list(range(record_rate, iter_num, record_rate))
video_frames=int(0)
# snapshot = [i for i in range(0, iter_num, 100)]


time_now = datetime.now()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")

# ground truth is plotted in red
model = {
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
            rep=2,
            fix_ref_est=False,
            archive_length=100,
            estimate_rate=1,
            video_frames=10
        ), 
        'color': 'purple'
    },
    # 'MINE_hidden=100': {
    #     'model': Mine(
    #         lr=lr, 
    #         batch_size=batch_size,
    #         ma_rate=moving_average_rate,
    #         hidden_size=hidden_size,
    #         snapshot=snapshot,
    #         iter_num=iter_num,
    #         log=True,
    #         verbose=False,
    #         full_ref=False,
    #         load_dict=True,
    #         ref_factor=1,
    #         rep=2,
    #         fix_ref_est=False,
    #         archive_length=100,
    #         full_batch_ref=True,
    #         estimate_rate=1,
    #         video_frames=10
    #     ),
    #     'color': 'orange'
    # },
    # 'MINE_hidden=300': {
    #     'model': Mine(
    #         lr=lr, 
    #         batch_size=batch_size,
    #         ma_rate=moving_average_rate,
    #         hidden_size=hidden_size*3,
    #         snapshot=snapshot,
    #         iter_num=iter_num,
    #         log=True,
    #         verbose=False,
    #         full_ref=False,
    #         load_dict=True,
    #         ref_factor=1,
    #         rep=2,
    #         fix_ref_est=False,
    #         archive_length=5000
    #     ),
    #     'color': 'magenta'
    # },
    # 'MINEE_ref=2x': {
    #     'model': Minee(
    #         lr=lr, 
    #         batch_size=batch_size,
    #         hidden_size=hidden_size,
    #         snapshot=snapshot,
    #         iter_num=iter_num,
    #         log=True,
    #         verbose=False,
    #         ref_window_scale=1,
    #         ref_batch_factor=2,
    #         load_dict=True
    #     ), 
    #     'color': 'purple'
    # },
    # 'MINEE_ref=4x': {
    #     'model': Minee(
    #         lr=lr, 
    #         batch_size=batch_size,
    #         hidden_size=hidden_size,
    #         snapshot=snapshot,
    #         iter_num=iter_num,
    #         log=True,
    #         verbose=False,
    #         ref_window_scale=1,
    #         ref_batch_factor=4,
    #         load_dict=True
    #     ), 
    #     'color': 'purple'
    # },
    # 'MINEE_ref=16x': {
    #     'model': Minee(
    #         lr=lr, 
    #         batch_size=batch_size,
    #         hidden_size=hidden_size,
    #         snapshot=snapshot,
    #         iter_num=iter_num,
    #         log=True,
    #         verbose=False,
    #         ref_window_scale=1,
    #         ref_batch_factor=16,
    #         load_dict=True
    #     ), 
    #     'color': 'purple'
    # },
    # 'MINE_hidden=100': {
    #     'model': Mine(
    #         lr=lr, 
    #         batch_size=batch_size,
    #         ma_rate=moving_average_rate,
    #         hidden_size=hidden_size,
    #         snapshot=snapshot,
    #         iter_num=iter_num,
    #         log=True,
    #         verbose=False,
    #         full_ref=False,
    #         load_dict=True
    #     ),
    #     'color': 'orange'
    # },
    # 'MINE': {
    #     'model': Mine(
    #         lr=lr, 
    #         batch_size=batch_size,
    #         ma_rate=moving_average_rate,
    #         hidden_size=hidden_size*3,
    #         snapshot=snapshot,
    #         iter_num=iter_num,
    #         log=True,
    #         verbose=False,
    #         full_ref=False,
    #         load_dict=True,
    #         ref_factor=1,
    #         rep=2,
    #         fix_ref_est=False,
    #         archive_length=5000
    #     ),
    #     'color': 'magenta'
    # },
}

sample_size = 200
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
# rhos = [0.9]
widths = [
    2,
    # 4,
    # 6,
    # 8,
    # 10
]


data = {
    # 'Mixed Gaussian X': {
    #     'model': MixedGaussian,
    #     'kwargs': [  # list of params
    #         {
    #             'sample_size':sample_size, 
    #             'mean1':0, 
    #             'mean2':0, 
    #             'rho1': rho, 
    #             'rho2': -rho,
    #         } for rho in rhos
    #     ], 
    #     'varying_param_name': 'rho1', # the parameter name which denotes the x-axis of the plot
    #     'x_axis_name': 'correlation', 
    # }, 
    'Mixed Gaussian +': {
        'model': MixedGaussian,
        'kwargs': [  # list of params
            {
                'sample_size':sample_size, 
                'mean1':0, 
                'mean2':0, 
                'rho1': rho, 
                'rho2': -rho,
                'theta': np.pi/4.
            } for rho in rhos
        ], 
        'varying_param_name': 'rho1', # the parameter name which denotes the x-axis of the plot
        'x_axis_name': 'correlation', 
    }, 
    # 'Gaussian': {
    #     'model': Gaussian, 
    #     'kwargs': [
    #         {
    #             'sample_size':sample_size, 
    #             'rho': rho,
    #             'mean':[0,0], 
    #         } for rho in rhos
    #     ], 
    #     'varying_param_name': 'rho', 
    #     'x_axis_name': 'correlation', 
    # },
    # '2-Dimension Gaussian': {
    #     'model': Gaussian, 
    #     'kwargs': [
    #         {
    #             'sample_size':sample_size, 
    #             'rho': rho,
    #             'mean':np.zeros(4).tolist(), 
    #         } for rho in rhos
    #     ], 
    #     'varying_param_name': 'rho', 
    #     'x_axis_name': 'correlation', 
    # },
    # '4-Dimension Gaussian': {
    #     'model': Gaussian, 
    #     'kwargs': [
    #         {
    #             'sample_size':sample_size, 
    #             'rho': rho,
    #             'mean':np.zeros(8).tolist(), 
    #         } for rho in rhos
    #     ], 
    #     'varying_param_name': 'rho', 
    #     'x_axis_name': 'correlation', 
    # },
    # '4-Dimension Gaussian': {
    #     'model': Gaussian, 
    #     'kwargs': [
    #         {
    #             'sample_size':sample_size, 
    #             'rho': rho,
    #             'mean':np.zeros(8).tolist(), 
    #         } for rho in rhos
    #     ], 
    #     'varying_param_name': 'rho', 
    #     'x_axis_name': 'correlation', 
    # },
    # '20-Dimension Gaussian': {
    #     'model': Gaussian, 
    #     'kwargs': [
    #         {
    #             'sample_size':sample_size, 
    #             'rho': rho,
    #             'mean':np.zeros(40).tolist(), 
    #         } for rho in rhos
    #     ], 
    #     'varying_param_name': 'rho', 
    #     'x_axis_name': 'correlation', 
    # },
    # 'Mixed Uniform': {
    #     'model': MixedUniform, 
    #     'kwargs': [
    #         {
    #             'sample_size':sample_size, 
    #             'width_a': width, 
    #             'width_b': width, 
    #             'mix': 0.5
    #         } for width in widths
    #     ], 
    #     'varying_param_name': 'width_a', 
    #     'x_axis_name': 'width'
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
