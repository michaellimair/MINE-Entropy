import numpy as np
random_seed = 2
np.random.seed(seed=random_seed)

def resample(data,batch_size,replace=False):
    index = np.random.choice(range(data.shape[0]), size=batch_size, replace=replace)
    batch = data[index]
    return batch

# def uniform_sample(data_min,data_max,batch_size):
#     return (data_max - data_min) * np.random.random((batch_size, data_min.shape[0])) + data_min

def uniform_sample(data, batch_size, window_scale = 1):
    dim = data.shape[1]
    if dim == 1:
        data_min = data.min()
        data_max = data.max()
    else:
        data_min = data.min(axis=0)[0]
        data_max = data.max(axis=0)[0]
    if window_scale != 1:
        data_med = (data_max + data_min) / 2
        data_rad = (data_max - data_min) / 2
        data_min = data_med - window_scale * data_rad
        data_max = data_med + window_scale * data_rad
    return (data_max - data_min) * np.random.random((batch_size, dim)) + data_min