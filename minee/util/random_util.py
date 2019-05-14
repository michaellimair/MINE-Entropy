import numpy as np

def resample(data,batch_size,replace=False):
    index = np.random.choice(range(data.shape[0]), size=batch_size, replace=replace)
    batch = data[index]
    return batch

def uniform_sample(data_min,data_max,batch_size):
    return (data_max - data_min) * np.random.random((batch_size, data_min.shape[0])) + data_min