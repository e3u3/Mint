## Initialization for parameters
import math
import numpy as np


def uniform_(param, low=0., high=1.):
    ## uniform dist from low to high
    for i in range(param.shape[0]):
        param[i] = np.random.uniform(low=low, high=high, size=param[i].shape)    


def normal_(param, mean=0., std=1.):
    ## normal dist with mean (default zero) and standard deviation (default one)
    for i in range(param.shape[0]):
        param[i] = np.random.normal(loc=mean, scale=mean, size=param[i].shape)


def _calculate_fan_in_and_fan_out(param):

    if len(param.shape) == 1:
        fan_in, fan_out = param.shape[0], 1 
    elif len(param.shape) == 2:
        fan_in, fan_out = param.shape[1], param.shape[0]
    else:
        num_in_maps, num_out_maps = param.shape[1], param.shape[0]
        receptive_field_size = param[0][0].size
        fan_in, fan_out = num_in_maps * receptive_field_size, num_out_maps * receptive_field_size
    return fan_in, fan_out
        

def xavier_uniform_(param, gain=1.):

    fan_in, fan_out = _calculate_fan_in_and_fan_out(param)
    std = gain * math.sqrt(2. / float(fan_in + fan_out))
    bound = math.sqrt(3.) * std

    uniform_(param, low=-bound, high=bound)


def xavier_normal_(param, gain=1.):

    fan_in, fan_out = _calculate_fan_in_and_fan_out(param)
    std = gain * math.sqrt(2. / float(fan_in + fan_out))

    normal_(param, mean=0., std=std)

def constant_(param, val=0.):

    for i in range(param.shape[0]):
        param[i] = np.zeros(param[i].shape, dtype=np.float32) + val


def zero_(param):

    for i in range(param.shape[0]):
        param[i] = np.zeros(param[i].shape, dtype=np.float32)


def one_(param):

    constant_(param, val=1.)