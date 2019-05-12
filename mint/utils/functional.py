## Functions
import math
import numpy as np

def linear(input, weight, bias=None):

    output = np.matmul(input, weight.T)
    if bias is not None:
        output += bias
    
    return output

def conv2d(input, weight, bias=None, stride=(1, 1), padding=(1, 1), dilation=1, groups=1, padding_mode='zeros'):
    
    input = np.pad(input, ((0, 0), (0, 0), 
            (padding[0], padding[0]), (padding[1], padding[1])), 
            'constant', constant_values=0.)
    batch_size, _, height, width = input.shape
    out_c, _, k1, k2 = weight.shape
    assert (height - k1) % stride[0] == 0
    assert (width - k2) % stride[1] == 0
    output = np.zeros((batch_size, out_c, (height - k1) // stride[0] + 1, (width - k2) // stride[1] + 1))
    for b in range(batch_size):
        for c in range(out_c):
            for x in range(output.shape[2]):
                for y in range(output.shape[3]):
                    x_o, y_o =  x * stride[0], y * stride[1]
                    output[b, c, x, y] =  np.sum(input[b, :,x_o:x_o+k1, y_o:y_o+k2] * weight[c, :, :, :])
    if bias is not None:
        for c in range(out_c):
            output[:, c, :, :] += bias[c]
    
    return output

def _insert_zeros_in_rows_and_cols(input, stride=(1, 1)):

    ## insert zeros between rows and cols
    batch_size, channel, height, width = input.shape
    new_input = np.zeros((batch_size, channel, (stride[0] - 1) * (height - 1) + height, 
                         (stride[1] - 1) * (width - 1) + width))
    for b in range(batch_size):
        for c in range(channel):
            for x in range(0, new_input.shape[2], stride[0]):
                for y in range(0, new_input.shape[3], stride[1]):
                    new_input[b, c, x, y] = input[b, c, x // stride[0], y // stride[1]]
    # input = None
    
    return new_input

def _remove_padding(input, padding=(1, 1)):

    h, w = input.shape[2], input.shape[3]
    return  input[:, :, padding[0]:h-padding[0], padding[1]:h-padding[1]]

def relu(input):

    if np.isnan(np.sum(input)):
        print(input)

    return input * (input > 0)

def flat(input):

    return np.reshape(input, (input.shape[0], -1))

def softmax(input):

    assert len(input.shape) == 2
    batch_size, num_class = input.shape
    output = np.zeros(input.shape)
    for b in range(batch_size):
        total = 0.
        for c in range(num_class):
            # if abs(output[b, c]) < EXP_BOUND:
            output[b, c] = math.exp(input[b, c])
            # else:
            #     output[b, c] = 1 if output[b, c] > 0 else EPSILON
            total += output[b, c]
        output[b, :] /= total
    
    return output

def maxpool2d(input, kernel_size, stride, padding, dilation, ceil_mode, require_hot_map=False):

    ## TODO: dilation/ceil/return_indices
    input = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
            'constant', constant_values=0.)
    batch_size, channel, h, w = input.shape
    output = np.zeros((batch_size, channel, (h - kernel_size) // stride + 1, 
            (w - kernel_size) // stride + 1), dtype=np.float32)
    hot_map = output.copy() if require_hot_map else None
    for b in range(batch_size):
        for c in range(channel):
            for x in range(output.shape[2]):
                for y in range(output.shape[3]):
                    x_o, y_o =  x * stride, y * stride
                    output[b, c, x, y] = np.max(input[b, c, x_o:x_o+kernel_size, y_o:y_o+kernel_size])
                    if require_hot_map:
                        hot_map[b, c, x, y] = np.argmax(input[b, c, x_o:x_o+kernel_size, y_o:y_o+kernel_size])
    if require_hot_map:
        return output, hot_map
    
    return output
