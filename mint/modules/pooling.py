import numpy as np
from .module import Module
from ..utils import functional as F

class _MaxPoolNd(Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation',
                     'return_indices', 'ceil_mode']
    
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(_MaxPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

class MaxPool2d(_MaxPoolNd):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        
        super(MaxPool2d, self).__init__(kernel_size, stride, padding, dilation,
                 return_indices, ceil_mode)
        
        self.init_buffer()
    
    def init_buffer(self):

        self.buffer['in_map'] = None


    def forward(self, input):

        if self.training:
            self.buffer['in_map'] = np.lib.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), 
                    (self.padding, self.padding)), 'constant', constant_values=0.)

        return F.maxpool2d(input, self.kernel_size, self.stride, self.padding, self.dilation,
                self.return_indices, self.ceil_mode).data

    
    def backward(self, input):

        assert self.training
        output = np.zeros(self.buffer['in_map'].shape)
        batch_size, channel, h, w = input.shape
        for b in range(batch_size):
            for c in range(channel):
                for x in range(h):
                    for y in range(w):
                        x_o, y_o =  x * self.stride, y * self.stride
                        hot_idx = np.argmax(self.buffer['in_map'][b, c, 
                                x_o:x_o+self.kernel_size, y_o:y_o+self.kernel_size])
                        x_hot, y_hot = hot_idx // self.kernel_size + x_o, hot_idx % self.kernel_size + y_o
                        output[b, c, x_hot, y_hot] = input[b, c, x, y]
        
        return output.data
            


