## Convolutional layer
import numpy as np
from .module import Module
from ..utils import functional as F
from ..utils import init as init
from ..parameter import Parameter
from .helper import _pair

class _ConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,  groups, 
                 bias, padding_mode):
        
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight = Parameter((in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter((out_channels, in_channels // groups, *kernel_size))
        self.bias = None if not bias else Parameter((out_channels, ), dtype=np.float32)

        self.init_buffer()
        self.reset_parameters()
    
    def init_buffer(self):

        self.buffer['in_map'] = None
        ## gradient for weights
        self.buffer['grad_weight'] = None
        self.buffer['moment_weight'] = 0.
        ## gradient for bias
        self.buffer['grad_bias'] = None
        self.buffer['moment_bias'] = 0.
        ## gradient to be propagated to the previous layer
        # self.buffer['grad_back'] = None
    
    def reset_parameters(self):

        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zero_(self.bias)


class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, 
                stride, padding, dilation, False, _pair(0), groups, bias, padding_mode)
    
    def forward(self, input):

        if self.training:
            self.buffer['in_map'] = np.lib.pad(input, ((0, 0), (0, 0), 
                (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), 
                'constant', constant_values=0.)
        
        # bias = self.bias.data if self.bias is not None else None
        return F.conv2d(input, self.weight, bias=self.bias, stride=self.stride, 
                padding=self.padding, dilation=self.dilation, groups=self.groups, 
                padding_mode=self.padding_mode)
    
    def backward(self, input):

        assert self.training
        if self.bias is not None:
            self.buffer['grad_bias'] = np.sum(np.sum(np.sum(input, axis=-1), axis=-1), axis=0)
        input = F._insert_zeros_in_rows_and_cols(input, stride=self.stride)
        ## in_map is of size (N, in_C, H, W)
        ## input is of size (N, out_C, H', W')
        self.buffer['grad_weight'] = np.swapaxes(F.conv2d(np.swapaxes(self.buffer['in_map'], 0, 1), 
                np.swapaxes(input, 0, 1), padding=(0, 0), bias=None), 0, 1) 

        return F._remove_padding(F.conv2d(input, np.swapaxes(self.weight, 0, 1), None, 
                padding=(self.kernel_size[0] - 1, self.kernel_size[1] - 1)), self.padding)
        

        

            