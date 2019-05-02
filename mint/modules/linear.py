## Dense layer
import numpy as np
from .module import Module
from ..utils import functional as F
from ..utils import init as init
from ..parameter import Parameter

class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):

        super(Linear, self).__init__()
        self.weight = Parameter((out_features, in_features), dtype=np.float32)
        self.bias = None if not bias else Parameter((out_features,), dtype=np.float32)

        self.init_buffer()
        self.reset_parameters()
    
    def init_buffer(self):
        
        ## input feature in the last forward
        self.buffer['in_feature'] = None
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
    
    def forward(self, input):

        if self.training: 
            self.buffer['in_feature'] = input

        # bias = self.bias.data if self.bias is not None else None
        return F.linear(input, self.weight, bias=self.bias)
    
    def backward(self, input):

        assert self.training
        in_feature = self.buffer['in_feature']
        self.buffer['grad_weight'] = np.matmul(input.T, in_feature)
        # self.buffer['in_feature'] = None    
        if self.bias is not None:
            self.buffer['grad_bias'] = np.sum(input, axis=0)
        # self.buffer['grad_back'] = np.matmul(input, self.weight)
        return np.matmul(input, self.weight)