## Loss functions
import math
import numpy as np
from .module import Module
from ..utils import functional as F

class _Loss(Module):

    def __init__(self, reduction='mean'):

        super(_Loss, self).__init__()
        self.reduction = reduction


class CrossEntropy(_Loss):

    def __init__(self, reduction='mean'):

        super(CrossEntropy, self).__init__(reduction)

        self.init_buffer()

    def init_buffer(self):

        self.buffer['in_feature'] = None
        self.buffer['softmax'] = None
        self.buffer['target'] = None
    
    def forward(self, input, target):

        softmax = F.softmax(input)
        if self.training:
            self.buffer['in_feature'] = input
            self.buffer['softmax'] = softmax
            self.buffer['target'] = target
        output = 0.
        batch_size = target.shape[0]
        for b in range(batch_size):
            output -= math.log(softmax[b, int(target[b])])
        if self.reduction == 'mean':
            return output / float(batch_size)
        
        return output
    
    def backward(self, input=None):

        assert self.training
        grad = np.zeros(self.buffer['softmax'].shape)
        batch_size, num_class = self.buffer['target'].shape[0], grad.shape[1]
        for b in range(batch_size):
            correct_c = int(self.buffer['target'][b])
            for c in range(num_class):
                softmax_v = float(self.buffer['softmax'][b, c])
                grad[b, c] = softmax_v - 1 if c == correct_c else softmax_v
        if input is not None:
            grad *= input
        if self.reduction == 'mean':
            return grad / batch_size

        return grad 