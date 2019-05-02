## Flattern
import numpy as np
from .module import Module
from ..utils import functional as F

class Flat(Module):

    def __init__(self):

        super(Flat, self).__init__()
        self.in_size = None
    
    def forward(self, input):

        if self.training:
            self.in_size = input.shape
        return F.flat(input)
    
    def backward(self, input):

        return np.reshape(input, self.in_size)
