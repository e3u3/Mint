## Activation functions
from .module import Module
from ..utils import functional as F

class ReLU(Module):

    def __init__(self, in_place=False):

        super(ReLU, self).__init__()
        self.in_place = in_place

        self.init_buffer()

    def init_buffer(self):

        self.buffer['activated'] = None
    
    def forward(self, input):

        if self.training and self.in_place:
            self.buffer['activated'] = input >= 0
            # print(self.buffer['activated'])

        return F.relu(input)
    
    def backward(self, input):

        assert self.training
        if self.in_place:
            input *= self.buffer['activated']
        
        return input