## Base class
from collections import OrderedDict
from ..parameter import Parameter

class Module(object):

    def __init__(self):

        self.training = True
        self.buffer = OrderedDict()
        self._parameters = set()
        self.param_got = False
    
    def reset_parameters(self):

        raise NotImplementedError
    
    def init_buffer(self):

        raise NotImplementedError

    def forward(self, input):

        raise NotImplementedError
    
    def backward(self, input):

        raise NotImplementedError
    
    def train(self):

        self.training = True
    
    def eval(self):

        self.training = False

    def register_parameters(self):

        for key, value in self.__dict__.items():
            if isinstance(value, Parameter):
                self._parameters.add(key)
        self.param_got = True
    
    def parameters(self):

        if not self.param_got:
            self.register_parameters()
        
        return self._parameters
        
