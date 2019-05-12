## Optimizer base class
from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import chain
from ..modules import Module 
import numpy as np


class Optimizer(object):

    def __init__(self, modules, defaults):

        # print(modules, defaults)
        self.modules = OrderedDict()
        self.defaults = defaults
        if isinstance(modules, OrderedDict):
            self.modules = modules
        else:
            raise TypeError("modules should be a dict or sole module.")
    
    def zero_grad(self):
        
        for module in self.modules.values():
            for key in module.parameters():
                name = 'grad_' + str(key)
                if module.buffer[name] is not None:
                    module.buffer[name] = np.zeros(module.buffer[name].shape)
    
    def step(self):

        for module in self.modules.values():
            for key in module.parameters():
                self._step(module, key)
    
    def _step(self, module, key):

        raise NotImplementedError


    
    


