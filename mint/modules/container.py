## Container of modules
import operator
from itertools import islice
from collections import OrderedDict
from .module import Module

class Container(Module):

    def __init__(self):

        super(Container, self).__init__()
        self.modules = OrderedDict()

    def add_module(self, key, value):

        ## TODO: invalid check
        self.modules[key] = value


class Sequential(Container):

    def __init__(self, *args):

        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, value in args[0].items():
                self.add_module(key, value)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    
    def _get_item_by_idx(self, iterator, idx):

        idx = operator.index(idx)
        
        return next(islice(iterator, idx, None))
    
    def __getitem__(self, idx):

        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self.modules.items())[idx]))
        else:
            return self._get_item_by_idx(self.modules.values(), idx)
    
    def __len__(self):

        return len(self.modules)
    
    def forward(self, input):

        for module in self.modules.values():
            # print(input)
            input = module.forward(input)
        
        return input
    
    def backward(self, input):

        assert self.training
        for module in list(self.modules.values())[::-1]:
            input = module.backward(input)
    
    def train(self):

        self.training = True
        for module in self.modules.values():
            module.train()
    
    def eval(self):

        self.training = False
        for module in self.modules.values():
            module.eval()