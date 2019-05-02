## Stochastic gradient descent
from .optimizer import Optimizer

class SGD(Optimizer):

    def __init__(self, modules, lr=1e-3, moment=0, dampening=0, weight_decay=0, nesterov=False):

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        defaults = dict(lr=lr, moment=moment, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        # print(modules, defaults)
        super(SGD, self).__init__(modules, defaults)
    
    def _step(self, module, key):

        lr = self.defaults['lr']
        weight_decay = self.defaults['weight_decay']
        momentum = self.defaults['moment']
        dampening = self.defaults['dampening']
        nesterov = self.defaults['nesterov']
        grad_name = 'grad_' + str(key)
        moment_name = 'moment_' + str(key)
        grad = module.buffer[grad_name]
        if weight_decay != 0:
             grad += module.__dict__[key] * weight_decay
        ## compute momentum cumulated
        if momentum != 0:
            module.buffer[moment_name] *= momentum
            grad += module.buffer[moment_name]
            module.buffer[moment_name] = grad
            # print(module.buffer[moment_name])
        ## TODO: nesterov
        ## update params
        module.__dict__[key] -= lr * grad