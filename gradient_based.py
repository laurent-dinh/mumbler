import theano
import theano.tensor as T
import numpy as np
floatX = "float32"

class SGD_momentum(object):
    def __init__(self, params, mode):
        self.params = params
        self.mode = mode
        self.params_momentum = []
        if hasattr(self.params, "__iter__"):
            for param in self.params:
                param_momentum = np.zeros_like(param.get_value(), dtype=floatX)
                param_momentum = theano.shared(value = param_momentum, \
                                               allow_downcast = True)
                if param.name is not None: param_momentum.name = param.name+"_momentum"
                self.params_momentum.append(param_momentum)
            
        else:
            self.params_momentum = np.zeros_like(self.params.get_value(), dtype=floatX)
            self.params_momentum = theano.shared(value = self.params_momentum, \
                                                 allow_downcast = True)
            if self.params.name is not None: self.params.name = self.params.name+"_momentum"
    
    def get_update(self, cost, learning_rate, momentum_coeff, max_norm = None):
        
        grad_cost = T.grad(cost, self.params)
        updates = {}
        
        if hasattr(self.params, "__iter__"):
            for (param, gparam, mparam) in zip(self.params, grad_cost,
            self.params_momentum):
                if param.name is not None:
                    print param.name
                    print mparam.name
                
                next_momentum = momentum_coeff*mparam - gparam
                next_param = param + learning_rate*next_momentum
                
                updates[mparam] = T.cast(next_momentum, floatX)
                updates[param] = T.unbroadcast(T.cast(next_param, floatX))
            
            next_momentum_norm = T.sqrt(reduce(lambda x,y:x + y, \
            map(lambda x:(x**2).sum(), \
            [updates[mparam] for mparam in self.params_momentum])))
            
            for (param, gparam, mparam) in zip(self.params, grad_cost,
            self.params_momentum):
                next_momentum = momentum_coeff*mparam - gparam
                if self.mode == "normalized":
                    next_momentum = next_momentum/next_momentum_norm
                elif self.mode == "clipped":
                    assert max_norm is not None
                    next_momentum = T.switch((next_momentum_norm < max_norm), \
                                    next_momentum, \
                                    next_momentum*(max_norm/next_momentum_norm))
                next_param = param + learning_rate*next_momentum
                
                updates[mparam] = T.cast(next_momentum, floatX)
                updates[param] = T.unbroadcast(T.cast(next_param, floatX))
            
        else:
            next_momentum = momentum_coeff*self.params_momentum - learning_rate*grad_cost
            next_param = self.params + next_momentum
            next_momentum_norm = T.sqrt((next_momentum**2).sum())
            if self.mode == "normalized":
                assert max_norm is not None
                next_momentum = next_momentum*(max_norm/next_momentum_norm)
            elif self.mode == "clipped":
                assert max_norm is not None
                next_momentum = T.switch((next_momentum_norm < max_norm), \
                                next_momentum, \
                                next_momentum*(max_norm/next_momentum_norm))
            
            updates[self.params_momentum] = T.cast(next_momentum, floatX)
            updates[self.params] = T.cast(next_param, floatX)
        
        return (updates, next_momentum_norm)

