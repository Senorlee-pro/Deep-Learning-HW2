from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocities = []
        for layer in self.model.layers:
            if layer.optimizable:
                v = {}
                for key in layer.params.keys():
                    v[key] = np.zeros_like(layer.params[key])
                self.velocities.append(v)

    def step(self):
        v_idx = 0
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    self.velocities[v_idx][key] *= self.mu
                    self.velocities[v_idx][key] += layer.grads[key]
                    layer.params[key] -= self.init_lr * self.velocities[v_idx][key]
                v_idx += 1