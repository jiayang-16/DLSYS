"""Optimization module"""
from collections import defaultdict

from python import needle as ndl
import numpy as np

from python.needle import Tensor


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            if self.weight_decay > 0:
                grad = p.grad.data + self.weight_decay * p.data
            else:
                grad = p.grad.data
            if p not in self.u:
                self.u[p] = (1 - self.momentum) * grad.data
            else:
                self.u[p] = self.momentum * self.u[p].data + (1 - self.momentum) * grad.data
            p.data = p.data - self.lr * self.u[p]


class Adam(Optimizer):
    def __init__(
            self,
            params,
            lr=0.01,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for p in self.params:
            if self.weight_decay > 0:
                grad = p.grad.data + self.weight_decay * p.data
            else:
                grad = p.grad.data
            if p not in self.m:
                self.m[p] = (1 - self.beta1) * grad.data
            else:
                self.m[p] = self.beta1 * self.m[p].data + (1 - self.beta1) * grad.data
            if p not in self.v:
                self.v[p] = (1 - self.beta2) * (grad.data ** 2)
            else:
                self.v[p] = self.beta2 * self.v[p].data + (1 - self.beta2) * (grad.data ** 2)
            unbias_m = self.m[p] / (1 - self.beta1 ** self.t)
            unbias_v = self.v[p] / (1 - self.beta2 ** self.t)
            p.data = p.data - self.lr * unbias_m / (unbias_v ** 0.5 + self.eps)
