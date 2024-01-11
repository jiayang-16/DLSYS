"""The module.
"""
from functools import reduce
from typing import List, Callable, Any
from python.needle.autograd import Tensor
from python.needle import ops
import python.needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(
            init.kaiming_uniform(out_features, 1, device=device, dtype=dtype,
                                 requires_grad=True).transpose()) if bias else None

    def forward(self, X: Tensor) -> Tensor:
        out = X.matmul(self.weight)
        if self.bias:
            out += self.bias.broadcast_to(out.shape)
        return out


class Flatten(Module):
    def forward(self, X):
        from operator import mul
        size = reduce(mul, X.shape)
        return ops.reshape(X, (X.shape[0], size // X.shape[0]))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for module in self.modules:
            out = module(out)
        return out


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        one_hot = init.one_hot(logits.shape[1], y)
        return (ops.summation(ops.logsumexp(logits, axes=(1,))) - ops.summation(one_hot * logits)) / logits.shape[0]


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, requires_grad=True))
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            ex = (ops.summation(x, (0,)) / x.shape[0])
            varx = (ops.summation((x - ex.broadcast_to(x.shape)) ** 2, (0,)) / x.shape[0])
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * ex.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * varx.data
            return self.weight.broadcast_to(x.shape) * (x - ex.broadcast_to(x.shape)) / (
                    varx.broadcast_to(x.shape) + self.eps) ** 0.5 + self.bias.broadcast_to(x.shape)
        else:
            return self.weight.broadcast_to(x.shape) * (x - self.running_mean.broadcast_to(x.shape)) / (
                    self.running_var.broadcast_to(x.shape) + self.eps) ** 0.5 + self.bias.broadcast_to(x.shape)


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        ex = (ops.summation(x, (1,)) / x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        varx = (ops.summation((x - ex) ** 2, (1,)) / x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * (x - ex) / (varx + self.eps) ** 0.5 + self.bias.broadcast_to(x.shape)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return init.randb(*x.shape, p=1 - self.p) * x / (1 - self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
