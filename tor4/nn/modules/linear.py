import math

import tor4

from ...tensor import Tensor
from .. import functional as F
from .. import init
from ..parameter import Parameter
from .module import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.in_features = in_features
        super().__init__()

        self.out_features = out_features
        self.weight = Parameter(tor4.empty(out_features, in_features))
        if bias:
            self.bias = Parameter(tor4.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if hasattr(self, "bias"):
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={hasattr(self, "bias")}'

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias if hasattr(self, "bias") else None)
