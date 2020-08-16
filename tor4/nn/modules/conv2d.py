import math
import typing as t

import tor4

from ...tensor import Tensor
from .. import functional as F
from .. import init
from ..parameter import Parameter
from .module import Module
from .utils import _pair


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: t.Union[int, t.Tuple[int, int]] = 1,
        padding: t.Union[int, t.Tuple[int, int]] = 0,
        dilation: t.Union[int, t.Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.weight = Parameter(
            tor4.empty(self.out_channels, self.in_channels, *self.kernel_size)
        )
        if bias:
            self.bias = Parameter(tor4.zeros(self.out_channels))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if hasattr(self, "bias"):
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            + f"kernel_size={self.kernel_size}, "
            + f"stride={self.stride}, "
            + (self.padding != (0, 0)) * f"padding={self.padding}, "
            + (self.dilation != (1, 1)) * f"dilation={self.dilation}, "
            + (self.groups != 1) * f"groups={self.groups}, "
            + (not hasattr(self, "bias")) * "bias=False"
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(
            x,
            self.weight,
            self.bias if hasattr(self, "bias") else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
