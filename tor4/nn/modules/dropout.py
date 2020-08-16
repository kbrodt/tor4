from ...tensor import Tensor
from .. import functional as F
from .module import Module


class _DropoutNd(Module):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()

        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}"


class Dropout(_DropoutNd):
    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, p=self.p, training=self.training, inplace=self.inplace)


class Dropout2d(_DropoutNd):
    def forward(self, x: Tensor) -> Tensor:
        return F.dropout2d(x, p=self.p, training=self.training, inplace=self.inplace)
