from ...tensor import Tensor
from .. import functional as F
from .module import Module


class ReLU(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()

        self.inplace = inplace

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}" if self.inplace else ""

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x, inplace=self.inplace)
