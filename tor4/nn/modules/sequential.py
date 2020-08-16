from ...tensor import Tensor
from .module import Module


class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        super().__init__()

        self.n = len(args)
        for i, module in enumerate(args):
            setattr(self, str(i), module)

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.n):
            x = getattr(self, str(i))(x)

        return x
