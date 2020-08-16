from typing import Iterable

from ..tensor import no_grad
from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters: Iterable, lr: float = 1e-3) -> None:
        super().__init__(parameters)

        self.lr = lr

    def step(self) -> None:
        with no_grad():
            for p in self.parameters:
                p -= self.lr * p.grad
