from typing import Iterable


class Optimizer:
    def __init__(self, parameters: Iterable) -> None:
        param_group = list(parameters)
        if len(param_group) == 0:
            raise ValueError("Optimizer got an empty parameter list")

        self.parameters = param_group

    def zero_grad(self) -> None:
        for p in self.parameters:
            if p.grad is not None:
                p.grad.zero_()
