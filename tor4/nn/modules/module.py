from typing import Iterable

from ...tensor import Tensor
from ..parameter import Parameter


def _addindent(s_: str, num_spaces: int) -> str:
    s = s_.split("\n")
    if len(s) == 1:
        return s_

    first = s.pop(0)
    s = [(num_spaces * " ") + line for line in s]
    ss = "\n".join(s)
    ss = first + "\n" + ss

    return ss


class Module:
    def __init__(self) -> None:
        self.training = True

    def extra_repr(self) -> str:
        return ""

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")

        child_lines = []
        for key, module in self.__dict__.items():
            if not isinstance(module, Module):
                continue

            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)

        lines = extra_lines + child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"

        return main_str

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def parameters(self) -> Iterable[Parameter]:
        for _, value in self.__dict__.items():
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            assert parameter.grad is not None
            parameter.grad.zero_()

    def train(self, mode: bool = True) -> None:
        self.training = mode
        for _, value in self.__dict__.items():
            if isinstance(value, Module):
                value.train(mode)

    def eval(self) -> None:
        self.train(False)
