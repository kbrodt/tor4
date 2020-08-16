from typing import Optional, Type

import numpy as np

from .tensor import Tensor, float32


def empty(*shape: int, dtype: Type = float32, requires_grad: bool = False) -> Tensor:
    return Tensor(data=np.empty(shape, dtype=dtype), requires_grad=requires_grad)


def zeros(*shape: int, dtype: Type = float32, requires_grad: bool = False) -> Tensor:
    return Tensor(data=np.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def ones(*shape: int, dtype: Type = float32, requires_grad: bool = False) -> Tensor:
    return Tensor(data=np.ones(shape, dtype=dtype), requires_grad=requires_grad)


def arange(
    start: int,
    end: Optional[int] = None,
    step: Optional[int] = None,
    dtype: Type = None,
    requires_grad: bool = False,
) -> Tensor:
    return Tensor(
        data=np.arange(start=start, stop=end, step=step, dtype=dtype),
        requires_grad=requires_grad,
    )


def randn(*shape: int, dtype: Type = float32, requires_grad: bool = False) -> Tensor:
    return Tensor(
        data=np.random.randn(*shape), dtype=dtype, requires_grad=requires_grad
    )
