from typing import (
    Any,
    Callable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)

import numpy as np
from scipy import special

Array = Union[np.ndarray]
Numeric = Union[int, float]
# Lists = Union[Numeric, List['Lists']]
Tuplist = Union[Tuple[int, ...], List[int]]
Dim = Union[int, Tuplist]
Arrayable = Union[Numeric, Tuplist, Array]
Tensorable = Union[Arrayable, "Tensor"]

float32 = np.float32
float64 = np.float64
int64 = np.int64


def ensure_array(arr: Arrayable, dtype: Optional[Union[str, Type]] = None) -> Array:
    return np.array(arr, dtype=dtype, copy=False)


def ensure_tensor(arr: Tensorable) -> "Tensor":
    if not isinstance(arr, Tensor):
        return Tensor(data=arr)

    return arr


class Dependancy(NamedTuple):
    tensor: "Tensor"
    grad_fn: Callable[[Array], Array]


class Tensor:
    no_grad = False

    def __init__(
        self,
        data: Arrayable,
        dtype: Optional[Union[str, Type]] = None,
        depends_on: Optional[List[Dependancy]] = None,
        requires_grad: bool = False,
    ) -> None:
        self._data: Array = ensure_array(data, dtype=dtype)
        self._dtype: str = self._data.dtype.name
        if requires_grad and "float" not in self._dtype:
            raise RuntimeError("Only float tensors support requires_grad")

        self._depends_on: List[Dependancy] = depends_on or []
        self._is_leaf: bool = not self._depends_on
        self._requires_grad: bool = requires_grad
        self._grad: Optional[Tensor] = None

    def __neg__(self) -> "Tensor":
        return neg(self)

    def __add__(self, other: Tensorable) -> "Tensor":
        return add(self, ensure_tensor(other))

    def __radd__(self, other: Tensorable) -> "Tensor":
        return self.__add__(other)

    def __iadd__(self, other: Tensorable) -> "Tensor":
        self.data += ensure_tensor(other).data

        return self

    def __sub__(self, other: Tensorable) -> "Tensor":
        return sub(self, ensure_tensor(other))

    def __rsub__(self, other: Tensorable) -> "Tensor":
        return self.__sub__(other)

    def __isub__(self, other: Tensorable) -> "Tensor":
        self.data -= ensure_tensor(other).data

        return self

    def __mul__(self, other: Tensorable) -> "Tensor":
        return mul(self, ensure_tensor(other))

    def __rmul__(self, other: Tensorable) -> "Tensor":
        return self.__mul__(other)

    def __imul__(self, other: Tensorable) -> "Tensor":
        self.data *= ensure_tensor(other).data

        return self

    def __truediv__(self, other: Tensorable) -> "Tensor":
        return div(self, ensure_tensor(other))

    def __rtruediv__(self, other: Tensorable) -> "Tensor":
        return div(ensure_tensor(other), self)

    def __itruediv__(self, other: Tensorable) -> "Tensor":
        self.data /= ensure_tensor(other).data

        return self

    def __pow__(self, other: Numeric) -> "Tensor":
        return pow(self, other)

    def __matmul__(self, other: Tensorable) -> "Tensor":
        return matmul(self, ensure_tensor(other))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, indices: Union[None, int, slice, Tuple[Any, ...]]
    ) -> "Tensor":
        return tslice(self, indices)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"tensor({self._data}, requires_grad={self.requires_grad})"

    @property
    def data(self) -> Array:
        return self._data

    @data.setter
    def data(self, new_data: Array) -> None:
        if not self.no_grad and self.requires_grad:
            raise RuntimeError(
                "Variable that requires grad has been used an in-place operation."
            )

        self._data = new_data

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        if not self.is_leaf:
            raise RuntimeError(
                "you can only change requires_grad flags of leaf variables"
            )

        self._requires_grad = value

    @property
    def is_leaf(self) -> bool:
        return self._is_leaf

    @property
    def depends_on(self) -> List[Dependancy]:
        return self._depends_on

    @property
    def grad(self) -> Optional["Tensor"]:
        return self._grad

    @grad.setter
    def grad(self, other: Optional["Tensor"]) -> None:
        self._grad = other

    def numel(self) -> int:
        return numel(self)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape  # type: ignore

    @overload
    def size(self) -> Tuple[int, ...]:
        ...

    @overload
    def size(self, dim: int) -> int:
        ...

    def size(self, dim: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
        if dim is None:
            return self.shape

        return self.shape[dim]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def dim(self) -> int:
        return len(self.shape)

    def ndimension(self) -> int:
        return len(self.shape)

    def reshape(self, shape: Tuplist) -> "Tensor":
        return reshape(self, shape)

    def view(self, *shape: int) -> "Tensor":
        return self.reshape(shape)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        return transpose(self, dim0=dim0, dim1=dim1)

    def t(self) -> "Tensor":
        return self.transpose(dim0=0, dim1=1)

    def item(self) -> Numeric:
        return self.data.item()  # type: ignore

    def tolist(self) -> list:
        return self.data.tolist()  # type: ignore

    def numpy(self) -> Array:
        if self.requires_grad:
            raise RuntimeError(
                "Can't call numpy() on Variable that requires grad. Use .detach().numpy() istead"
            )

        return self.data

    def cpu(self) -> "Tensor":
        return self

    def cuda(self) -> "Tensor":
        return self

    def sum(self, dim: Optional[Dim] = None, keepdim: bool = False) -> "Tensor":
        return reduce_sum(self, dim=dim, keepdim=keepdim)

    def mean(self, dim: Optional[Dim] = None, keepdim: bool = False) -> "Tensor":
        return reduce_mean(self, dim=dim, keepdim=keepdim)

    @overload
    def max(self) -> "Tensor":
        ...

    @overload
    def max(self, dim: int) -> Tuple["Tensor", "Tensor"]:
        ...

    @overload
    def max(self, dim: int, keepdim: bool) -> Tuple["Tensor", "Tensor"]:
        ...

    def max(
        self, dim: Optional[int] = None, keepdim: bool = False
    ) -> Union["Tensor", Tuple["Tensor", "Tensor"]]:
        return reduce_max(self, dim=dim, keepdim=keepdim)

    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        return reduce_argmax(self, dim=dim, keepdim=keepdim)

    def pow(self, val: Numeric) -> "Tensor":
        return pow(self, val)

    def exp(self) -> "Tensor":
        return exp(self)

    def log(self) -> "Tensor":
        return log(self)

    def log1p(self) -> "Tensor":
        return log1p(self)

    def sigmoid(self) -> "Tensor":
        return sigmoid(self)

    def tanh(self) -> "Tensor":
        return tanh(self)

    def detach(self) -> "Tensor":
        return Tensor(data=self.data)

    def zero_(self) -> None:
        if not Tensor.no_grad and self.requires_grad:
            raise RuntimeError(
                "a leaf Variable that requires grad has been used in an in-place operation."
            )

        self._data.fill(0)

    def uniform_(self, _from: Numeric = 0, to: Numeric = 1) -> None:
        if not Tensor.no_grad and self.requires_grad:
            raise RuntimeError(
                "a leaf Variable that requires grad has been used in an in-place operation."
            )

        self._data[:] = np.random.uniform(low=_from, high=to, size=self._data.shape)

    def backward(
        self, grad: Optional["Tensor"] = None, retain_graph: bool = False
    ) -> None:
        self._backward(grad)

        if not retain_graph:
            self._free_buffers()

    def _backward(self, grad: Optional["Tensor"] = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("Variable has to be differentiable")

        if grad is None:
            if np.prod(self.shape) == 1:
                grad = Tensor(1, dtype=self.data.dtype)
            else:
                raise RuntimeError("Gradient shape is not the same as s one")

        if self.is_leaf:
            if self._grad is None:
                self._grad = Tensor(np.zeros_like(self.data))

            self._grad.data += grad.data

        for dependancy in self.depends_on:
            backward_grad = dependancy.grad_fn(grad.data)
            dependancy.tensor._backward(Tensor(backward_grad))

    def _free_buffers(self) -> None:
        for dependancy in self.depends_on:
            dependancy.tensor._free_buffers()

        self._depends_on = []


def tensor(
    data: Arrayable,
    dtype: Optional[Union[str, Type]] = None,
    requires_grad: bool = False,
) -> Tensor:
    return Tensor(data=data, dtype=dtype, requires_grad=requires_grad)


class no_grad:
    def __enter__(self) -> None:
        self.prev = Tensor.no_grad
        Tensor.no_grad = True

    def __exit__(self, *args: Any) -> None:
        Tensor.no_grad = self.prev


# Tensors


def numel(input: Tensor) -> int:
    return input.data.size  # type: ignore


# Reduction operations


def reduce_sum(
    input: Tensor, dim: Optional[Dim] = None, keepdim: bool = False
) -> Tensor:
    if dim is None:
        assert not keepdim

    data = input.data.sum(axis=dim, keepdims=keepdim)

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            nonlocal dim
            shape = [1] * input.data.ndim
            if dim is not None:
                if not keepdim:
                    grad = np.expand_dims(grad, dim)

                if isinstance(dim, int):
                    dim = [dim]

                for d in dim:
                    shape[d] = input.shape[d]

            adjoint = np.ones(shape=shape, dtype=input.dtype)

            return grad * adjoint

        depends_on.append(Dependancy(tensor=input, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def reduce_mean(
    input: Tensor, dim: Optional[Dim] = None, keepdim: bool = False
) -> Tensor:
    shape = np.array(input.shape)[dim]

    return reduce_sum(input / np.prod(shape), dim=dim, keepdim=keepdim)


def reduce_max(
    input: Tensor, dim: Optional[int] = None, keepdim: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    if dim is None:
        assert not keepdim

    argmax = input.data.argmax(axis=dim)

    if dim is not None:
        data = np.take_along_axis(
            input.data, np.expand_dims(argmax, axis=dim), axis=dim
        )
        if not keepdim:
            data = data.squeeze(axis=dim)
    else:
        argmax_unravel = np.unravel_index(argmax, input.data.shape)
        data = input.data[argmax_unravel]

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            adjoint = np.zeros_like(input.data)

            if dim is not None:
                np.put_along_axis(
                    adjoint, np.expand_dims(argmax, axis=dim), 1, axis=dim
                )
                if not keepdim:
                    grad = np.expand_dims(grad, axis=dim)
            else:
                adjoint[argmax_unravel] = 1

            return grad * adjoint

        depends_on.append(Dependancy(tensor=input, grad_fn=grad_fn))

    out = Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)

    if dim is None:
        return out

    if keepdim:
        indices = np.expand_dims(argmax, axis=dim)
    else:
        indices = argmax

    return out, Tensor(data=indices)


def reduce_argmax(
    input: Tensor, dim: Optional[int] = None, keepdim: bool = False
) -> Tensor:
    if dim is None:
        assert not keepdim

    data = input.data.argmax(axis=dim)

    if keepdim:
        data = np.expand_dims(data, axis=dim)

    return Tensor(data=data)


# Pointwise operations


def neg(input: Tensor) -> Tensor:
    data = -input.data

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            return -grad

        depends_on.append(Dependancy(tensor=input, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def handle_grad_broadcasting(grad: Array, shape: Tuplist) -> Array:
    # https://stackoverflow.com/questions/45428696/more-pythonic-way-to-compute-derivatives-of-broadcast-addition-in-numpy

    ndim = grad.ndim - len(shape)
    axis_first = tuple(range(ndim))
    axis = axis_first + tuple(i + ndim for i, dim in enumerate(shape) if dim == 1)
    grad = np.sum(grad, axis=axis, keepdims=True)
    grad = np.squeeze(grad, axis=axis_first)

    return grad


def add(left: Tensor, right: Tensor) -> Tensor:
    data = left.data + right.data

    depends_on = []
    if not Tensor.no_grad and left.requires_grad:

        def grad_fn_left(grad: Array) -> Array:
            adjoint = np.ones_like(left.data)
            grad = grad * adjoint
            grad = handle_grad_broadcasting(grad, left.shape)

            return grad

        depends_on.append(Dependancy(tensor=left, grad_fn=grad_fn_left))

    if not Tensor.no_grad and right.requires_grad:

        def grad_fn_right(grad: Array) -> Array:
            adjoint = np.ones_like(right.data)
            grad = grad * adjoint
            grad = handle_grad_broadcasting(grad, right.shape)

            return grad

        depends_on.append(Dependancy(tensor=right, grad_fn=grad_fn_right))

    requires_grad = not Tensor.no_grad and (left.requires_grad or right.requires_grad)

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def sub(left: Tensor, right: Tensor) -> Tensor:
    return add(left, neg(right))


def mul(left: Tensor, right: Tensor) -> Tensor:
    data = left.data * right.data

    depends_on = []
    if not Tensor.no_grad and left.requires_grad:

        def grad_fn_left(grad: Array) -> Array:
            adjoint = right.data
            grad = grad * adjoint
            grad = handle_grad_broadcasting(grad, left.shape)

            return grad

        depends_on.append(Dependancy(tensor=left, grad_fn=grad_fn_left))

    if not Tensor.no_grad and right.requires_grad:

        def grad_fn_right(grad: Array) -> Array:
            adjoint = left.data
            grad = grad * adjoint
            grad = handle_grad_broadcasting(grad, right.shape)

            return grad

        depends_on.append(Dependancy(tensor=right, grad_fn=grad_fn_right))

    requires_grad = not Tensor.no_grad and (left.requires_grad or right.requires_grad)

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def div(left: Tensor, right: Tensor) -> Tensor:
    data = left.data / right.data

    depends_on = []
    if not Tensor.no_grad and left.requires_grad:

        def grad_fn_left(grad: Array) -> Array:
            adjoint = 1 / right.data
            grad = grad * adjoint
            grad = handle_grad_broadcasting(grad, left.shape)

            return grad

        depends_on.append(Dependancy(tensor=left, grad_fn=grad_fn_left))

    if not Tensor.no_grad and right.requires_grad:

        def grad_fn_right(grad: Array) -> Array:
            adjoint = -left.data / right.data ** 2
            grad = grad * adjoint
            grad = handle_grad_broadcasting(grad, right.shape)

            return grad

        depends_on.append(Dependancy(tensor=right, grad_fn=grad_fn_right))

    requires_grad = not Tensor.no_grad and (left.requires_grad or right.requires_grad)

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def pow(input: Tensor, exponent: Numeric) -> Tensor:
    data = input.data ** exponent

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            adjoint = exponent * input.data ** (exponent - 1)

            return grad * adjoint

        depends_on.append(Dependancy(tensor=input, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def exp(input: Tensor) -> Tensor:
    data = np.exp(input.data)

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            adjoint = data

            return grad * adjoint

        depends_on.append(Dependancy(tensor=input, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def log(input: Tensor) -> Tensor:
    data = np.log(input.data)

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            adjoint = 1 / input.data

            return grad * adjoint

        depends_on.append(Dependancy(tensor=input, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def log1p(input: Tensor) -> Tensor:
    data = np.log1p(input.data)

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            adjoint = 1 / (1 + input.data)

            return grad * adjoint

        depends_on.append(Dependancy(tensor=input, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def sigmoid(input: Tensor) -> Tensor:
    data = special.expit(input.data)

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            adjoint = data * (1 - data)

            return grad * adjoint

        depends_on.append(Dependancy(tensor=input, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def tanh(input: Tensor) -> Tensor:
    data = np.tanh(input.data)

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            adjoint = 1 - data ** 2

            return grad * adjoint

        depends_on.append(Dependancy(tensor=input, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


# Indexing, slicing operations


def tslice(input: Tensor, indices: Union[None, int, slice, Tuple[Any, ...]]) -> Tensor:
    data = input.data[indices]

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            adjoint = np.zeros_like(input.data)
            adjoint[indices] = grad

            return adjoint

        depends_on.append(Dependancy(tensor=input, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def reshape(input: Tensor, shape: Tuplist) -> Tensor:
    old_shape = input.shape
    data = input.data.reshape(shape)

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            return grad.reshape(old_shape)

        depends_on.append(Dependancy(tensor=input, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    data = input.data.swapaxes(dim0, dim1)

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            return grad.swapaxes(dim0, dim1)

        depends_on.append(Dependancy(tensor=input, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


# BLAS and LAPACK operations


def matmul(left: Tensor, right: Tensor) -> Tensor:
    data = left.data @ right.data

    depends_on = []
    if not Tensor.no_grad and left.requires_grad:

        def grad_fn_left(grad: Array) -> Array:
            adjoint = right.data.T

            N = (grad.ndim + adjoint.ndim) - left.ndim
            assert N % 2 == 0

            N //= 2
            axes = (tuple(range(-1, -N - 1, -1)), tuple(range(N)))

            return np.tensordot(grad, adjoint, axes=axes)

        depends_on.append(Dependancy(tensor=left, grad_fn=grad_fn_left))

    if not Tensor.no_grad and right.requires_grad:

        def grad_fn_right(grad: Array) -> Array:
            adjoint = left.data.T

            N = (grad.ndim + adjoint.ndim) - right.ndim
            assert N % 2 == 0

            N //= 2
            axes = (tuple(range(-1, -N - 1, -1)), tuple(range(N)))

            return np.tensordot(adjoint, grad, axes=axes)

        depends_on.append(Dependancy(tensor=right, grad_fn=grad_fn_right))

    requires_grad = not Tensor.no_grad and (left.requires_grad or right.requires_grad)

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)
