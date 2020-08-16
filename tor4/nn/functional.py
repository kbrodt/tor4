import typing as t

import numpy as np
from scipy import special

import tor4

from ..tensor import Array, Tensor
from .modules.utils import _pair

# Activations


def relu(input: Tensor, inplace: bool = False) -> Tensor:
    mask = input.data > 0
    data = mask * input.data

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            adjoint = mask

            return grad * adjoint

        depends_on.append(tor4.Dependancy(tensor=input, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def softmax(input: Tensor, dim: int) -> Tensor:
    data = special.softmax(input.data, axis=dim)

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            data_t = np.swapaxes(data, dim, -1)
            adjoint = -np.expand_dims(data_t, -1) @ np.expand_dims(data_t, -2)
            bs = adjoint.shape[:-2] if data.ndim > 1 else [1]
            step = adjoint.shape[-1] + 1
            adjoint.reshape(*bs, -1)[..., ::step] += data_t.reshape(*bs, -1)

            grad_t = np.swapaxes(grad, dim, -1)
            if data.ndim > 1:
                grad_t = grad_t[..., None]
            out = np.swapaxes(adjoint, -1, -2) @ grad_t
            if data.ndim > 1:
                out = out.squeeze(-1)

            return np.swapaxes(out, dim, -1)

        depends_on.append(tor4.Dependancy(tensor=input, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def log_softmax(input: Tensor, dim: int) -> Tensor:
    data = input.data - special.logsumexp(input.data, axis=dim, keepdims=True)

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            data_softmax = special.softmax(input.data, axis=dim)
            data_softmax_t = np.swapaxes(data_softmax, dim, -1)
            adjoint = -np.repeat(
                np.expand_dims(data_softmax_t, -2), data_softmax_t.shape[-1], axis=-2
            )
            bs = adjoint.shape[:-2] if data.ndim > 1 else [1]
            step = adjoint.shape[-1] + 1
            adjoint.reshape(*bs, -1)[..., ::step] += 1

            grad_t = np.swapaxes(grad, dim, -1)
            if data.ndim > 1:
                grad_t = grad_t[..., None]
            out = np.swapaxes(adjoint, -1, -2) @ grad_t
            if data.ndim > 1:
                out = out.squeeze(-1)

            return np.swapaxes(out, dim, -1)

        depends_on.append(tor4.Dependancy(tensor=input, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


# Linear functions


def linear(input: Tensor, weight: Tensor, bias: t.Optional[Tensor] = None) -> Tensor:
    input = input @ weight.t()
    if bias is not None:
        input = input + bias

    return input


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: t.Optional[Tensor] = None,
    stride: t.Union[int, t.Tuple[int, int]] = 1,
    padding: t.Union[int, t.Tuple[int, int]] = 0,
    dilation: t.Union[int, t.Tuple[int, int]] = 1,
    groups: int = 1,
) -> Tensor:
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    assert padding == 0 or padding == (0, 0)

    b, c_in, h_in, w_in = input.shape
    c_out, c_in_o_groups, *kernel_size = weight.shape

    assert c_in_o_groups * groups == c_in

    h_out = int(
        (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    w_out = int(
        (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )

    if not (h_out > 0 and w_out > 0):
        raise RuntimeError(f"Output dimension is {h_out}x{w_out}")

    # im2col operation
    i = dilation[0] * np.arange(kernel_size[0])
    i = np.repeat(i, kernel_size[1])
    i = stride[0] * np.arange(h_out)[:, None] + i
    i = np.repeat(i, w_out, 0)
    i = np.tile(i, (1, c_in_o_groups))

    j = dilation[1] * np.arange(kernel_size[1])
    j = np.tile(j, kernel_size[0])
    j = stride[1] * np.arange(w_out)[:, None] + j
    j = np.tile(j, (h_out, 1))
    j = np.tile(j, (1, c_in_o_groups))

    k = np.repeat(np.arange(c_in_o_groups), np.prod(kernel_size)).reshape(1, -1)

    n_patches = h_out * w_out
    input_matrix = input.data[:, k, i, j].reshape(
        b, n_patches, c_in_o_groups * np.prod(kernel_size)
    )
    weight_matrix = weight.data.reshape(c_out, -1).transpose(1, 0)

    # https://github.com/numpy/numpy/issues/8957
    # data = (input_matrix @ weight_matrix).transpose(0, 2, 1).reshape(b, c_out, h_out, w_out)
    data = (
        (input_matrix.reshape(b * n_patches, -1) @ weight_matrix)
        .reshape(b, n_patches, -1)
        .transpose(0, 2, 1)
        .reshape(b, c_out, h_out, w_out)
    )

    depends_on = []
    if not Tensor.no_grad and input.requires_grad:

        def grad_fn_left(grad: Array) -> Array:
            """
                grad: [b, c_out, h_out, w_out]
            """

            adjoint = weight_matrix.T
            grad = grad.reshape(b, c_out, n_patches).swapaxes(-1, -2)

            N = (grad.ndim + adjoint.ndim) - input_matrix.ndim
            assert N % 2 == 0

            N //= 2
            axes = (tuple(range(-1, -N - 1, -1)), tuple(range(N)))

            out = np.tensordot(grad, adjoint, axes=axes)
            assert out.shape == (b, n_patches, c_in_o_groups * np.prod(kernel_size))

            # col2im operation
            # https://github.com/numpy/numpy/issues/5922
            # input_data = np.zeros_like(input.data)
            # np.add.at(input_data, (slice(None), k, i, j), out)

            inds = np.ravel_multi_index((k, i, j), input.shape[1:]).ravel()
            inds = np.tile(inds, (b, 1))
            inds = c_in * h_in * w_in * np.arange(b)[:, None] + inds
            input_data = np.bincount(
                inds.ravel(), weights=out.ravel(), minlength=np.prod(input.shape)
            )
            input_data = input_data.reshape(b, c_in, h_in, w_in)

            return input_data

        depends_on.append(tor4.Dependancy(tensor=input, grad_fn=grad_fn_left))

    if not Tensor.no_grad and weight.requires_grad:

        def grad_fn_right(grad: Array) -> Array:
            adjoint = input_matrix.T
            grad = grad.reshape(b, c_out, n_patches).swapaxes(-1, -2)

            N = (grad.ndim + adjoint.ndim) - weight_matrix.ndim
            assert N % 2 == 0

            N //= 2
            axes = (tuple(range(-1, -N - 1, -1)), tuple(range(N)))

            out = np.tensordot(adjoint, grad, axes=axes)
            out = out.transpose(1, 0).reshape(c_out, c_in_o_groups, *kernel_size)

            return out

        depends_on.append(tor4.Dependancy(tensor=weight, grad_fn=grad_fn_right))

    requires_grad = not Tensor.no_grad and (input.requires_grad or weight.requires_grad)

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


# Dropout functions


def dropout(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    if training:
        mask = np.random.binomial(1, 1 - p, size=input.shape) / (1 - p)
        data = mask * input.data
    else:
        data = input.data

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            if training:
                adjoint = mask

                return grad * adjoint

            return grad

        depends_on.append(tor4.Dependancy(tensor=input, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def dropout2d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    if training:
        size = input.shape[:2] + (1,) * len(input.shape[2:])
        mask = np.random.binomial(1, 1 - p, size=size) / (1 - p)
        data = mask * input.data
    else:
        data = input.data

    requires_grad = not Tensor.no_grad and input.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            if training:
                adjoint = mask

                return grad * adjoint

            return grad

        depends_on.append(tor4.Dependancy(tensor=input, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


# Loss functions


def mse_loss_slow(input: Tensor, target: Tensor) -> Tensor:
    return ((input - target) ** 2).mean()


def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    diff = input.data - target.data
    data = (diff ** 2).mean()

    depends_on = []
    if not Tensor.no_grad and input.requires_grad:

        def grad_fn_left(grad: Array) -> Array:
            adjoint = 2 * diff / np.prod(diff.shape)

            return grad * adjoint

        depends_on.append(tor4.Dependancy(tensor=input, grad_fn=grad_fn_left))

    if not Tensor.no_grad and target.requires_grad:

        def grad_fn_right(grad: Array) -> Array:
            adjoint = -2 * diff / np.prod(diff.shape)

            return grad * adjoint

        depends_on.append(tor4.Dependancy(tensor=target, grad_fn=grad_fn_right))

    requires_grad = not Tensor.no_grad and (input.requires_grad or target.requires_grad)

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def binary_cross_entropy_with_logits(input: Tensor, target: Tensor) -> Tensor:
    log1pexp = np.logaddexp(np.zeros_like(input.data), -input.data)
    data = (input.data - input.data * target.data + log1pexp).mean()

    depends_on = []
    if not Tensor.no_grad and input.requires_grad:

        def grad_fn_left(grad: Array) -> Array:
            adjoint = (special.expit(input.data) - target.data) / np.prod(input.shape)

            return grad * adjoint

        depends_on.append(tor4.Dependancy(tensor=input, grad_fn=grad_fn_left))

    if not Tensor.no_grad and target.requires_grad:

        def grad_fn_right(grad: Array) -> Array:
            adjoint = -input.data / np.prod(input.shape)

            return grad * adjoint

        depends_on.append(tor4.Dependancy(tensor=target, grad_fn=grad_fn_right))

    requires_grad = not Tensor.no_grad and (input.requires_grad or target.requires_grad)

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def cross_entropy_slow(input: Tensor, target: Tensor) -> Tensor:
    max_i = input.max(dim=-1, keepdim=True)[0]
    exp_i = (input - max_i).exp()
    soft_max = exp_i / exp_i.sum(dim=-1, keepdim=True)
    xent = -target * soft_max.log()
    xent = xent.sum(-1)
    xent = xent.mean(0)

    return xent


def cross_entropy_slow2(input: Tensor, target: Tensor) -> Tensor:
    sm = softmax(input, dim=-1)
    xent = -target * sm.log()
    xent = xent.sum(-1)
    xent = xent.mean(0)

    return xent


def cross_entropy_slow3(input: Tensor, target: Tensor) -> Tensor:
    log_sm = log_softmax(input, dim=-1)
    xent = -target * log_sm
    xent = xent.sum(-1)
    xent = xent.mean(0)

    return xent


def cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    target_exp = np.expand_dims(target.data, 1)
    data = np.take_along_axis(input.data, target_exp, axis=1) - special.logsumexp(
        input.data, axis=1, keepdims=True
    )
    data = -data.mean()

    depends_on = []
    if not Tensor.no_grad and input.requires_grad:

        def grad_fn_left(grad: Array) -> Array:
            adjoint = np.zeros_like(input.data)
            np.put_along_axis(adjoint, target_exp, -1, axis=1)
            adjoint += special.softmax(input.data, axis=1)
            adjoint /= np.prod(input.shape) / input.shape[1]

            return grad * adjoint

        depends_on.append(tor4.Dependancy(tensor=input, grad_fn=grad_fn_left))

    requires_grad = not Tensor.no_grad and (input.requires_grad or target.requires_grad)

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)
