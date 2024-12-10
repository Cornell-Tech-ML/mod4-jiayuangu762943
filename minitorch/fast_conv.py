from typing import Tuple, TypeVar, Any

from numba import prange # type: ignore
from numba import njit as _njit # type: ignore

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:  # noqa: D103
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    # TODO: Implement for Task 4.1.

    batch_size, out_channels, out_width = out_shape  # out_width = T
    batch_size_in, in_channels, in_width = input_shape  # in_width = T
    out_channels_wt, in_channels_wt, kernel_width = weight_shape  # kernel_width = K

    assert (
        batch_size == batch_size_in
        and in_channels == in_channels_wt
        and out_channels == out_channels_wt
    )

    s0, s1, s2 = input_strides  # Input strides
    w0, w1, w2 = weight_strides  # Weight strides
    o0, o1, o2 = out_strides  # Output strides

    for b in prange(batch_size):
        for oc in range(out_channels):
            for w_out in range(out_width):
                tmp = 0.0
                for ic in range(in_channels):
                    for k in range(kernel_width):
                        if reverse:
                            # Backward convolution (reverse=True)
                            w_in = w_out - k
                            weight_k = k
                        else:
                            # Forward convolution (reverse=False)
                            w_in = w_out + k
                            weight_k = k
                        # Check if w_in is within bounds
                        if 0 <= w_in < in_width:
                            input_idx = b * s0 + ic * s1 + w_in * s2
                            weight_idx = oc * w0 + ic * w1 + weight_k * w2
                            tmp += input[input_idx] * weight[weight_idx]
                        else:
                            # Out-of-bounds input values are considered zero
                            pass
                out_idx = b * o0 + oc * o1 + w_out * o2
                out[out_idx] = tmp


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:  # noqa: D102
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    # TODO: Implement for Task 4.2.
    batch_size, out_channels, out_height, out_width = out_shape
    batch_size_in, in_channels, in_height, in_width = input_shape
    out_channels_wt, in_channels_wt, kernel_height, kernel_width = weight_shape

    assert (
        batch_size == batch_size_in
        and in_channels == in_channels_wt
        and out_channels == out_channels_wt
    )

    s0, s1, s2, s3 = input_strides  # Input strides
    w0, w1, w2, w3 = weight_strides  # Weight strides
    o0, o1, o2, o3 = out_strides  # Output strides

    for b in prange(batch_size):
        for oc in range(out_channels):
            for h_out in range(out_height):
                for w_out in range(out_width):
                    tmp = 0.0
                    for ic in range(in_channels):
                        for kh in range(kernel_height):
                            for kw in range(kernel_width):
                                if reverse:
                                    h_in = h_out - kh
                                    w_in = w_out - kw
                                    weight_kh = kh
                                    weight_kw = kw
                                else:
                                    h_in = h_out + kh
                                    w_in = w_out + kw
                                    weight_kh = kh
                                    weight_kw = kw

                                # Check if indices are within input bounds
                                if 0 <= h_in < in_height and 0 <= w_in < in_width:
                                    input_idx = b * s0 + ic * s1 + h_in * s2 + w_in * s3
                                    weight_idx = (
                                        oc * w0
                                        + ic * w1
                                        + weight_kh * w2
                                        + weight_kw * w3
                                    )
                                    tmp += input[input_idx] * weight[weight_idx]

                    out_idx = b * o0 + oc * o1 + h_out * o2 + w_out * o3
                    out[out_idx] = tmp


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradient of 2D Convolution

        Args:
        ----
            ctx : Context
            grad_output : Tensor

        Returns:
        -------
            (:class: Tuple[Tensor, Tensor])

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
