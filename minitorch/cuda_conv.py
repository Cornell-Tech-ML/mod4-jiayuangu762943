# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_ops import TensorOps

FakeCUDAKernel = Any

Fn = TypeVar("Fn")

def device_jit(fn: Fn, **kwargs) -> Fn:  # noqa: ANN003
    return _jit(device=True, **kwargs)(fn)  # type: ignore

def jit(fn, **kwargs) -> FakeCUDAKernel:  # noqa: ANN001, ANN003
    return _jit(**kwargs)(fn)  # type: ignore

to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32

class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def conv1d(input: Tensor, weight: Tensor, reverse: bool = False) -> Tensor:
        # input: (batch, in_channels, width)
        # weight: (out_channels, in_channels, kernel_width)
        batch, in_channels, in_width = input.shape
        out_channels, in_channels_wt, kernel_width = weight.shape
        assert in_channels == in_channels_wt
        out_width = in_width

        out = input.zeros((batch, out_channels, out_width))
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
        tensor_conv1d_kernel[blockspergrid, threadsperblock](
            *out.tuple(), out.size,
            *input.tuple(),
            *weight.tuple(),
            reverse
        )
        return out

    @staticmethod
    def conv2d(input: Tensor, weight: Tensor, reverse: bool = False) -> Tensor:
        # input: (batch, in_channels, height, width)
        # weight: (out_channels, in_channels, k_height, k_width)
        batch, in_channels, in_height, in_width = input.shape
        out_channels, in_channels_wt, k_height, k_width = weight.shape
        assert in_channels == in_channels_wt

        out = input.zeros((batch, out_channels, in_height, in_width))
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
        tensor_conv2d_kernel[blockspergrid, threadsperblock](
            *out.tuple(), out.size,
            *input.tuple(),
            *weight.tuple(),
            reverse
        )
        return out


# Below are the conv kernels

def tensor_conv1d_kernel(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    inp: Storage,
    inp_shape: Shape,
    inp_strides: Strides,
    wgt: Storage,
    wgt_shape: Shape,
    wgt_strides: Strides,
    reverse: bool,
):
    # out: (batch, out_channels, width)
    # inp: (batch, in_channels, width)
    # wgt: (out_channels, in_channels, kernel_width)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= out_size:
        return

    batch = out_shape[0]
    out_channels = out_shape[1]
    width_out = out_shape[2]
    in_channels = inp_shape[1]
    in_width = inp_shape[2]
    kernel_width = wgt_shape[2]

    # Convert linear index to (b, oc, w_out)
    out_index = cuda.local.array(MAX_DIMS, numba.int32)
    to_index(i, out_shape, out_index)
    b, oc, w_out = out_index[0], out_index[1], out_index[2]

    val = 0.0
    for ic in range(in_channels):
        for k in range(kernel_width):
            if reverse:
                w_in = w_out - k
            else:
                w_in = w_out + k

            if 0 <= w_in < in_width:
                inp_pos = (b * inp_strides[0] + ic * inp_strides[1] + w_in * inp_strides[2])
                w_pos = (oc * wgt_strides[0] + ic * wgt_strides[1] + k * wgt_strides[2])
                val += inp[inp_pos] * wgt[w_pos]

    out_pos = (b * out_strides[0] + oc * out_strides[1] + w_out * out_strides[2])
    out[out_pos] = val

tensor_conv1d_kernel = cuda.jit()(tensor_conv1d_kernel)


def tensor_conv2d_kernel(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    inp: Storage,
    inp_shape: Shape,
    inp_strides: Strides,
    wgt: Storage,
    wgt_shape: Shape,
    wgt_strides: Strides,
    reverse: bool,
):
    # out: (batch, out_channels, height, width)
    # inp: (batch, in_channels, height, width)
    # wgt: (out_channels, in_channels, k_height, k_width)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= out_size:
        return

    batch = out_shape[0]
    out_channels = out_shape[1]
    out_height = out_shape[2]
    out_width = out_shape[3]
    in_channels = inp_shape[1]
    in_height = inp_shape[2]
    in_width = inp_shape[3]
    k_height = wgt_shape[2]
    k_width = wgt_shape[3]

    out_index = cuda.local.array(MAX_DIMS, numba.int32)
    to_index(i, out_shape, out_index)
    b, oc, h_out, w_out = out_index[0], out_index[1], out_index[2], out_index[3]

    val = 0.0
    for ic in range(in_channels):
        for kh in range(k_height):
            for kw in range(k_width):
                if reverse:
                    h_in = h_out - kh
                    w_in = w_out - kw
                else:
                    h_in = h_out + kh
                    w_in = w_out + kw

                if 0 <= h_in < in_height and 0 <= w_in < in_width:
                    inp_pos = (b * inp_strides[0] + ic * inp_strides[1] + h_in * inp_strides[2] + w_in * inp_strides[3])
                    w_pos = (oc * wgt_strides[0] + ic * wgt_strides[1] + kh * wgt_strides[2] + kw * wgt_strides[3])
                    val += inp[inp_pos] * wgt[w_pos]

    out_pos = (b * out_strides[0] + oc * out_strides[1] + h_out * out_strides[2] + w_out * out_strides[3])
    out[out_pos] = val

tensor_conv2d_kernel = cuda.jit()(tensor_conv2d_kernel)
