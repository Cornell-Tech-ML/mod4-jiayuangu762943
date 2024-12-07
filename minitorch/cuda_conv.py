from numba import cuda
import numpy as np

@cuda.jit
def cuda_conv1d_kernel(
    out, input, weight, batch, out_channels, width, in_channels, kw, reverse
):
    b, oc, w = cuda.grid(3)
    if b < batch and oc < out_channels and w < width:
        tmp = 0.0
        for ic in range(in_channels):
            for k in range(kw):
                if reverse:
                    w_in = w - (kw - 1 - k)
                else:
                    w_in = w + k - kw // 2

                if 0 <= w_in < width:
                    tmp += input[b, ic, w_in] * weight[oc, ic, k]
        out[b, oc, w] = tmp

def cuda_conv1d(input, weight, reverse=False):
    batch, in_channels, width = input.shape
    out_channels, _, kw = weight.shape
    out = np.zeros((batch, out_channels, width), dtype=input.dtype)

    threadsperblock = (8, 8, 8)
    blockspergrid_x = (batch + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (out_channels + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_z = (width + threadsperblock[2] - 1) // threadsperblock[2]
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    cuda_conv1d_kernel[blockspergrid, threadsperblock](
        out, input, weight, batch, out_channels, width, in_channels, kw, reverse
    )
    return out

# Similar implementation for cuda_conv2d
