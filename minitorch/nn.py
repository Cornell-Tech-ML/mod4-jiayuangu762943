from typing import Tuple

from .tensor import Tensor
from .autodiff import Context
from .tensor_functions import Function

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw

    # Reshape and permute the input tensor
    input_reshaped = input.view(
        batch, channel, new_height, kh, new_width, kw
    )
    input_permuted = input_reshaped.permute(0, 1, 2, 4, 3, 5)
    input_tiled = input_permuted.contiguous().view(
        batch, channel, new_height, new_width, kh * kw
    )

    return input_tiled, new_height, new_width


# TODO: Implement for Task 4.3.
class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int) -> Tensor:
        ctx.save_for_backward(input, dim)
        return input.f.max_reduce(input, dim)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        input, dim = ctx.saved_values
        max_vals = input.f.max_reduce(input, dim)
        max_mask = input == max_vals
        grad_input = grad_output * max_mask
        return grad_input

def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, dim)

def softmax(input: Tensor, dim: int) -> Tensor:
    # Subtract max for numerical stability
    input_max = max(input, dim=dim)
    input_shifted = input - input_max
    exp_input = input_shifted.exp()
    sum_exp = exp_input.sum(dim=dim)
    softmax = exp_input / sum_exp
    return softmax

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    input_max = max(input, dim=dim)
    input_shifted = input - input_max
    log_sum_exp = input_shifted.exp().sum(dim=dim).log()
    return input_shifted - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    input_tiled, new_height, new_width = tile(input, kernel)
    # Apply max over the last dimension (kernel elements)
    pooled = max(input_tiled, dim=4)
    return pooled

def dropout(input: Tensor, rate: float, train: bool = True) -> Tensor:
    if not train:
        return input
    else:
        mask = rand(input.shape, backend=input.backend) > rate
        return input * mask
    



