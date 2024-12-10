from typing import Optional, Tuple
from .tensor import Tensor
from .tensor_functions import Function
from .autodiff import Context
import numpy as np
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
    
    # Calculate new dimensions after tiling
    new_height = height // kh
    new_width = width // kw

    tiled = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    tiled = tiled.permute(0, 1, 2, 4, 3, 5)
    tiled = tiled.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


# TODO: Implement for Task 4.3.
class AvgPool2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, kernel: Tensor) -> Tensor:
        """Compute 2D average pooling on a tensor.

        Args:
            ctx (Context): Context for storing saved tensors.
            input (Tensor): Input tensor of shape (batch, channel, height, width).
            kernel (Tensor): Tensor containing (kernel_height, kernel_width).

        Returns:
            Tensor: Pooled tensor of shape (batch, channel, new_height, new_width).

        """
        kh, kw = int(kernel._tensor._storage[0]), int(kernel._tensor._storage[1])
        tiled, new_height, new_width = tile(input, (kh, kw))
        # Compute mean along the last dimension
        pooled = tiled.mean(dim=len(tiled.shape)-1)
        pooled = pooled.view(pooled.shape[0],pooled.shape[1],pooled.shape[2],pooled.shape[3])
        # Save for backward
        ctx.save_for_backward(tiled, kernel)
        return pooled

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, int]:
        """Backward pass for 2D average pooling.

        Args:
            ctx (Context): Context containing saved tensors.
            grad_output (Tensor): Gradient of the loss with respect to the output tensor.

        Returns:
            Tuple[Tensor, None]: Gradient with respect to the input tensor and None for kernel (no gradient needed).

        """
        tiled, kernel = ctx.saved_tensors
        kh, kw = int(kernel._tensor._storage[0]), int(kernel._tensor._storage[1])
        batch, channel, new_height, new_width = grad_output.shape

        # Expand grad_output to match the tiled shape by adding two singleton dimensions
        grad_output_expanded = grad_output.view(batch, channel, new_height, new_width, 1, 1)

        # Distribute the gradient equally to each element in the pooling window
        # Shape after repeat: (batch, channel, new_height, new_width, kh, kw)
        grad_input_tiled = (grad_output_expanded / (kh * kw)).repeat((1, 1, 1, 1, kh, kw))
        
        # Permute and reshape to match the input tensor's shape (batch, channel, height, width)
        grad_input = grad_input_tiled.permute(0, 1, 2, 4, 3, 5).contiguous().view(batch, channel, new_height * kh, new_width * kw)
        
        return grad_input, -1

def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Compute 2D average pooling on a tensor.

    Args:
        input (Tensor): Input tensor of shape (batch, channel, height, width).
        kernel (Tuple[int, int]): Tuple specifying the (kernel_height, kernel_width).

    Returns:
        Tensor: Pooled tensor of shape (batch, channel, new_height, new_width).

    """
    assert len(input.shape) == 4, f"Input tensor must be 4D, but got shape {input.shape}"
    kernel_tensor = Tensor.make([kernel[0], kernel[1]], shape=(2,), backend=input.backend)
    return AvgPool2dFun.apply(input, kernel_tensor)


class Max(Function):
    """Max function over tensor elements."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max.

        Args:
            ctx (Context): The context to save information for backward computation.
            t1 (Tensor): The input tensor.
            dim (Tensor): The dimension to reduce over.

        Returns:
            Tensor: The max values along the specified dimension.

        """
        dim_int = int(dim.item())
        input_array = t1.to_numpy()  # Correct accessor

        # Compute max values and indices
        max_vals = np.max(input_array, axis=dim_int, keepdims=True)
        mask = (input_array == max_vals).astype(float)

        # Count number of maxima per slice
        num_max = np.sum(mask, axis=dim_int, keepdims=True)

        # Save mask and num_max for backward
        mask_tensor = Tensor.make(mask.flatten(), shape=t1.shape, backend=t1.backend)
        num_max_tensor = Tensor.make(num_max.flatten(), shape=max_vals.shape, backend=t1.backend)
        ctx.save_for_backward(mask_tensor, num_max_tensor, dim_int)

        # Compute output max values
        max_vals = np.squeeze(max_vals, axis = dim_int)
        max_vals_tensor = Tensor.make(max_vals.flatten(), shape=tuple(max_vals.shape), backend=t1.backend)
        return max_vals_tensor

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, int]:
        """Backward pass for max.

        Args:
            ctx (Context): The context with saved tensors.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tuple[Tensor, None]: The gradient with respect to the input tensor and None for dim.

        """
        mask_tensor, num_max_tensor, dim_int = ctx.saved_tensors
        mask = mask_tensor.to_numpy()
        num_max = num_max_tensor.to_numpy()
        grad_output_array = grad_output.to_numpy()

        # Broadcast grad_output to match mask shape
        grad_output_broadcast = np.expand_dims(grad_output_array, axis=dim_int)
        # Assign gradient equally to all maxima
        grad_input_array = mask * (grad_output_broadcast / num_max)

        # Flatten grad_input_array for Tensor.make
        grad_input_flat = grad_input_array.flatten()

        # Create a Tensor for grad_input
        grad_input = Tensor.make(
            grad_input_flat,
            shape=tuple(grad_input_array.shape),
            backend=grad_output.backend
        )
        return grad_input, -1
    
def max(input: Tensor, dim: int) -> Tensor:
    """Compute the max of input along the specified dimension.

    Args:
        input (Tensor): Input tensor.
        dim (int): Dimension along which to compute the max.

    Returns:
        Tensor: Tensor containing the maximum values along the specified dimension.

    """
    dim_tensor = Tensor.make([dim], shape=(1,), backend=input.backend)
    return Max.apply(input, dim_tensor)

class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim_tensor: Tensor) -> Tensor:
        """Forward pass for the Softmax function.

        Args:
            ctx (Context): Context to save intermediate values for backward computation.
            input (Tensor): Input tensor.
            dim_tensor (Tensor): Tensor containing the dimension along which to apply Softmax.

        Returns:
            Tensor: Tensor containing the Softmax results.

        """
        dim = int(dim_tensor.item())
        input_array = input.to_numpy()

        # Subtract max for numerical stability
        max_vals_array = np.max(input_array, axis=dim, keepdims=True)
        x_shifted = input_array - max_vals_array

        exp_x_shifted = np.exp(x_shifted)

        # Sum exp_x_shifted along the specified dimension
        sum_exp_x = np.sum(exp_x_shifted, axis=dim, keepdims=True)
        softmax_array = exp_x_shifted / sum_exp_x

        # Save softmax_array and dimension for backward
        ctx.save_for_backward(softmax_array, dim)

        return Tensor.make(softmax_array.flatten(), shape=input.shape, backend=input.backend)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, int]:
        """Backward pass for the Softmax function.

        Args:
            ctx (Context): Context containing saved tensors from the forward pass.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
            Tuple[Tensor, int]: Gradient of the loss with respect to the input and a dummy integer.

        """
        softmax_array, dim = ctx.saved_tensors
        grad_output_array = grad_output.to_numpy()

        # Compute grad_input
        grad_input_array = softmax_array * (grad_output_array - np.sum(grad_output_array * softmax_array, axis=dim, keepdims=True))

        grad_input = Tensor.make(grad_input_array.flatten(), shape=grad_output.shape, backend=grad_output.backend)

        return grad_input, -1
    
def softmax(input: Tensor, dim: int) -> Tensor:
    """Functional API for the Softmax function.

    Args:
        input (Tensor): Input tensor.
        dim (int): Dimension along which to apply Softmax.

    Returns:
        Tensor: Tensor containing the Softmax results.

    """
    dim_tensor = Tensor.make([dim], shape=(1,), backend=input.backend)
    return Softmax.apply(input, dim_tensor)


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim_tensor: Tensor) -> Tensor:
        """Forward pass for the LogSoftmax function.

        Args:
            ctx (Context): Context to save intermediate values for backward computation.
            input (Tensor): Input tensor.
            dim_tensor (Tensor): Tensor containing the dimension along which to apply LogSoftmax.

        Returns:
            Tensor: Tensor containing the LogSoftmax results.

        """
        dim = int(dim_tensor.item())

        # Step 1: Compute max along dim for numerical stability
        max_tensor = input.f.max_reduce(input, dim)  # Shape: same as input with dim=1

        # Step 2: Subtract max_tensor from input (broadcasted)
        shifted = input - max_tensor  # Element-wise subtraction

        # Step 3: Exponentiate the shifted tensor
        exp_shifted = shifted.f.exp_map(shifted)  # Element-wise exponentiation

        # Step 4: Sum of exponentials along dim
        sum_exp = exp_shifted.sum(dim)  # Sum along dim

        # Step 5: Log of sum_exp
        log_sum_exp = sum_exp.f.log_map(sum_exp)  # Element-wise logarithm

        # Step 6: Compute LogSoftmax by subtracting log_sum_exp from shifted
        log_softmax = shifted - log_sum_exp  # Element-wise subtraction

        # Step 7: Compute softmax = exp(log_softmax)
        softmax = log_softmax.f.exp_map(log_softmax)  # Element-wise exponentiation

        # Save softmax and dim for backward pass
        ctx.save_for_backward(softmax, dim)

        return log_softmax
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, int]:
        """Backward pass for the LogSoftmax function.

        Args:
            ctx (Context): Context containing saved tensors from the forward pass.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
            Tuple[Tensor, int]: Gradient of the loss with respect to the input and a dummy integer.

        """
        softmax, dim_int = ctx.saved_tensors

        # Step 1: Compute the sum of grad_output along dim
        sum_grad = grad_output.sum(dim_int)  # Shape: same as sum_exp

        # Step 2: Compute softmax_sum_grad = softmax * sum_grad
        softmax_sum_grad = softmax.f.mul_zip(softmax, sum_grad)  # Element-wise multiplication

        # Step 3: Compute grad_input = grad_output - softmax_sum_grad
        grad_input = grad_output - softmax_sum_grad  # Element-wise subtraction

        return grad_input, -1

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Functional API for the LogSoftmax function.

    Args:
        input (Tensor): Input tensor.
        dim (int): Dimension along which to apply LogSoftmax.

    Returns:
        Tensor: Tensor containing the LogSoftmax results.

    """
    dim_tensor = Tensor.make([dim], shape=(1,), backend=input.backend)
    return LogSoftmax.apply(input, dim_tensor)



class MaxPool2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, kernel: Tensor) -> Tensor:
        """Forward pass for 2D Max Pooling.

        Args:
            ctx (Context): Context to save intermediate values for backward computation.
            input (Tensor): Input tensor with shape (batch, channels, height, width).
            kernel (Tensor): Tensor specifying the kernel size (kh, kw).

        Returns:
            Tensor: Tensor containing the results of max pooling.

        """
        kh, kw = int(kernel[0]), int(kernel[1])

        batch, channels, height, width = input.shape

        # Compute output dimensions
        new_height = height // kh
        new_width = width // kw

        # Step 1: Reshape input to (batch, channels, new_height, kh, new_width, kw)
        reshaped = input.contiguous().view(batch, channels, new_height, kh, new_width, kw)

        # Step 2: Permute to (batch, channels, new_height, new_width, kh, kw)
        permuted = reshaped.permute(0, 1, 2, 4, 3, 5)

        # Step 3: Reshape to (batch, channels, new_height, new_width, kh * kw)
        tiled = permuted.contiguous().view(batch, channels, new_height, new_width, kh * kw)

        # Step 4: Compute max along the last dimension (kh * kw)
        max_tensor = tiled.f.max_reduce(tiled, len(tiled.shape) - 1)  # Shape: (batch, channels, new_height, new_width)
        
        # Step 5: Create mask where tiled == max_tensor (broadcasted)
        mask = tiled.f.eq_zip(tiled, max_tensor)     # Shape: (batch, channels, new_height, new_width, kh * kw)
        

        # Step 6: Count number of maxima per pooling window
        num_max = mask.sum(len(mask.shape)-1)       # Shape: (batch, channels, new_height, new_width)

        # Step 7: Save mask and num_max for backward pass
        ctx.save_for_backward(mask, num_max, kh, kw)
        
        max_tensor = max_tensor.contiguous().view(batch, channels, new_height, new_width)
        return max_tensor
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, int]:
        """Backward pass for 2D Max Pooling.

        Args:
            ctx (Context): Context containing saved tensors from the forward pass.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
            Tuple[Tensor, int]: Gradient of the loss with respect to the input and a dummy integer.

        """
        mask, num_max, kh, kw = ctx.saved_tensors
        # mask shape: (batch, channels, new_height, new_width, kh * kw)
        # num_max shape: (batch, channels, new_height, new_width)

        # Step 1: Compute 1 / num_max
        inv_num_max = num_max.f.inv_map(num_max)  # Shape: (batch, channels, new_height, new_width)
        
        inv_num_max = inv_num_max.contiguous().view(inv_num_max.shape[0],inv_num_max.shape[1],inv_num_max.shape[2], inv_num_max.shape[3])
        # Step 2: Scale grad_output by 1 / num_max
        grad_scaled = grad_output.f.mul_zip(grad_output, inv_num_max)  # Shape: (batch, channels, new_height, new_width)

        grad_scaled = grad_scaled.contiguous().view(grad_scaled.shape[0], grad_scaled.shape[1], grad_scaled.shape[2], grad_scaled.shape[3], 1)
        # Step 3: Multiply grad_scaled with mask
        grad_input_tiled = mask.f.mul_zip(mask, grad_scaled)  # Shape: (batch, channels, new_height, new_width, kh * kw)

        # Step 4: Reshape grad_input_tiled back to (batch, channels, height, width)
        grad_input = grad_input_tiled.contiguous().view(mask.shape[0], mask.shape[1], mask.shape[2] * kh, mask.shape[3] * kw)

        return grad_input, -1

def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Compute 2D max pooling on a tensor.

    Args:
        input (Tensor): Input tensor of shape (batch, channel, height, width).
        kernel (Tuple[int, int]): Tuple specifying the (kernel_height, kernel_width).

    Returns:
        Tensor: Pooled tensor of shape (batch, channel, new_height, new_width).

    """
    assert len(input.shape) == 4, f"Input tensor must be 4D, but got shape {input.shape}"
    kernel_tensor = Tensor.make([kernel[0], kernel[1]], shape=(2,), backend=input.backend)
    return MaxPool2dFun.apply(input, kernel_tensor)

class Dropout(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, p_tensor: Tensor) -> Tensor:  # noqa: D417
        """Apply dropout to the input tensor.

        Args:
            ctx (Context): Context to save information for backward computation.
            input (Tensor): Input tensor.
            p (Tensor): Probability of an element to be zeroed.
            train (bool): Whether in training mode.

        Returns:
            Tensor: Tensor after applying dropout.

        """
        p = p_tensor.item()  # Keep p as float
        # Generate mask with shape as tuple
        mask_array = (np.random.rand(*input.shape) > p).astype(np.float32).flatten()
        mask = Tensor.make(
            mask_array,
            shape=tuple(input.shape),
            backend=input.backend
        )
        # Scale the output
        if p == 1.0:
            # Use np.zeros with shape as tuple
            zeros = np.zeros(input.shape).astype(np.float32).flatten()
            return Tensor.make(
                zeros,
                shape=tuple(input.shape),
                backend=input.backend
            )


        # Compute output
        output = input * mask / (1.0 - p)
        ctx.save_for_backward(mask, p)        
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, int]:
        """Backward pass for dropout.

        Args:
            ctx (Context): Context containing saved tensors.
            grad_output (Tensor): Gradient of the loss with respect to the output tensor.

        Returns:
            Tuple[Tensor, None, None]: Gradient with respect to the input tensor, None for p, and None for train.

        """
        mask, p = ctx.saved_tensors
        grad_input = grad_output * mask / (1.0 - p)
        return grad_input, -1

def dropout(input: Tensor, p: float, ignore: Optional[bool] = None) -> Tensor:  # noqa: D417
    """Apply dropout to the input tensor.

    Args:
        input (Tensor): Input tensor.
        p (float): Probability of an element to be zeroed.
        train (bool): Whether in training mode.

    Returns:
        Tensor: Tensor after applying dropout.

    """
    if ignore:
        return input
    
    p_tensor = Tensor.make([p], shape=(1,), backend=input.backend)
    return Dropout.apply(input, p_tensor)
