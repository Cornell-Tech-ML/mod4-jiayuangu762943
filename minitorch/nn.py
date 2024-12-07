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
        softmax_array, dim = ctx.saved_tensors
        grad_output_array = grad_output.to_numpy()

        # Compute grad_input
        grad_input_array = softmax_array * (grad_output_array - np.sum(grad_output_array * softmax_array, axis=dim, keepdims=True))

        grad_input = Tensor.make(grad_input_array.flatten(), shape=grad_output.shape, backend=grad_output.backend)

        return grad_input, -1
    
def softmax(input: Tensor, dim: int) -> Tensor:
    dim_tensor = Tensor.make([dim], shape=(1,), backend=input.backend)
    return Softmax.apply(input, dim_tensor)


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim_tensor: Tensor) -> Tensor:
        dim = int(dim_tensor.item())
        input_array = input.to_numpy()


        # Subtract max for numerical stability

        max_vals_array = np.max(input_array, axis=dim)
        # Expand max_vals_array to match input_array's shape
        shape = list(input.shape)
        shape[dim] = 1
        expanded_max_vals = max_vals_array.reshape(shape)
        x_shifted = input_array - expanded_max_vals

        exp_x_shifted = np.exp(x_shifted)

        # Sum exp_x_shifted along the specified dimension
        sum_exp_x = np.sum(exp_x_shifted, axis=dim, keepdims=True)
        logsumexp = np.log(sum_exp_x)
        output_array = x_shifted - logsumexp

        # Save softmax_array and dimension for backward
        softmax_array = np.exp(output_array)
        
        ctx.save_for_backward(softmax_array, dim)

        return Tensor.make(output_array.flatten(), shape=input.shape, backend=input.backend)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, int]:
        softmax_array, dim = ctx.saved_tensors
        grad_output_array = grad_output.to_numpy()

        # Compute the sum of grad_output_array along the specified dimension
        sum_grad = np.sum(grad_output_array, axis=dim, keepdims=True)

        # Compute grad_input_array based on the correct derivative
        grad_input_array = grad_output_array - softmax_array * sum_grad

        # Flatten grad_input_array for Tensor.make
        grad_input_flat = grad_input_array.flatten()

        # Create a Tensor for grad_input
        grad_input = Tensor.make(
            grad_input_flat,
            shape=tuple(grad_output.shape),
            backend=grad_output.backend
        )
        return grad_input, -1

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    dim_tensor = Tensor.make([dim], shape=(1,), backend=input.backend)
    return LogSoftmax.apply(input, dim_tensor)



class MaxPool2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, kernel: Tensor) -> Tensor:
        kh, kw = int(kernel._tensor._storage[0]), int(kernel._tensor._storage[1])
        tiled, new_height, new_width = tile(input, (kh, kw))

        # Compute max along the last dimension (kh * kw)
        input_array = tiled._tensor._storage.reshape(tiled.shape)
        max_vals = np.max(input_array, axis=4)
        max_indices = np.argmax(input_array, axis=len(tiled.shape)-1)

        ctx.save_for_backward(tiled.shape, max_indices, kh, kw)

        max_vals_tensor = Tensor.make(max_vals.flatten(), shape=(tiled.shape[0], tiled.shape[1], new_height, new_width), backend=input.backend)

        return max_vals_tensor

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:
        tiled_shape, max_indices, kh, kw = ctx.saved_tensors
        batch, channel, new_height, new_width, _ = tiled_shape
        grad_output_array = grad_output._tensor._storage.reshape(grad_output.shape)

        grad_input_tiled = np.zeros(tiled_shape, dtype=grad_output_array.dtype)

        # Distribute gradients to the positions of max indices
        for b in range(batch):
            for c in range(channel):
                for h in range(new_height):
                    for w in range(new_width):
                        idx = max_indices[b, c, h, w]
                        grad_input_tiled[b, c, h, w, idx] = grad_output_array[b, c, h, w]

        grad_input_tiled_tensor = Tensor.make(grad_input_tiled.flatten(), shape=tiled_shape)

        # Reshape grad_input_tiled back to the input shape
        grad_input = grad_input_tiled_tensor.view(batch, channel, new_height, new_width, kh, kw)
        grad_input = grad_input.permute(0, 1, 2, 4, 3, 5).contiguous()
        grad_input = grad_input.view(batch, channel, new_height * kh, new_width * kw)
        return grad_input, None

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
    def forward(ctx: Context, input: Tensor, p_tensor: Tensor) -> Tensor:
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

def dropout(input: Tensor, p: float, ignore: Optional[bool] = None) -> Tensor:
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
