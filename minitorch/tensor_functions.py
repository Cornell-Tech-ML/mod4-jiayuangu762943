"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for negation.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The negated tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for negation.

        Args:
        ----
            ctx (Context): The context with saved variables.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient with respect to the input.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for reciprocal.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The reciprocal of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for reciprocal.

        Args:
        ----
            ctx (Context): The context with saved variables.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient with respect to the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for addition.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The sum of the input tensors.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for addition.

        Args:
        ----
            ctx (Context): The context with saved variables.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to each input tensor.

        """
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Forward pass to check if all elements are True.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            a (Tensor): The input tensor.
            dim (Optional[int]): The dimension to reduce over.

        Returns:
        -------
            Tensor: A tensor containing 1 if all elements are True, 0 otherwise.

        """
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(
                a.contiguous().view(int(operators.prod([ele for ele in a.shape]))), 0
            )


# TODO: Implement for Task 2.3.
class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for multiplication.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The product of the input tensors.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for multiplication.

        Args:
        ----
            ctx (Context): The context with saved variables.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to each input tensor.

        """
        t1, t2 = ctx.saved_values
        grad_t1 = grad_output.f.mul_zip(grad_output, t2)
        grad_t2 = grad_output.f.mul_zip(grad_output, t1)
        return grad_t1, grad_t2


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for sigmoid.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The result after applying the sigmoid function.

        """
        result = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for sigmoid.

        Args:
        ----
            ctx (Context): The context with saved variables.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient with respect to the input.

        """
        (sigmoid_t1,) = ctx.saved_values
        one = minitorch.Tensor.make(
            [1.0] * int(operators.prod(sigmoid_t1.shape)),
            shape=sigmoid_t1.shape,
            strides=None,
            backend=sigmoid_t1.backend,
        )

        grad = grad_output.f.mul_zip(
            grad_output, sigmoid_t1.f.mul_zip(sigmoid_t1, one - sigmoid_t1)
        )
        return grad


class ReLU(Function):
    """ReLU activation function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for ReLU.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The result after applying the ReLU function.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for ReLU.

        Args:
        ----
            ctx (Context): The context with saved variables.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient with respect to the input.

        """
        (t1,) = ctx.saved_values
        grad = grad_output.f.relu_back_zip(t1, grad_output)
        return grad


class Log(Function):
    """Natural logarithm function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for logarithm.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The natural logarithm of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for logarithm.

        Args:
        ----
            ctx (Context): The context with saved variables.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient with respect to the input.

        """
        (t1,) = ctx.saved_values
        grad = grad_output.f.log_back_zip(t1, grad_output)
        return grad


class Exp(Function):
    """Exponential function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for exponential.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The exponential of the input tensor.

        """
        result = t1.f.exp_map(t1)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for exponential.

        Args:
        ----
            ctx (Context): The context with saved variables.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient with respect to the input.

        """
        (exp_t1,) = ctx.saved_values
        grad = grad_output.f.mul_zip(grad_output, exp_t1)
        return grad


class Sum(Function):
    """Summation function over tensor elements."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for summation.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            t1 (Tensor): The input tensor.
            dim (Optional[int]): The dimension to reduce over.

        Returns:
        -------
            Tensor: The sum of the tensor elements.

        """
        if dim.shape == (1,) and int(dim.item()) == -1:
            # no dim argument is passed in
            ctx.save_for_backward(t1, int(dim.item()))
            size = int(operators.prod([ele for ele in t1.shape]))
            return t1.f.add_reduce(t1.contiguous().view(size), 0)
        else:
            dim_int = int(dim.item())
            ctx.save_for_backward(t1, dim_int)
            return t1.f.add_reduce(t1, dim_int)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, int]:
        """Backward pass for summation.

        Args:
        ----
            ctx (Context): The context with saved variables.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor]: The gradient with respect to the input tensor.

        """
        t1, dim = ctx.saved_values
        res = t1.expand(grad_output)
        return (res, dim)


class LT(Function):
    """Element-wise less-than comparison."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for less-than comparison.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A tensor of booleans where each element is True if t1 < t2.

        """
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for less-than comparison.

        Args:
        ----
            ctx (Context): The context with saved variables.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Zero gradients since comparison operators are not differentiable.

        """
        grad_t1 = grad_output.zeros()
        grad_t2 = grad_output.zeros()
        return grad_t1, grad_t2

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

        # Step 1: Compute max tensor using reduce with operators.max
        max_tensor = t1.f.max_reduce(t1, dim_int)
        # Step 2: Create mask tensor where input == max_tensor using zip with operators.eq
        mask = t1.f.eq_zip(t1, max_tensor)

        # Step 3: Count number of maxima along the dimension using reduce with operators.sum
        num_max = mask.f.add_reduce(mask, dim_int)

        # Step 4: Save mask and num_max for backward pass
        ctx.save_for_backward(mask, num_max, dim_int)

        return max_tensor

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, int]:
        """Backward pass for max.

        Args:
            ctx (Context): The context with saved tensors.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tuple[Tensor, None]: The gradient with respect to the input tensor and None for dim.

        """
        mask, num_max, dim_int = ctx.saved_tensors

        # Step 1: Compute inverse of num_max using map with operators.inv
        inv_num_max = num_max.f.inv_map(num_max)

        # Step 2: Scale grad_output by inv_num_max using zip with operators.mul
        grad_scaled = grad_output.f.mul_zip(grad_output, inv_num_max)

        # Step 3: Distribute scaled gradients only to positions where mask == 1 using zip with operators.mul
        grad_input = mask.f.mul_zip(mask, grad_scaled)
        return grad_input, -1
    
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

class EQ(Function):
    """Element-wise equality comparison."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for equality comparison.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A tensor of booleans where each element is True if t1 == t2.

        """
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for equality comparison.

        Args:
        ----
            ctx (Context): The context with saved variables.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Zero gradients since equality comparison is not differentiable.

        """
        grad_t1 = grad_output.zeros()
        grad_t2 = grad_output.zeros()
        return grad_t1, grad_t2


class IsClose(Function):
    """Element-wise approximation check."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass to check if elements are close.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A tensor of booleans where each element is True if t1 is close to t2.

        """
        return t1.f.is_close_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> None:
        """Backward pass for is_close.

        Args:
        ----
            ctx (Context): The context with saved variables.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            None: No gradient computation needed.

        """
        return None


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tensor) -> Tensor:
        """Forward pass for permuting tensor dimensions.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            t1 (Tensor): The input tensor.
            order (Tensor): A tensor containing the new order of dimensions.

        Returns:
        -------
            Tensor: The permuted tensor.

        """
        order_tuple = tuple(int(order[i]) for i in range(order.size))
        ctx.save_for_backward(order_tuple)
        # Permute the tensor dimensions
        return t1._new(t1._tensor.permute(*order_tuple))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, int]:
        """Backward pass for permuting tensor dimensions.

        Args:
        ----
            ctx (Context): The context with saved variables.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, None]: The gradient with respect to the input tensor and None for the 'order' gradient.

        """
        (order,) = ctx.saved_values
        # Compute inverse permutation
        inv_order = [0] * len(order)
        for i, o in enumerate(order):
            inv_order[o] = i
        # Permute gradient back to original order
        return (grad_output._new(grad_output._tensor.permute(*inv_order)), -1)


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Forward pass for viewing tensor with a new shape.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            a (Tensor): The input tensor.
            shape (Tensor): A tensor containing the new shape.

        Returns:
        -------
            Tensor: A new tensor with the specified shape.

        Raises:
        ------
            AssertionError: If the tensor is not contiguous.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod([ele for ele in shape])), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod([ele for ele in shape])))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the central difference approximation of the gradient.

    Args:
    ----
        f (Any): The function to differentiate.
        *vals (Tensor): The input tensors.
        arg (int, optional): The index of the argument to compute the gradient with respect to.
        epsilon (float, optional): The small shift for computing the difference. Defaults to 1e-6.
        ind (UserIndex): The index at which to compute the gradient.

    Returns:
    -------
        float: The approximated gradient at the specified index.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )

        # # Determine if the current index is a maximum in its slice
        # dim = 2  # Adjust this if testing different dimensions
        # slice_indices = list(ind[:dim]) + [slice(None)]
        # slice_vals = x.to_numpy()[tuple(slice_indices)]
        # max_val = np.max(slice_vals)
        # num_max = np.sum(slice_vals == max_val)
        # is_max = x[ind] == max_val
        
        # # If the index is a maximum, expected gradient is 1.0 / num_max
        # # Otherwise, expected gradient is 0.0
        # expected_grad = (1.0 / num_max) if is_max else 0.0
        
        # # Get received gradient from backward pass
        # received_grad = x.grad[ind]
        
        # # Compare gradients
        # np.testing.assert_allclose(
        #     received_grad,
        #     expected_grad,
        #     rtol=1e-2,
        #     atol=1e-2,
        #     err_msg=err_msg % (f, vals, received_grad, i, ind, expected_grad),
        # )
