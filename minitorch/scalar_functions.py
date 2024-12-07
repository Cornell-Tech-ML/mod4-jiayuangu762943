from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple."""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to the given scalar values.

        Converts inputs to `Scalar` objects if necessary, performs the forward pass,
        and sets up the backward pass.

        Args:
        ----
            cls: The class of the function being applied.
            *vals (ScalarLike): The input values to the function.

        Returns:
        -------
            Scalar: The result of applying the function to the inputs, as a `Scalar` object.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for addition.

        Args:
        ----
            ctx (Context): The context.
            a (float): First input value.
            b (float): Second input value.

        Returns:
        -------
            float: The sum of a and b.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for addition.

        Computes gradients with respect to inputs.

        Args:
        ----
            ctx (Context): The context.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, float]: Gradients with respect to a and b.

        """
        return d_output, d_output


class Log(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for logarithm.

        Saves input for use in the backward pass.

        Args:
        ----
            ctx (Context): The context to save intermediate variables.
            a (float): Input value.

        Returns:
        -------
            float: The natural logarithm of a.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for logarithm.

        Computes gradient with respect to input.

        Args:
        ----
            ctx (Context): The context with saved variables.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: Gradient with respect to a.

        """
        (a,) = ctx.saved_values
        return d_output / a


# To implement.


class Mul(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication.

        Saves inputs for use in the backward pass.

        Args:
        ----
            ctx (Context): The context to save intermediate variables.
            a (float): First input value.
            b (float): Second input value.

        Returns:
        -------
            float: The product of a and b.

        """
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for multiplication.

        Computes gradients with respect to inputs.

        Args:
        ----
            ctx (Context): The context with saved variables.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, float]: Gradients with respect to a and b.

        """
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inversion.

        Saves input for use in the backward pass.

        Args:
        ----
            ctx (Context): The context to save intermediate variables.
            a (float): Input value.

        Returns:
        -------
            float: The inverse of a.

        """
        ctx.save_for_backward(a)
        return 1.0 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for inversion.

        Computes gradient with respect to input.

        Args:
        ----
            ctx (Context): The context with saved variables.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: Gradient with respect to a.

        """
        (a,) = ctx.saved_values
        return -d_output / (a**2)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation.

        Args:
        ----
            ctx (Context): The context.
            a (float): Input value.

        Returns:
        -------
            float: The negated value of a.

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for negation.

        Computes gradient with respect to input.

        Args:
        ----
            ctx (Context): The context.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: Gradient with respect to a.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for sigmoid.

        Saves the sigmoid value for use in the backward pass.

        Args:
        ----
            ctx (Context): The context to save intermediate variables.
            a (float): Input value.

        Returns:
        -------
            float: The sigmoid of a.

        """
        sigmoid_val = 1 / (1 + operators.exp(-a))
        ctx.save_for_backward(sigmoid_val)
        return sigmoid_val

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid.

        Computes gradient with respect to input.

        Args:
        ----
            ctx (Context): The context with saved variables.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: Gradient with respect to a.

        """
        (sigmoid_val,) = ctx.saved_values
        return d_output * sigmoid_val * (1 - sigmoid_val)


class Relu(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for ReLU.

        Saves input for use in the backward pass.

        Args:
        ----
            ctx (Context): The context to save intermediate variables.
            a (float): Input value.

        Returns:
        -------
            float: The ReLU of a.

        """
        ctx.save_for_backward(a)
        return max(0.0, a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for ReLU.

        Computes gradient with respect to input.

        Args:
        ----
            ctx (Context): The context with saved variables.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: Gradient with respect to a.

        """
        (a,) = ctx.saved_values
        return d_output if a > 0 else 0.0


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^{x}$."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exponential.

        Saves the exponential value for use in the backward pass.

        Args:
        ----
            ctx (Context): The context to save intermediate variables.
            a (float): Input value.

        Returns:
        -------
            float: The exponential of a.

        """
        exp_val = operators.exp(a)
        ctx.save_for_backward(exp_val)
        return exp_val

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exponential.

        Computes gradient with respect to input.

        Args:
        ----
            ctx (Context): The context with saved variables.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: Gradient with respect to a.

        """
        (exp_val,) = ctx.saved_values
        return d_output * exp_val


class Lt(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less-than comparison.

        Args:
        ----
            ctx (Context): The context.
            a (float): First input value.
            b (float): Second input value.

        Returns:
        -------
            float: 1.0 if a < b, else 0.0.

        """
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less-than comparison.

        Since the function is not differentiable, gradients are zero.

        Args:
        ----
            ctx (Context): The context.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, float]: Zero gradients with respect to a and b.

        """
        return (0.0, 0.0)


class Eq(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equality comparison.

        Args:
        ----
            ctx (Context): The context.
            a (float): First input value.
            b (float): Second input value.

        Returns:
        -------
            float: 1.0 if a == b, else 0.0.

        """
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equality comparison.

        Since the function is not differentiable, gradients are zero.

        Args:
        ----
            ctx (Context): The context.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, float]: Zero gradients with respect to a and b.

        """
        return (0.0, 0.0)


# TODO: Implement for Task 1.2.
