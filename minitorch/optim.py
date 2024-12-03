from typing import Sequence

from .module import Parameter
from .scalar import Scalar


class Optimizer:
    def __init__(self, parameters: Sequence[Parameter]):
        """Base class for all optimizers.

        Args:
        ----
            parameters (Sequence[Parameter]): An iterable of parameters to optimize.

        """
        self.parameters = parameters


class SGD(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        """Stochastic Gradient Descent (SGD) optimizer.

        Args:
        ----
            parameters (Sequence[Parameter]): An iterable of parameters to optimize.
            lr (float, optional): Learning rate. Defaults to 1.0.

        """
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Reset the gradients of all optimized parameters to zero.

        This method should be called before computing the gradients
        in a new optimization step to prevent accumulation of gradients
        from multiple backward passes.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        """Perform a single optimization step.

        Updates the parameters based on the gradients computed during the backward pass.

        This method subtracts the product of the learning rate and the gradient
        from each parameter's value.

        For each parameter `p`:

        - If `p.value` has an attribute `derivative`, and it is not `None`, update
          `p.value` by subtracting `lr * p.value.derivative`.

        - Else if `p.value` has an attribute `grad`, and it is not `None`, update
          `p.value` by subtracting `lr * p.value.grad`.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)
