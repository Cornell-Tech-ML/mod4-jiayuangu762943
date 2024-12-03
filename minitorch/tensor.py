"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    # tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was used to construct the current Variable."""

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that handles multidimensional arrays."""

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        """Initialize a new Tensor.

        Args:
        ----
            v (TensorData): The underlying data storage for the tensor.
            back (Optional[History]): The history of operations for autodifferentiation.
            name (Optional[str]): An optional name for the tensor.
            backend (Optional[TensorBackend]): The backend to use for tensor operations.

        """
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Set whether this tensor requires gradient computation.

        Args:
        ----
            x (bool): If True, enables gradient computation for this tensor.

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Check if this tensor requires gradient computation.

        Returns
        -------
            bool: True if this tensor requires gradients, False otherwise.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Convert the tensor to a NumPy array.

        Returns
        -------
            numpy.ndarray: A NumPy array with the same data as this tensor.

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Ensure that the input is a tensor with the same backend.

        Args:
        ----
            b (TensorLike): A tensor-like object (float, int, or Tensor).

        Returns:
        -------
            Tensor: The input converted to a Tensor.

        """
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a Python float.

        Returns
        -------
            float: The single element of the tensor as a float.

        Raises
        ------
            AssertionError: If the tensor does not have exactly one element.

        """
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data.

        Returns
        -------
            Tensor: A contiguous tensor with the same data.

        """
        return Copy.apply(self)

    def __repr__(self) -> str:
        """Return a string representation of the tensor.

        Returns
        -------
            str: String representation of the tensor.

        """
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        """Get an item from the tensor.

        Args:
        ----
            key (int or tuple of ints): Index or indices to access.

        Returns:
        -------
            float: The value at the specified index.

        """
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        """Set an item in the tensor.

        Args:
        ----
            key (int or tuple of ints): Index or indices to access.
            val (float): The value to set at the specified index.

        """
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        """Set the backend for this tensor.

        Args:
        ----
            backend (TensorBackend): The backend to set.

        """
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        """Create a new tensor with the same backend from TensorData.

        Args:
        ----
            tensor_data (TensorData): The tensor data to use.

        Returns:
        -------
            Tensor: A new tensor with the given data and the same backend.

        """
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data.

        Args:
        ----
            storage (List[float] or Storage): The data for the tensor.
            shape (tuple of ints): The shape of the tensor.
            strides (Optional[tuple of ints]): The strides for the tensor.
            backend (Optional[TensorBackend]): The backend to use.

        Returns:
        -------
            Tensor: A new tensor with the given data and shape.

        """
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Expand the tensor to match the shape of another tensor for backpropagation.

        This method is used when the output of `backward` is a different size than the input of `forward`.

        Args:
        ----
            other (Tensor): The backward tensor (must broadcast with self).

        Returns:
        -------
            Tensor: Expanded version of `other` with the right derivatives.

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Create a tensor filled with zeros.

        Args:
        ----
            shape (Optional[tuple of ints]): The shape of the tensor. If None, uses self.shape.

        Returns:
        -------
            Tensor: A tensor of zeros with the specified shape.

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod([ele for ele in shape])),
                shape,
                backend=self.backend,
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple.

        Returns
        -------
            Tuple[Storage, Shape, Strides]: The storage, shape, and strides of the tensor.

        """
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach the tensor from the computation graph.

        Returns
        -------
            Tensor: A new tensor that does not require gradients.

        """
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add a value to the derivative accumulated on this variable.

        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x (Any): Value to be accumulated.

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod([ele for ele in self.shape])),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """Check if this variable was created by the user (no `last_fn`).

        Returns
        -------
            bool: True if this is a leaf variable, False otherwise.

        """
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if this tensor is constant (no history).

        Returns
        -------
            bool: True if this tensor is constant, False otherwise.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Get the parent variables of this tensor.

        Returns
        -------
            Iterable[Variable]: The parent variables.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to compute gradients with respect to inputs.

        Args:
        ----
            d_output (Any): The gradient of the output.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: Tuples of (input variable, gradient).

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Compute the gradients of this tensor with respect to its inputs.

        Args:
        ----
            grad_output (Optional[Tensor]): The gradient of the output. If None, uses a tensor of ones.

        Raises:
        ------
            AssertionError: If grad_output is None and the tensor is not a scalar.

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        """Divide this tensor by another tensor or scalar.

        Args:
        ----
            b (TensorLike): The denominator tensor or scalar.

        Returns:
        -------
            Tensor: The result of the division.

        """
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        """Divide a tensor or scalar by this tensor.

        Args:
        ----
            b (TensorLike): The numerator tensor or scalar.

        Returns:
        -------
            Tensor: The result of the division.

        """
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Perform matrix multiplication with another tensor.

        Args:
        ----
            b (Tensor): The right-hand tensor to multiply.

        Returns:
        -------
            Tensor: The result of the matrix multiplication.

        Note:
        ----
            Not used until Module 3.

        """
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Get the shape of the tensor.

        Returns
        -------
            tuple of ints: The shape of the tensor.

        """
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Get the total number of elements in the tensor.

        Returns
        -------
            int: The total number of elements.

        """
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Get the number of dimensions of the tensor.

        Returns
        -------
            int: The number of dimensions.

        """
        return self._tensor.dims

    def __add__(self, b: TensorLike) -> Tensor:
        """Add another tensor or scalar to this tensor.

        Args:
        ----
            b (TensorLike): The tensor or scalar to add.

        Returns:
        -------
            Tensor: The result of the addition.

        """
        b = self._ensure_tensor(b)
        return Add.apply(self, b)

    def __sub__(self, b: TensorLike) -> Tensor:
        """Subtract another tensor or scalar from this tensor.

        Args:
        ----
            b (TensorLike): The tensor or scalar to subtract.

        Returns:
        -------
            Tensor: The result of the subtraction.

        """
        b = self._ensure_tensor(b)
        return Add.apply(self, Neg.apply(b))

    def __mul__(self, b: TensorLike) -> Tensor:
        """Multiply this tensor by another tensor or scalar element-wise.

        Args:
        ----
            b (TensorLike): The tensor or scalar to multiply.

        Returns:
        -------
            Tensor: The result of the multiplication.

        """
        b = self._ensure_tensor(b)
        return Mul.apply(self, b)

    def __lt__(self, b: TensorLike) -> Tensor:
        """Element-wise less-than comparison between tensors.

        Args:
        ----
            b (TensorLike): The tensor or scalar to compare.

        Returns:
        -------
            Tensor: A tensor of booleans where each element is True if self < b.

        """
        b = self._ensure_tensor(b)
        return LT.apply(self, b)

    def __eq__(self, b: TensorLike) -> Tensor:
        """Element-wise equality comparison between tensors.

        Args:
        ----
            b (TensorLike): The tensor or scalar to compare.

        Returns:
        -------
            Tensor: A tensor of booleans where each element is True if self == b.

        """
        b = self._ensure_tensor(b)
        return EQ.apply(self, b)

    def __gt__(self, b: TensorLike) -> Tensor:
        """Element-wise greater-than comparison between tensors.

        Args:
        ----
            b (TensorLike): The tensor or scalar to compare.

        Returns:
        -------
            Tensor: A tensor of booleans where each element is True if self > b.

        """
        b = self._ensure_tensor(b)
        return LT.apply(b, self)  # a > b is equivalent to b < a

    def __neg__(self) -> Tensor:
        """Negate the tensor element-wise.

        Returns
        -------
            Tensor: The negated tensor.

        """
        return Neg.apply(self)

    def __radd__(self, b: TensorLike) -> Tensor:
        """Right-hand addition for scalar + tensor.

        Args:
        ----
            b (TensorLike): The scalar or tensor to add.

        Returns:
        -------
            Tensor: The result of the addition.

        """
        return self + b

    def __rmul__(self, b: TensorLike) -> Tensor:
        """Right-hand multiplication for scalar * tensor.

        Args:
        ----
            b (TensorLike): The scalar or tensor to multiply.

        Returns:
        -------
            Tensor: The result of the multiplication.

        """
        return self * b

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Check if all elements along a dimension are True.

        Args:
        ----
            dim (Optional[int]): The dimension to reduce over. If None, reduces over all dimensions.

        Returns:
        -------
            Tensor: A tensor containing the result.

        """
        if dim is None:
            return All.apply(self)
        else:
            dim_tensor = self._ensure_tensor(dim)
            return All.apply(self, dim_tensor)

    def is_close(self, b: TensorLike) -> Tensor:
        """Element-wise check if values are close between tensors.

        Args:
        ----
            b (TensorLike): The tensor or scalar to compare.

        Returns:
        -------
            Tensor: A tensor of booleans where each element is True if values are close.

        """
        b = self._ensure_tensor(b)
        return IsClose.apply(self, b)

    def sigmoid(self) -> Tensor:
        """Apply the sigmoid function element-wise.

        Returns
        -------
            Tensor: The result after applying sigmoid.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Apply the ReLU function element-wise.

        Returns
        -------
            Tensor: The result after applying ReLU.

        """
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Apply the natural logarithm function element-wise.

        Returns
        -------
            Tensor: The result after applying log.

        """
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Apply the exponential function element-wise.

        Returns
        -------
            Tensor: The result after applying exp.

        """
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Sum the tensor elements over a given dimension.

        Args:
        ----
            dim (Optional[int]): The dimension to reduce over. If None, sums over all dimensions.

        Returns:
        -------
            Tensor: The sum of elements.

        """
        if dim is None:
            # Sum over all dimensions
            dim_tensor = Tensor.make([-1], (1,), backend=self.backend)
            return Sum.apply(self, dim_tensor)
        else:
            dim_tensor = Tensor.make([dim], (1,), backend=self.backend)
            return Sum.apply(self, dim_tensor)

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Compute the mean of tensor elements over a given dimension.

        Args:
        ----
            dim (Optional[int]): The dimension to reduce over. If None, computes mean over all dimensions.

        Returns:
        -------
            Tensor: The mean of elements.

        """
        total = self.sum(dim)
        if dim is None:
            count = self.size
        else:
            count = self.shape[dim]
        return total / count

    def permute(self, *order: int) -> Tensor:
        """Permute the dimensions of the tensor according to a given order.

        Args:
        ----
            *order (int): The desired ordering of dimensions.

        Returns:
        -------
            Tensor: A new tensor with permuted dimensions.

        """
        # Convert the order tuple into a tensor
        order_tensor = Tensor.make(list(order), (len(order),), backend=self.backend)
        return Permute.apply(self, order_tensor)

    def view(self, *shape: int) -> Tensor:
        """Return a new tensor with the same data but a different shape.

        Args:
        ----
            *shape (int): The desired shape.

        Returns:
        -------
            Tensor: A new tensor with the specified shape.

        """
        shape_tensor = Tensor.make(list(shape), (len(shape),), backend=self.backend)
        return View.apply(self, shape_tensor)

    def zero_grad_(self) -> None:
        """Set the gradients of the tensor to zero."""
        self.grad = None
