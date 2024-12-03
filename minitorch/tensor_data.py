from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Convert a multidimensional tensor `index` into a single-dimensional position in storage based on strides.

    Args:
    ----
        index (Index): Index tuple of ints.
        strides (Strides): Tensor strides.

    Returns:
    -------
        int: Position in storage.

    """
    # TODO: Implement for Task 2.1.
    position = 0
    for ind, stride in zip(index, strides):
        position += ind * stride

    return position


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.

    Should ensure that enumerating position 0 ... size of a tensor produces every index exactly once.
    It may not be the inverse of `index_to_position`.

    Args:
    ----
        ordinal (int): Ordinal position to convert.
        shape (Shape): Tensor shape.
        out_index (OutIndex): Return index corresponding to position.

    """
    # TODO: Implement for Task 2.1.
    cur_ord = ordinal + 0
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_ord % shape[i])
        cur_ord = cur_ord // sh


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Convert a `big_index` in `big_shape` to a smaller `out_index` in `shape` following broadcasting rules.

    In this case, it may be larger or with more dimensions than the `shape` given.
    Additional dimensions may need to be mapped to 0 or removed.

    Args:
    ----
        big_index (Index): Multidimensional index of bigger tensor.
        big_shape (Shape): Tensor shape of bigger tensor.
        shape (Shape): Tensor shape of smaller tensor.
        out_index (OutIndex): Multidimensional index of smaller tensor.

    """
    for i, s in enumerate(shape):
        if s > 1:
            out_index[i] = big_index[i + len(big_shape) - len(shape)]
        else:
            out_index[i] = 0

    return None


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
    ----
        shape1 (UserShape): First shape.
        shape2 (UserShape): Second shape.

    Returns:
    -------
        UserShape: Broadcasted shape.

    Raises:
    ------
        IndexingError: If shapes cannot be broadcast.

    """
    len1 = len(shape1)
    len2 = len(shape2)
    max_len = max(len1, len2)

    # Pad the shorter shape with ones at the front
    padded_shape1 = [1] * (max_len - len1) + list(shape1)
    padded_shape2 = [1] * (max_len - len2) + list(shape2)

    result_shape = []
    for dim1, dim2 in zip(padded_shape1, padded_shape2):
        if dim1 == dim2:
            result_shape.append(dim1)
        elif dim1 == 1:
            result_shape.append(dim2)
        elif dim2 == 1:
            result_shape.append(dim1)
        else:
            raise IndexingError("Cannot broadcast shapes")
    return tuple(result_shape)


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Return a contiguous stride for a given shape.

    Args:
    ----
        shape (UserShape): The shape of the tensor.

    Returns:
    -------
        UserStrides: The strides corresponding to the shape.

    """
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    """Manage the data and indexing for tensors.

    Attributes
    ----------
        _storage (Storage): The underlying storage array for tensor data.
        _strides (Strides): The strides for indexing into the storage.
        _shape (Shape): The shape of the tensor.
        strides (UserStrides): User-friendly strides.
        shape (UserShape): User-friendly shape.
        dims (int): Number of dimensions.
        size (int): Total number of elements.

    """

    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int
    size: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        """Initialize a TensorData object.

        Args:
        ----
            storage (Union[Sequence[float], Storage]): The data for the tensor.
            shape (UserShape): The shape of the tensor.
            strides (Optional[UserStrides]): The strides for the tensor. If None, will be calculated from the shape.

        """
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod([ele for ele in shape]))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Convert the storage to CUDA memory."""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check if the tensor data is stored contiguously.

        Returns
        -------
            bool: True if the data is contiguous, False otherwise.

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Broadcast two shapes to create a new union shape.

        Args:
        ----
            shape_a (UserShape): First shape.
            shape_b (UserShape): Second shape.

        Returns:
        -------
            UserShape: Broadcasted shape.

        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Convert a multidimensional index into a single-dimensional position in storage.

        Args:
        ----
            index (Union[int, UserIndex]): The index to convert.

        Returns:
        -------
            int: The position in storage.

        Raises:
        ------
            IndexingError: If the index is invalid.

        """
        if isinstance(index, int):
            aindex: Index = array([index])
        else:
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(aindex, self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Generate all possible indices for the tensor.

        Yields
        ------
            UserIndex: Each valid index for the tensor.

        """
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Get a random valid index for the tensor.

        Returns
        -------
            UserIndex: A random index within the tensor's shape.

        """
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Get the value at the specified index.

        Args:
        ----
            key (UserIndex): The index to access.

        Returns:
        -------
            float: The value at the specified index.

        """
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Set the value at the specified index.

        Args:
        ----
            key (UserIndex): The index to modify.
            val (float): The value to set.

        """
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return the core tensor data as a tuple.

        Returns
        -------
            Tuple[Storage, Shape, Strides]: The storage, shape, and strides.

        """
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
        ----
            *order (int): A permutation of the dimensions.

        Returns:
        -------
            TensorData: A new TensorData with permuted dimensions.

        Raises:
        ------
            AssertionError: If the provided order is not a valid permutation.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        new_shape = tuple(self.shape[i] for i in order)
        new_strides = tuple(self.strides[i] for i in order)
        return TensorData(self._storage, new_shape, new_strides)

    def to_string(self) -> str:
        """Convert the tensor data to a string representation.

        Returns
        -------
            str: The string representation of the tensor.

        """
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
