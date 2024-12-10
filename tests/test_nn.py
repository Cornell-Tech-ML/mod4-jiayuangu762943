import pytest  # type: ignore
from hypothesis import given  # type: ignore

import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors
import numpy as np  # type: ignore


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))  # type: ignore
def test_avg(t: Tensor) -> None:
    print(t)
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))  # type: ignore
def test_max(t: Tensor) -> None:
    # TODO: Implement for Task 4.4.
    # Convert to numpy for easy checking
    np_t = t.to_numpy()
    np_t = np_t.reshape((2, 3, 4))

    # Test max along dimension 0
    # Expect shape: (3, 4)
    out_0 = minitorch.max(t, 0)
    np_out_0 = np.max(np_t, axis=0)

    # Compare all elements
    for i0 in range(np_out_0.shape[0]):
        for i1 in range(np_out_0.shape[1]):
            assert_close(out_0[i0, i1], np_out_0[i0, i1])

    # Test max along dimension 1
    # Expect shape: (2, 4)
    out_1 = minitorch.max(t, 1)
    np_out_1 = np.max(np_t, axis=1)
    for i0 in range(np_out_1.shape[0]):
        for i1 in range(np_out_1.shape[1]):
            assert_close(out_1[i0, i1], np_out_1[i0, i1])

    # Test max along dimension 2
    # Expect shape: (2, 3)
    out_2 = minitorch.max(t, 2)
    np_out_2 = np.max(np_t, axis=2)
    for i0 in range(np_out_2.shape[0]):
        for i1 in range(np_out_2.shape[1]):
            assert_close(out_2[i0, i1], np_out_2[i0, i1])

    # Perform gradient check on dim=2
    storage = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 5.0]]], dtype=np.float32)
    another_t = Tensor.make(
        storage.flatten().tolist(), shape=(1, 2, 3), backend=t.backend
    )

    # Perform gradient check on dim=2
    minitorch.grad_check(lambda a: minitorch.max(a, dim=2), another_t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))  # type: ignore
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())  # type: ignore
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))  # type: ignore
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))  # type: ignore
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)
