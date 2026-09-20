import numpy as np
import pytest
from leanpass import Tensor


def test_log_numerical_stability():
    # Non-positive inputs should be clamped and not produce -inf or nan
    x = Tensor([0.0, -5.0, -100.0, 1e-20, 1.0, 2.71828], requires_grad=True)
    out = x.log()
    assert not np.isnan(out.data).any()
    assert not np.isneginf(out.data).any()
    assert not np.isposinf(out.data).any()

    # Backward pass should not blow up to nan or inf
    loss = out.sum()
    loss.backward()
    assert not np.isnan(x.grad).any()
    assert not np.isinf(x.grad).any()


def test_exp_numerical_stability():
    # Large positive inputs should not overflow to inf
    x = Tensor([0.0, 10.0, 700.0, 1000.0, -1000.0], requires_grad=True)
    out = x.exp()
    assert not np.isnan(out.data).any()
    assert not np.isposinf(out.data).any()
    assert (out.data >= 0).all()

    # Backward pass
    loss = out.sum()
    loss.backward()
    assert not np.isnan(x.grad).any()
    assert not np.isposinf(x.grad).any()


def test_sigmoid_numerical_stability():
    # Large negative and large positive values should be stably computed
    x = Tensor([-1000.0, -500.0, -10.0, 0.0, 10.0, 500.0, 1000.0], requires_grad=True)
    out = x.sigmoid()
    assert not np.isnan(out.data).any()
    assert not np.isinf(out.data).any()
    assert np.all((out.data >= 0.0) & (out.data <= 1.0))
    # Close to 0 for large negative, close to 1 for large positive, 0.5 for 0
    assert np.isclose(out.data[0], 0.0)
    assert np.isclose(out.data[3], 0.5)
    assert np.isclose(out.data[6], 1.0)

    # Backward pass
    loss = out.sum()
    loss.backward()
    assert not np.isnan(x.grad).any()
    assert not np.isinf(x.grad).any()
    assert np.all((x.grad >= 0.0) & (x.grad <= 0.25 + 1e-6))


def test_softmax_numerical_stability():
    # Extreme inputs should not overflow or produce NaNs
    x = Tensor([[1000.0, 1001.0, 1002.0], [-1000.0, -1000.0, -1000.0]], requires_grad=True)
    out = x.softmax(axis=-1)
    assert not np.isnan(out.data).any()
    assert not np.isinf(out.data).any()
    assert np.all((out.data >= 0.0) & (out.data <= 1.0))
    # Probabilities must sum to 1 along the specified axis
    row_sums = out.data.sum(axis=-1)
    assert np.allclose(row_sums, [1.0, 1.0])

    # Backward pass
    loss = out.sum()
    loss.backward()
    assert not np.isnan(x.grad).any()
    assert not np.isinf(x.grad).any()


def test_eval_forward_stability():
    # Verify graph forward evaluation also respects numerical stability guards
    x = Tensor([-500.0, 0.0, 500.0], requires_grad=True)
    sig = x.sigmoid()
    assert not np.isnan(sig._eval_forward()).any()
    assert not np.isinf(sig._eval_forward()).any()

    y = Tensor([0.0, -1.0, 10.0], requires_grad=True)
    lg = y.log()
    assert not np.isnan(lg._eval_forward()).any()
    assert not np.isneginf(lg._eval_forward()).any()
