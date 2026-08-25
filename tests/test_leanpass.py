import numpy as np

from leanpass import Tensor, nn, optim


def test_tensor_basic_arithmetic():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = Tensor([3.0, 4.0], requires_grad=True)

    z = x * y + x
    z_sum = z.sum()
    z_sum.backward()

    assert np.allclose(x.grad, [4.0, 5.0])
    assert np.allclose(y.grad, [1.0, 2.0])


def test_linear_forward_backward():
    layer = nn.Linear(2, 1)
    x = Tensor([[1.0, 2.0]], requires_grad=False)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert layer.weight.grad.shape == layer.weight.data.shape
    assert layer.bias.grad.shape == layer.bias.data.shape


def test_mlp_training_step():
    model = nn.MLP([2, 4, 1])
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    x = Tensor([[1.0, 1.0]], requires_grad=False)
    y_true = Tensor([[2.0]], requires_grad=False)

    pred = model(x)
    loss = nn.mse_loss(pred, y_true)
    model.zero_grad()
    loss.backward()
    optimizer.step()

    assert all(param.grad is not None for param in model.parameters())
    assert any(np.any(param.data != 0) for param in model.parameters())


def test_cross_entropy_loss_backward():
    logits = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    target = Tensor([[0.0, 0.0, 1.0]], requires_grad=False)

    loss = nn.cross_entropy_loss(logits, target)
    loss.backward()

    assert loss.data.shape == ()
    assert logits.grad.shape == logits.data.shape
    assert np.all(logits.grad != 0)


def test_binary_cross_entropy_loss_backward():
    logits = Tensor([0.2, -1.0], requires_grad=True)
    target = Tensor([1.0, 0.0], requires_grad=False)

    loss = nn.binary_cross_entropy_loss(logits, target)
    loss.backward()

    assert loss.data.shape == ()
    assert logits.grad.shape == logits.data.shape
    assert np.all(logits.grad != 0)


def test_tensor_tanh_backward():
    x = Tensor([0.5, -0.5], requires_grad=True)
    y = x.tanh().sum()
    y.backward()

    assert y.data.shape == ()
    assert x.grad.shape == x.data.shape
    assert np.all(np.isfinite(x.grad))
    assert np.all(x.grad != 0)


def test_tensor_leaky_relu_backward():
    x = Tensor([-1.0, 2.0], requires_grad=True)
    y = x.leaky_relu(negative_slope=0.05).sum()
    y.backward()

    assert y.data.shape == ()
    assert x.grad.shape == x.data.shape
    assert np.allclose(x.grad, [0.05, 1.0])


def test_tensor_gelu_backward():
    x = Tensor([0.1, -0.1], requires_grad=True)
    y = x.gelu().sum()
    y.backward()

    assert y.data.shape == ()
    assert x.grad.shape == x.data.shape
    assert np.all(np.isfinite(x.grad))
    assert np.all(x.grad != 0)


def test_truediv_raises_on_zero_divisor():
    import pytest
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = Tensor([1.0, 0.0], requires_grad=False)
    with pytest.raises(ZeroDivisionError):
        _ = x / y
