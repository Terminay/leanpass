import numpy as np

from .tensor import Tensor


class Module:
    """A base class for layers and model containers."""

    def parameters(self):
        """Collect all trainable Tensor parameters recursively."""
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params

    def zero_grad(self):
        """Reset gradients before each optimization step."""
        for param in self.parameters():
            param.grad = np.zeros_like(param.data)


class Linear(Module):
    """A single fully connected layer: y = x @ W + b."""

    def __init__(self, in_features, out_features):
        scale = 1.0 / np.sqrt(in_features)
        self.weight = Tensor(
            np.random.uniform(-scale, scale, size=(in_features, out_features)),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    


class MLP(Module):
    """A simple feedforward network with ReLU activations."""

    def __init__(self, layer_sizes):
        self.layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(Linear(in_dim, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Propagate the input through each layer and apply ReLU except last."""
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = x.relu()
        return x

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self):
        """Return parameters from every Linear layer in the network."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class Dropout(Module):
    """Simple dropout layer.

    Usage: `Dropout(p=0.5)(x, training=True)` or call with `training=False` to
    disable dropout at evaluation time.
    """

    def __init__(self, p: float = 0.5):
        if not 0 <= p < 1:
            raise ValueError("p must be in the interval [0, 1)")
        self.p = float(p)

    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        if not training or self.p == 0.0:
            return x
        # create a binary mask and scale to preserve expectation (inverted dropout)
        mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float64) / (1.0 - self.p)
        return x * Tensor(mask)

    def __call__(self, x: Tensor, training: bool = True) -> Tensor:
        return self.forward(x, training=training)

    


def mse_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean squared error loss used for regression training."""
    return ((prediction - target) ** 2).mean()


def cross_entropy_loss(prediction: Tensor, target: Tensor, axis: int = -1) -> Tensor:
    """Categorical cross-entropy loss for one-hot target distributions.

    The function expects raw prediction logits and a target tensor with the same
    shape as the prediction, where each row is either a one-hot label vector or
    a target probability distribution.
    """
    probs = prediction.softmax(axis=axis)
    return -(target * probs.log()).sum(axis=axis).mean()


def binary_cross_entropy_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Binary cross-entropy loss for binary labels.

    The function accepts logits and applies a sigmoid before computing the loss.
    """
    probs = prediction.sigmoid()
    return -(target * probs.log() + (1 - target) * (1 - probs).log()).mean()
