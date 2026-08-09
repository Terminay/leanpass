# LeanPass

LeanPass is a lightweight NumPy-based autodiff library for building small neural network models and understanding automatic differentiation in a simple, readable way.

```bash
pip install leanpass
```

## Public API

### Tensor

The core data structure is `Tensor`, which wraps a NumPy array and supports automatic differentiation.

```python
from leanpass import Tensor

x = Tensor([[1.0, 2.0]], requires_grad=False)
y = Tensor([[3.0, 4.0]])
z = x + y
```

Supported operations:

- `Tensor + Tensor`
- `Tensor - Tensor`
- `Tensor * Tensor`
- `Tensor / Tensor`
- `Tensor ** Tensor`
- `Tensor @ Tensor`
- `Tensor.sum()`
- `Tensor.mean()`
- `Tensor.relu()`
- `Tensor.sigmoid()`
- `Tensor.tanh()`
- `Tensor.leaky_relu()`
- `Tensor.gelu()`
- `Tensor.softmax()`
- `Tensor.backward()`
 - `Tensor.clip(a_min, a_max)` / `Tensor.clamp(min, max)`
 - Basic indexing via `Tensor[...]` which participates in autodiff

### Neural network layers

```python
from leanpass import nn

layer = nn.Linear(4, 8)
mlp = nn.MLP([4, 16, 8])
```

Available components:

- `nn.Linear(in_features, out_features)` creates a linear layer with weights and bias.
- `nn.MLP(layer_sizes)` creates a multilayer perceptron with ReLU activations between layers.
 - `nn.Dropout(p=0.5)` creates an inverted-dropout layer for training-time regularization.
- `nn.mse_loss(predictions, targets)` computes mean squared error.
- `nn.cross_entropy_loss(predictions, targets)` computes categorical cross-entropy for multi-class targets.
- `nn.binary_cross_entropy_loss(predictions, targets)` computes binary cross-entropy for binary classification.

### Optimizers

```python
from leanpass import optim

optimizer = optim.SGD(model.parameters(), lr=0.01)
# or
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

Available optimizers (options):

- `optim.SGD(parameters, lr=..., momentum=0.0, weight_decay=0.0)` supports momentum and L2 weight decay.
- `optim.Adam(parameters, lr=..., weight_decay=0.0)` supports an optional L2 weight decay term.

### Example

```python
from leanpass import Tensor, nn, optim

x = Tensor([[1.0, 2.0]], requires_grad=False)
model = nn.MLP([2, 16, 3])
output = model(x)
print(output)
```

## Notes

This package is intended for clarity and educational use, with a compact implementation that makes the autodiff process easier to inspect.

Full documentation: [https://leanpass.kilobyte136.workers.dev/](https://leanpass.kilobyte136.workers.dev/)