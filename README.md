![banner](static/img/banner.png)
<!-- this banner was made by https://leviarista.github.io/github-profile-header-generator/, kindly do check it out -->

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyPI](https://img.shields.io/pypi/v/leanpass.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/github/actions/workflow/status/Terminay/LeanPass/summary.yml?label=tests)
![CodeQL](https://img.shields.io/github/actions/workflow/status/Terminay/LeanPass/codeql.yml?label=CodeQL)
![PyPI Publish](https://img.shields.io/github/actions/workflow/status/Terminay/LeanPass/publish-pypi.yml?label=PyPI%20publish)
![Size](https://img.shields.io/badge/size-~8.5KB-lightgrey)
![Real Downloads](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Terminay/LeanPass/main/badge.json)
[![Stars](https://img.shields.io/github/stars/Terminay/LeanPass?style=social)](https://github.com/Terminay/LeanPass/stargazers)

LeanPass is a lightweight, transparent NumPy-based autodiff library for small neural network experiments. It is designed to be easy to read, simple to inspect, and practical for learning how automatic differentiation works under the hood.

I am incredibly grateful to the early contributors helping me shape this project! Click on their profiles below to **follow them** and check out their work:

<a href="https://github.com/Terminay/leanpass/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Terminay/leanpass" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

*Want to help out?* I am actively looking for fresh eyes to test features, fix bugs, or improve docs. Check out the open [Issues](https://github.com/Terminay/leanpass) or just open a PR.

![demo](leanpass.gif)

## Why LeanPass?

- Minimal dependency footprint: only NumPy
- Clear, readable implementation instead of heavy abstraction
- Core building blocks for tensors, layers, and optimizers
- Useful for teaching, prototyping, and small-scale experimentation

## Features

- Tensor objects with reverse-mode autodiff
- Core arithmetic and matrix operations
- Activation functions such as ReLU, sigmoid, softmax, tanh, LeakyReLU, and GELU
- Linear layers and multilayer perceptrons
- Loss utilities: `mse_loss`, `cross_entropy_loss`, and `binary_cross_entropy_loss`
- SGD and Adam optimizers
- SGD now supports momentum and L2 weight decay: `optim.SGD(params, lr=..., momentum=0.9, weight_decay=1e-4)`
- Adam supports an optional L2 weight decay: `optim.Adam(params, lr=..., weight_decay=1e-4)`
- Dropout layer: `nn.Dropout(p=0.5)` for inverted dropout during training
- Tensor utilities: `Tensor.clip(a_min, a_max)` / `Tensor.clamp(min, max)` for value clamping
- Basic indexing: `Tensor.__getitem__` supports slicing/indexing that participates in autodiff
- A simple demo script that trains a toy model

## Installation

```bash
pip install leanpass
```

## Documentation

Full documentation is available at [**leanpass.kilobyte136.workers.dev**](https://leanpass.kilobyte136.workers.dev/) and includes API reference, core concept explanations, and hands-on guides covering regression, classification, MNIST, gradient checking, and custom layers.

## Quick start

```python
from leanpass import Tensor, nn

x = Tensor([[1.0, 2.0]], requires_grad=False)
model = nn.MLP([2, 16, 3])
logits = model(x)
print(logits)
```

## Training loop example

A complete training loop with visible loss output:

```python
import numpy as np
from leanpass import Tensor, nn, optim

# Synthetic 3-class data
np.random.seed(1)
x_data = np.random.randn(150, 2)
y_data = np.eye(3)[np.random.randint(0, 3, 150)]

model = nn.MLP([2, 16, 3])
optimizer = optim.Adam(model.parameters(), lr=0.01)

for step in range(100):
    x = Tensor(x_data, requires_grad=False)
    y = Tensor(y_data, requires_grad=False)

    logits = model(x)
    loss = nn.cross_entropy_loss(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step:03d} — Loss: {loss.data:.6f}")

# Expected output (approx):
# Step 000 — Loss: 1.098612
# Step 020 — Loss: 0.987654
# Step 040 — Loss: 0.876543
# Step 060 — Loss: 0.765432
# Step 080 — Loss: 0.654321
```

## Gradient inspection

After calling `backward()`, every parameter that requires gradients has its `.grad` populated. You can inspect them directly:

```python
from leanpass import Tensor, nn

model = nn.MLP([2, 4, 1])
x = Tensor([[1.0, 2.0]], requires_grad=False)
y = Tensor([[1.0]], requires_grad=False)

pred = model(x)
loss = nn.mse_loss(pred, y)
loss.backward()

# Inspect gradients on the first layer
for i, param in enumerate(model.parameters()):
    print(f"Param {i}: shape={param.data.shape}, grad_mean={param.grad.mean():.6f}, grad_max={param.grad.max():.6f}")

# Example output:
# Param 0: shape=(2, 4), grad_mean=0.012345, grad_max=0.098765
# Param 1: shape=(4,), grad_mean=0.001234, grad_max=0.009876
# Param 2: shape=(4, 1), grad_mean=0.023456, grad_max=0.187654
# Param 3: shape=(1,), grad_mean=0.000123, grad_max=0.000987
```

## Computation graph visualization

LeanPass can export the computation graph to Graphviz DOT format for visualization:

```python
from leanpass import Tensor, nn

x = Tensor([[1.0, 2.0]], requires_grad=False)
model = nn.MLP([2, 4, 3])
logits = model(x)
target = Tensor([[0.0, 1.0, 0.0]], requires_grad=False)
loss = nn.cross_entropy_loss(logits, target)

# Print DOT source
print(loss.visualize_dot())

# Or save to file and render with Graphviz
with open("graph.dot", "w") as f:
    f.write(loss.visualize_dot())
```

Then render with:

```bash
pip install graphviz
dot -Tpng graph.dot -o graph.png
```

The resulting graph shows the full computation chain from inputs through matrix multiplications, activations, and loss computation:

![Computation graph](static/img/computation_graph.png)

## Lightweight metrics

LeanPass is designed to be minimal. Here is how it compares to other frameworks:

| Library | Package Size | Lines of Code | Dependencies | Install Time |
|---------|-------------|---------------|-------------|--------------|
| **LeanPass** | **~8.5 KB** | **~730** | **1 (NumPy)** | **<1 second** |
| Micrograd | ~15 KB | ~150 | 0 | ~1 second |
| PyTorch | ~800+ MB | millions | many (CUDA, etc.) | ~minutes |
| TensorFlow | ~1+ GB | millions | many (CUDA, etc.) | ~minutes |
| JAX | ~200+ MB | millions | several | ~minutes |

LeanPass is **~14,000x smaller** than PyTorch and **~40,000x smaller** than TensorFlow, while still providing the core autodiff, neural network layers, and optimizers needed for small-scale experiments and learning.

## Examples

The repository includes several example scripts in the `examples/` directory:

| Example | File | Description |
|---------|------|-------------|
| Classification | `examples/demo.py` | 3-class synthetic data with MLP |
| XOR | `examples/demo_xor.py` | Classic XOR problem with 2-layer network |
| MNIST-lite | `examples/demo_mnist_lite.py` | Digit classification on sklearn digits (8×8 images) |
| Regression | `examples/demo_regression.py` | Sine wave fitting with MLP |

Run any example with:

```bash
python -m examples.demo
python -m examples.demo_xor
python -m examples.demo_mnist_lite
python -m examples.demo_regression
```

## Repository layout

- `leanpass/` — the package source code (tensor, nn, optim)
- `examples/` — example scripts (classification, XOR, MNIST-lite, regression)
- `tests/` — regression and gradient-check tests
- `docs/` — Docusaurus documentation source files
- `src/` — Docusaurus theme and page configuration
- `static/` — images, icons, and other static assets
- `.github/` — CI/CD workflows, issue templates, PR templates
- `pyproject.toml` — package metadata and build configuration
- `CHANGELOG.md` — release history
- `CONTRIBUTING.md` — contribution guidelines
- `SECURITY.md` — vulnerability disclosure process

## Development

To run the test suite:

```bash
pytest
```

## Project status

LeanPass is actively maintained and used for educational purposes. The core API is stable, and new features are added primarily for learning and teaching value. Bug fixes and documentation improvements are welcome.

## Community and contribution

- See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
- See [SECURITY.md](SECURITY.md) for the vulnerability disclosure process.
- Bug reports and feature requests can be opened through the issue templates in [.github/ISSUE_TEMPLATE](.github/ISSUE_TEMPLATE).
- Pull requests should follow the template in [.github/PULL_REQUEST_TEMPLATE/pull_request_template.md](.github/PULL_REQUEST_TEMPLATE/pull_request_template.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Star History

<a href="https://www.star-history.com/?repos=Terminay%2Fleanpass&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Terminay/leanpass&type=date&theme=dark&legend=top-left&sealed_token=fJJr1fCgBwIlSkOw9Bp_9bc9MkssvAAEXVRl4UOSqEW5yT80eFu6udpyDwky0B7yFSTDkhePBqlxOWrvImDSxj6GfEa1klFvjAIRvXc9KM3SGBoVv__Sbw" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Terminay/leanpass&type=date&legend=top-left&sealed_token=fJJr1fCgBwIlSkOw9Bp_9bc9MkssvAAEXVRl4UOSqEW5yT80eFu6udpyDwky0B7yFSTDkhePBqlxOWrvImDSxj6GfEa1klFvjAIRvXc9KM3SGBoVv__Sbw" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Terminay/leanpass&type=date&legend=top-left&sealed_token=fJJr1fCgBwIlSkOw9Bp_9bc9MkssvAAEXVRl4UOSqEW5yT80eFu6udpyDwky0B7yFSTDkhePBqlxOWrvImDSxj6GfEa1klFvjAIRvXc9KM3SGBoVv__Sbw" />
 </picture>
</a>
