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

Leanpass is a lightweight, transparent NumPy-based autodiff library for small neural network experiments. It is designed to be easy to read, simple to inspect, and practical for learning how automatic differentiation works under the hood.

_Leanpass is indexed in the [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning) repository, under the **python** section._

![demo](leanpass.gif)

I am incredibly grateful to the early contributors helping me shape this project! Click on their profiles below to **follow them** and check out their work:

<a href="https://github.com/Terminay/leanpass/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Terminay/leanpass" />
</a>

<!-- CONTRIBUTORS-TABLE:START -->

| Contributor | Follow | First PR |
| ----------- | ------ | -------- |
| [eeshsaxena](https://github.com/eeshsaxena) | [Follow](https://github.com/eeshsaxena) | [#18](https://github.com/Terminay/leanpass/pull/18) |
| [VedantMadane](https://github.com/VedantMadane) | [Follow](https://github.com/VedantMadane) | [#40](https://github.com/Terminay/leanpass/pull/40) |
| [Chirudeva-Reddy](https://github.com/Chirudeva-Reddy) | [Follow](https://github.com/Chirudeva-Reddy) | [#56](https://github.com/Terminay/leanpass/pull/56) |

<!-- CONTRIBUTORS-TABLE:END -->

Made with [contrib.rocks](https://contrib.rocks).

*Want to help out?* I am actively looking for fresh eyes to test features, fix bugs, or improve docs. Check out the open [Issues](https://github.com/Terminay/leanpass) or just open a PR.
> [!NOTE]
> If your profile picture isn't showing up yet, Contrib.Rocks can take a few days to update due to high traffic.

## Table of contents

- [Why LeanPass?](#why-leanpass)
- [Features](#features)
- [Installation](#installation)
- [Documentation](#documentation)
- [Quick start](#quick-start)
- [Examples](#examples)
- [Development](#development)
- [Project status](#project-status)
- [Community and contribution](#community-and-contribution)
- [License](#license)
- [Star History](#star-history)
- [RepoVibes](#repovibes)

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

Full documentation is available at [**leanpass.kilobyte136.workers.dev**](https://leanpass.kilobyte136.workers.dev/) and includes API reference, core concept explanations, and hands-on guides covering regression, classification, MNIST, gradient checking, and custom layers. See the [changelog](docs/changelog.md) for the complete release history.

### DeepWiki

To delve more into the development of leanpass, you could check out the *devin indexed* deepwiki docs.
The deepwiki docs update *1 day later* of every release or update made to leanpass, note that.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Terminay/leanpass)

## Quick start

When leanpass is installed, build a small neural network and run a forward pass:

```python
from leanpass import Tensor, nn

x = Tensor([[1.0, 2.0]], requires_grad=False)
model = nn.MLP([2, 16, 3])
logits = model(x)
print(logits)
```

## Examples

Explore runnable Python examples in [`examples/`](examples/):

| Example | File | Description |
|---------|------|-------------|
| Classification | `examples/demo.py` | 3-class synthetic data with MLP |
| XOR | `examples/demo_xor.py` | Classic XOR problem with 2-layer network |
| MNIST-lite | `examples/demo_mnist_lite.py` | Digit classification on sklearn digits (8×8 images) |
| Regression | `examples/demo_regression.py` | Sine wave fitting with MLP |

Run any example with:

```bash
python -m examples.demo                 # 3-class classification
python -m examples.demo_xor             # XOR classification
python -m examples.demo_mnist_lite      # MNIST-lite
python -m examples.demo_regression      # Regression
```

## Development

Run the test suite from the repository root:

```bash
pytest
```

## Project status

LeanPass is an actively maintained NumPy autodiff project for education and small-scale neural network experiments. The core API is stable; bug reports and documentation improvements are welcome.

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

## RepoVibes

![RepoVibes](https://repovibes.vercel.app/api/hexagon/Terminay/leanpass.svg)
