---
sidebar_position: 1
---

# Tensor API Reference

Full API specification for `leanpass.Tensor`.

## Constructor

```python
Tensor(data, requires_grad=False, name=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | array-like | — | Input data (list, tuple, NumPy array, or Tensor) |
| `requires_grad` | bool | `False` | Whether to track gradients for this tensor |
| `name` | str or None | `None` | Optional label for debugging and graph visualization |

**Raises:** `ValueError` if data cannot be converted to a float64 NumPy array.

## Attributes

### `data`
The underlying NumPy array (`np.float64`). Read/write.

### `grad`
Gradient array of the same shape as `data`. `None` if `requires_grad=False`.
Initialized to zeros when `requires_grad=True`.

### `requires_grad`
Whether this tensor participates in gradient computation. Read-only after creation.

### `shape`
Tuple — the shape of the underlying data array.

### `name`
Optional string label.

## Methods

### `backward(gradient=None)`
Computes gradients by reverse-mode autodiff.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gradient` | array-like or None | `None` | Initial gradient (defaults to ones with shape `self.data.shape`) |

Topologically sorts the graph, seeds the gradient, then walks backward calling each node's `_backward` function.

### `zero_grad_all()`
Resets `.grad` to zeros for all tensors in the computation graph.

### `relu()`
Returns a new Tensor: `max(0, x)` element-wise. Gradient-aware.

### `sigmoid()`
Returns a new Tensor: `1 / (1 + exp(-x))` element-wise. Gradient-aware.

### `tanh()`
Returns a new Tensor: hyperbolic tangent, `tanh(x)`, element-wise. Gradient-aware (the gradient is `1 - tanh(x)^2`).

### `leaky_relu(negative_slope=0.01)`
Returns a new Tensor: `x` where `x >= 0`, otherwise `negative_slope * x`, element-wise. Gradient-aware (the gradient is `1` where `x >= 0`, otherwise `negative_slope`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `negative_slope` | float | `0.01` | Multiplier applied to negative inputs |

### `gelu()`
Returns a new Tensor with GELU applied element-wise. Uses the tanh approximation, not the exact erf-based form:
`0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`. Gradient-aware.

### `softmax(axis=-1)`
Returns a new Tensor with softmax applied along the specified axis.
Shifts inputs by max for numerical stability before computing exponentials.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `axis` | int | `-1` | Axis along which to compute softmax |

### `exp()`
Returns a new Tensor: `e^x` element-wise.

### `log()`
Returns a new Tensor: `ln(x)` element-wise (natural logarithm).
Note: `log(0)` returns `-inf`, `log(negative)` returns `nan`.

### `sum(axis=None, keepdims=False)`
Returns a new Tensor with the sum of all elements (full reduction, not per-axis).

### `mean(axis=None, keepdims=False)`
Returns a new Tensor with the mean of all elements (full reduction, not per-axis).

### `matmul(other)`
Matrix multiplication equivalent to `self @ other`.
Returns a new Tensor.

### `visualize()`
Returns a plain-text description of the computation graph ending at this tensor, as a `str`.
There is one `Node <id>: <op> shape=<shape>` line per node (leaf tensors show `leaf`, and the tensor's `name` is appended in parentheses if set), followed by one `Edge <src> -> <dst>` line per connection. The ids are Python object ids, so they differ between runs.

### `visualize_dot()`
Returns the computation graph ending at this tensor as a Graphviz DOT string (`digraph computation_graph`, laid out left to right). Leaf tensors are drawn as ellipses and operations as boxes; a tensor's `name` is added to its label. The method only builds the string: it does not write a file or need Graphviz installed. Render it yourself, for example by saving it to `graph.dot` and running `dot -Tpng graph.dot -o graph.png`.

```python
a = Tensor([1.0, 2.0], requires_grad=True, name="a")
out = (a * 2).relu()
print(out.visualize())
dot = out.visualize_dot()
```

## Operators

| Operator | Method | Description |
|----------|--------|-------------|
| `+` | `__add__`, `__radd__` | Element-wise addition |
| `-` | `__sub__` | Element-wise subtraction |
| `-` (unary) | `__neg__` | Element-wise negation |
| `*` | `__mul__`, `__rmul__` | Element-wise multiplication |
| `/` | `__truediv__` | Element-wise division |
| `**` | `__pow__` | Element-wise power (only exponent 2) |
| `@` | `__matmul__` | Matrix multiplication |

All operators support broadcasting following NumPy semantics.
