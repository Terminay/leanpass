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
Returns a new Tensor with the sum of the elements, reduced over all elements or over the given axis.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `axis` | int, tuple of int, or None | `None` | Axis or axes to reduce. Negative axes are allowed. `None` reduces over all elements and returns a scalar (0-d) Tensor |
| `keepdims` | bool | `False` | If `True`, the reduced axes are kept with size 1 so the result broadcasts against the input |

For an input of shape `(2, 3)`: `.sum()` has shape `()`, `.sum(axis=0)` has shape `(3,)`, `.sum(axis=1, keepdims=True)` has shape `(2, 1)`.

### `mean(axis=None, keepdims=False)`
Returns a new Tensor with the mean of the elements, reduced over all elements or over the given axis.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `axis` | int, tuple of int, or None | `None` | Axis or axes to reduce. Negative axes are allowed. `None` reduces over all elements and returns a scalar (0-d) Tensor |
| `keepdims` | bool | `False` | If `True`, the reduced axes are kept with size 1 so the result broadcasts against the input |

For an input of shape `(2, 3)`: `.mean()` has shape `()`, `.mean(axis=0)` has shape `(3,)`, `.mean(axis=1, keepdims=True)` has shape `(2, 1)`.

### `matmul(other)`
Matrix multiplication equivalent to `self @ other`.
Returns a new Tensor.

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
