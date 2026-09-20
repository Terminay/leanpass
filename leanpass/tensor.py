import numpy as np


def _ensure_array(data):
    """Convert inputs to NumPy arrays so math is always numeric."""
    if isinstance(data, Tensor):
        return data.data
    return np.array(data, dtype=np.float64)


def _sum_to_shape(grad, shape):
    """Reduce broadcasted gradients back to the shape of the original tensor."""
    if grad.shape == shape:
        return grad

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


class Tensor:
    """A tiny automatic differentiation tensor backed by NumPy."""

    def __init__(self, data, requires_grad=False, name=None):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = ()
        self._op = ""
        self._meta = {}
        self.name = name

    def __repr__(self):
        name = f" name={self.name}" if self.name else ""
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad}{name})"

    def _create_child(self, data, op, prev, name=None, meta=None):
        out = Tensor(data, requires_grad=any(node.requires_grad for node in prev), name=name)
        out._prev = tuple(prev)
        out._op = op
        out._meta = meta or {}
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._create_child(self.data + other.data, "+", (self, other))

        def _backward():
            if self.requires_grad:
                self.grad += _sum_to_shape(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _sum_to_shape(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = self._create_child(-self.data, "neg", (self,))

        def _backward():
            if self.requires_grad:
                self.grad += _sum_to_shape(-out.grad, self.data.shape)

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._create_child(self.data - other.data, "-", (self, other))

        def _backward():
            if self.requires_grad:
                self.grad += _sum_to_shape(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _sum_to_shape(-out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._create_child(self.data * other.data, "*", (self, other))

        def _backward():
            if self.requires_grad:
                self.grad += _sum_to_shape(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _sum_to_shape(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if np.any(other.data == 0):
            raise ZeroDivisionError("Tensor division by zero")
        out = self._create_child(self.data / other.data, "/", (self, other))

        def _backward():
            if self.requires_grad:
                self.grad += _sum_to_shape(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _sum_to_shape(-out.grad * self.data / (other.data ** 2), other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        exponent = float(exponent)
        out = self._create_child(self.data ** exponent, "**", (self,), meta={"exponent": exponent})

        def _backward():
            if self.requires_grad:
                self.grad += _sum_to_shape(out.grad * exponent * self.data ** (exponent - 1), self.data.shape)

        out._backward = _backward
        return out

    def exp(self, a_max=700.0):
        """Compute element-wise exponential with numerical stability guard against overflow."""
        clipped = np.clip(self.data, None, a_max)
        out = self._create_child(np.exp(clipped), "exp", (self,), meta={"a_max": a_max})

        def _backward():
            if self.requires_grad:
                self.grad += _sum_to_shape(out.grad * out.data, self.data.shape)

        out._backward = _backward
        return out

    def log(self, eps=1e-15):
        """Compute element-wise natural logarithm with numerical stability guard for non-positive inputs."""
        clipped = np.clip(self.data, eps, None)
        out = self._create_child(np.log(clipped), "log", (self,), meta={"eps": eps})

        def _backward():
            if self.requires_grad:
                self.grad += _sum_to_shape(out.grad / clipped, self.data.shape)

        out._backward = _backward
        return out

    def sigmoid(self):
        """Compute element-wise sigmoid with numerical stability avoiding overflow for negative inputs."""
        pos_mask = self.data >= 0
        neg_mask = ~pos_mask
        out_data = np.empty_like(self.data, dtype=np.float64)
        out_data[pos_mask] = 1.0 / (1.0 + np.exp(-self.data[pos_mask]))
        exp_neg = np.exp(self.data[neg_mask])
        out_data[neg_mask] = exp_neg / (1.0 + exp_neg)
        out = self._create_child(out_data, "sigmoid", (self,))

        def _backward():
            if self.requires_grad:
                sigmoid_grad = out.data * (1.0 - out.data)
                self.grad += _sum_to_shape(out.grad * sigmoid_grad, self.data.shape)

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        """Compute softmax with numerical stability via max-subtraction and exponent clipping."""
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        shifted = np.clip(shifted, -700.0, 0.0)
        exp_values = np.exp(shifted)
        sum_exp = exp_values.sum(axis=axis, keepdims=True)
        probabilities = exp_values / np.maximum(sum_exp, 1e-15)
        out = self._create_child(probabilities, "softmax", (self,), meta={"axis": axis})

        def _backward():
            if self.requires_grad:
                grad = out.grad
                sum_grad = (grad * out.data).sum(axis=axis, keepdims=True)
                self.grad += _sum_to_shape(out.data * (grad - sum_grad), self.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._create_child(self.data @ other.data, "@", (self, other))

        def _backward():
            if self.requires_grad:
                self.grad += _sum_to_shape(out.grad @ other.data.T, self.data.shape)
            if other.requires_grad:
                other.grad += _sum_to_shape(self.data.T @ out.grad, other.data.shape)

        out._backward = _backward
        return out

    def relu(self):
        out = self._create_child(np.maximum(0, self.data), "relu", {self})

        def _backward():
            if self.requires_grad:
                grad_input = out.grad * (self.data > 0).astype(np.float64)
                self.grad += _sum_to_shape(grad_input, self.data.shape)

        out._backward = _backward
        return out

    def tanh(self):
        out_data = np.tanh(self.data)
        out = self._create_child(out_data, "tanh", {self})

        def _backward():
            if self.requires_grad:
                grad_input = out.grad * (1 - out.data ** 2)
                self.grad += _sum_to_shape(grad_input, self.data.shape)

        out._backward = _backward
        return out

    def leaky_relu(self, negative_slope=0.01):
        out_data = np.where(self.data >= 0, self.data, self.data * negative_slope)
        out = self._create_child(out_data, "leaky_relu", {self}, meta={"negative_slope": negative_slope})

        def _backward():
            if self.requires_grad:
                grad_input = np.where(self.data >= 0, out.grad, out.grad * negative_slope)
                self.grad += _sum_to_shape(grad_input, self.data.shape)

        out._backward = _backward
        return out

    def gelu(self):
        out_data = 0.5 * self.data * (1 + np.tanh(np.sqrt(2 / np.pi) * (self.data + 0.044715 * self.data ** 3)))
        out = self._create_child(out_data, "gelu", {self})

        def _backward():
            if self.requires_grad:
                tanh_arg = np.sqrt(2 / np.pi) * (self.data + 0.044715 * self.data ** 3)
                tanh_val = np.tanh(tanh_arg)
                grad_term = 0.5 * (1 + tanh_val) + (0.5 * self.data * (1 - tanh_val ** 2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * self.data ** 2))
                self.grad += _sum_to_shape(out.grad * grad_term, self.data.shape)

        out._backward = _backward
        return out

    def clip(self, a_min=None, a_max=None):
        """Clamp tensor values to the interval [a_min, a_max].

        Both bounds are optional. The backward pass only propagates gradient
        for elements that were not clipped (standard straight-through behavior).
        """
        out_data = np.clip(self.data, a_min, a_max)
        out = self._create_child(out_data, "clip", (self,), meta={"min": a_min, "max": a_max})

        def _backward():
            if self.requires_grad:
                mask = np.ones_like(self.data, dtype=np.float64)
                if a_min is not None:
                    mask = mask * (self.data > a_min)
                if a_max is not None:
                    mask = mask * (self.data < a_max)
                self.grad += _sum_to_shape(out.grad * mask, self.data.shape)

        out._backward = _backward
        return out

    # alias common name
    def clamp(self, min=None, max=None):
        return self.clip(min, max)

    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, name="sum")
        out._prev = (self,)
        out._op = "sum"
        out._meta = {"axis": axis, "keepdims": keepdims, "shape": self.data.shape}

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None:
                    grad = np.expand_dims(grad, axis=axis) if not keepdims else grad
                    self.grad += np.broadcast_to(grad, self.data.shape)
                else:
                    self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def __getitem__(self, idx):
        """Basic indexing / slicing returning a new Tensor view (not a view in-place).

        The returned tensor participates in autodiff; gradients are placed back
        into the source tensor at the same indices during the backward pass.
        """
        out_data = self.data[idx]
        out = self._create_child(out_data, "getitem", (self,), meta={"index": idx})

        def _backward():
            if self.requires_grad:
                if out.grad is None:
                    return
                grad_buf = np.zeros_like(self.data)
                grad_buf[idx] = out.grad
                self.grad += _sum_to_shape(grad_buf, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out_data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, name="mean")
        out._prev = (self,)
        out._op = "mean"
        out._meta = {"axis": axis, "keepdims": keepdims, "shape": self.data.shape}

        def _backward():
            if self.requires_grad:
                grad = out.grad
                divisor = self.data.size if axis is None else np.prod([self.data.shape[i] for i in axis]) if isinstance(axis, tuple) else self.data.shape[axis]
                if axis is not None:
                    grad = np.expand_dims(grad, axis=axis) if not keepdims else grad
                    self.grad += np.broadcast_to(grad * (1.0 / divisor), self.data.shape)
                else:
                    self.grad += np.ones_like(self.data) * out.grad * (1.0 / divisor)

        out._backward = _backward
        return out

    def visualize(self):
        """Return a simple text representation of the computation graph."""
        nodes = []
        edges = []
        visited = set()

        def build(v):
            if v in visited:
                return
            visited.add(v)
            label = v._op or "leaf"
            if v.name:
                label += f" ({v.name})"
            nodes.append((id(v), label, v.data.shape))
            for child in v._prev:
                edges.append((id(child), id(v)))
                build(child)

        build(self)

        lines = [f"Node {nid}: {label} shape={shape}" for nid, label, shape in nodes]
        lines += [f"Edge {src} -> {dst}" for src, dst in edges]
        return "\n".join(lines)

    def visualize_dot(self):
        """Return a Graphviz DOT representation of the computation graph."""
        nodes, edges = self._graph_nodes()
        lines = ["digraph computation_graph {", "  rankdir=LR;", "  node [shape=box, style=filled, fillcolor=lightgray];"]

        for node in nodes:
            label = node._op or "leaf"
            if node.name:
                label += f"\n{node.name}"
            shape = "ellipse" if len(node._prev) == 0 else "box"
            lines.append(f"  n{ id(node) } [label=\"{label}\", shape={shape}];")

        for src, dst in edges:
            lines.append(f"  n{src} -> n{dst};")

        lines.append("}")
        return "\n".join(lines)

    def backward(self, gradient=None):
        if gradient is None:
            gradient = np.ones_like(self.data)
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad = self.grad + np.array(gradient, dtype=np.float64)

        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        for node in topo:
            if node.requires_grad and node.grad is None:
                node.grad = np.zeros_like(node.data)

        for node in reversed(topo):
            node._backward()

    def _graph_nodes(self):
        nodes = []
        edges = []
        visited = set()

        def build(v):
            if v in visited:
                return
            visited.add(v)
            nodes.append(v)
            for child in v._prev:
                edges.append((id(child), id(v)))
                build(child)

        build(self)
        return nodes, edges

    def grad_check(self, eps=1e-6, tol=1e-4):
        """Compare backward gradients against finite differences."""
        if not self.requires_grad:
            raise ValueError("grad_check requires the output tensor to require gradients")

        nodes, _ = self._graph_nodes()
        numeric = {}

        for node in nodes:
            if not node.requires_grad:
                continue
            original = node.data.copy()
            numeric_grad = np.zeros_like(node.data)
            for idx in np.ndindex(node.data.shape):
                node.data[idx] = original[idx] + eps
                plus = self._eval_forward()
                node.data[idx] = original[idx] - eps
                minus = self._eval_forward()
                node.data[idx] = original[idx]
                numeric_grad[idx] = (np.sum(plus) - np.sum(minus)) / (2 * eps)

            self.zero_grad_all()
            self.backward()
            numeric[node] = numeric_grad
            node.data = original

        errors = []
        for node in nodes:
            if not node.requires_grad:
                continue
            diff = np.max(np.abs(node.grad - numeric[node]))
            if diff > tol:
                errors.append((node, diff, node.grad, numeric[node]))

        return errors

    def _eval_forward(self):
        """Evaluate the graph forward using current leaf values without mutating nodes."""
        topo = []
        visited = set()
        values = {}

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        for node in topo:
            if len(node._prev) == 0:
                values[node] = node.data
                continue

            if node._op == "+":
                values[node] = values[node._prev[0]] + values[node._prev[1]]
            elif node._op == "-":
                values[node] = values[node._prev[0]] - values[node._prev[1]]
            elif node._op == "*":
                values[node] = values[node._prev[0]] * values[node._prev[1]]
            elif node._op == "/":
                values[node] = values[node._prev[0]] / values[node._prev[1]]
            elif node._op == "**":
                exponent = node._meta.get("exponent", 2.0)
                values[node] = values[node._prev[0]] ** exponent
            elif node._op == "@":
                values[node] = values[node._prev[0]] @ values[node._prev[1]]
            elif node._op == "relu":
                values[node] = np.maximum(0, values[node._prev[0]])
            elif node._op == "exp":
                a_max = node._meta.get("a_max", 700.0)
                values[node] = np.exp(np.clip(values[node._prev[0]], None, a_max))
            elif node._op == "log":
                eps = node._meta.get("eps", 1e-15)
                values[node] = np.log(np.clip(values[node._prev[0]], eps, None))
            elif node._op == "sigmoid":
                x = values[node._prev[0]]
                pos = x >= 0
                neg = ~pos
                res = np.empty_like(x, dtype=np.float64)
                res[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
                exp_neg = np.exp(x[neg])
                res[neg] = exp_neg / (1.0 + exp_neg)
                values[node] = res
            elif node._op == "softmax":
                x = values[node._prev[0]]
                axis = node._meta.get("axis", -1)
                shifted = np.clip(x - x.max(axis=axis, keepdims=True), -700.0, 0.0)
                exp_values = np.exp(shifted)
                sum_exp = exp_values.sum(axis=axis, keepdims=True)
                values[node] = exp_values / np.maximum(sum_exp, 1e-15)
            elif node._op == "sum":
                values[node] = values[node._prev[0]].sum()
            elif node._op == "mean":
                values[node] = values[node._prev[0]].mean()
            elif node._op == "neg":
                values[node] = -values[node._prev[0]]
            else:
                values[node] = node.data

        return values[self]

    def zero_grad_all(self):
        """Zero all gradients in the current graph."""
        nodes, _ = self._graph_nodes()
        for node in nodes:
            if node.requires_grad:
                node.grad = np.zeros_like(node.data)

    def _graph_nodes(self):
        nodes = []
        edges = []
        visited = set()

        def build(v):
            if v in visited:
                return
            visited.add(v)
            nodes.append(v)
            for child in v._prev:
                edges.append((id(child), id(v)))
                build(child)

        build(self)
        return nodes, edges

    def grad_check(self, eps=1e-6, tol=1e-4):
        """Compare analytical gradients to numerical finite differences."""
        if not self.requires_grad:
            raise ValueError("grad_check requires the output tensor to require gradients")

        nodes, _ = self._graph_nodes()
        numeric = {}

        def scalar_forward():
            return float(self.data.sum()) if self.data.size == 1 else None

        for node in nodes:
            if not node.requires_grad:
                continue
            analytic = np.zeros_like(node.data)
            numeric_grad = np.zeros_like(node.data)
            original = node.data.copy()

            it = np.nditer(node.data, flags=["multi_index"], op_flags=["readwrite"])
            while not it.finished:
                idx = it.multi_index
                node.data[idx] = original[idx] + eps
                plus = self._eval_forward()
                node.data[idx] = original[idx] - eps
                minus = self._eval_forward()
                node.data[idx] = original[idx]
                numeric_grad[idx] = (plus - minus) / (2 * eps)
                it.iternext()

            node.zero_grad_all()
            self.backward()
            analytic = node.grad.copy()
            numeric[node] = numeric_grad
            node.data = original

        errors = []
        for node in nodes:
            if not node.requires_grad:
                continue
            diff = np.max(np.abs(node.grad - numeric[node]))
            if diff > tol:
                errors.append((node, diff, node.grad, numeric[node]))

        return errors

    def _eval_forward(self):
        """Evaluate current graph data forward to this tensor using NumPy semantics."""
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        for node in topo:
            if node._op == "leaf":
                continue
            if node._op == "+":
                node.data = list(node._prev)[0].data + list(node._prev)[1].data
            elif node._op == "-":
                node.data = list(node._prev)[0].data - list(node._prev)[1].data
            elif node._op == "*":
                node.data = list(node._prev)[0].data * list(node._prev)[1].data
            elif node._op == "/":
                node.data = list(node._prev)[0].data / list(node._prev)[1].data
            elif node._op == "**":
                node.data = list(node._prev)[0].data ** float(2)
            elif node._op == "@":
                node.data = list(node._prev)[0].data @ list(node._prev)[1].data
            elif node._op == "relu":
                node.data = np.maximum(0, list(node._prev)[0].data)
            elif node._op == "exp":
                a_max = node._meta.get("a_max", 700.0)
                node.data = np.exp(np.clip(list(node._prev)[0].data, None, a_max))
            elif node._op == "log":
                eps = node._meta.get("eps", 1e-15)
                node.data = np.log(np.clip(list(node._prev)[0].data, eps, None))
            elif node._op == "sigmoid":
                x = list(node._prev)[0].data
                pos = x >= 0
                neg = ~pos
                res = np.empty_like(x, dtype=np.float64)
                res[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
                exp_neg = np.exp(x[neg])
                res[neg] = exp_neg / (1.0 + exp_neg)
                node.data = res
            elif node._op == "softmax":
                x = list(node._prev)[0].data
                axis = node._meta.get("axis", -1)
                shifted = np.clip(x - x.max(axis=axis, keepdims=True), -700.0, 0.0)
                exp_values = np.exp(shifted)
                sum_exp = exp_values.sum(axis=axis, keepdims=True)
                node.data = exp_values / np.maximum(sum_exp, 1e-15)
            elif node._op == "sum":
                node.data = list(node._prev)[0].data.sum()
            elif node._op == "mean":
                node.data = list(node._prev)[0].data.mean()
        return float(self.data) if np.isscalar(self.data) else self.data

    def zero_grad_all(self):
        """Zero all gradients in the current graph."""
        nodes, _ = self._graph_nodes()
        for node in nodes:
            if node.requires_grad:
                node.grad = np.zeros_like(node.data)
