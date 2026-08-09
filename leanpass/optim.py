import numpy as np


class SGD:
    """Stochastic gradient descent with optional momentum and L2 weight decay.

    Args:
        params: iterable of `Tensor` parameters
        lr: learning rate
        momentum: momentum factor (0.0 means no momentum)
        weight_decay: L2 penalty factor applied to parameters (aka weight decay)
    """

    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        # velocity buffers for momentum (kept even if momentum==0 for simplicity)
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        """Perform a parameter update step."""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            # apply L2 weight decay directly to the gradient (common choice)
            g = param.grad
            if self.weight_decay:
                g = g + self.weight_decay * param.data

            if self.momentum:
                self.v[i] = self.momentum * self.v[i] + g
                update = self.v[i]
            else:
                update = g

            param.data = param.data - self.lr * update

    def zero_grad(self):
        """Zero out gradients so the next backward pass starts clean."""
        for param in self.params:
            param.grad = np.zeros_like(param.data)


class Adam:
    """Adam optimizer with bias-corrected moment estimates and optional L2 weight decay."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = float(weight_decay)
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        """Update each parameter using Adam's adaptive moment estimates."""
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            # apply L2 weight decay to the gradient
            g = param.grad
            if self.weight_decay:
                g = g + self.weight_decay * param.data

            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g ** 2)

            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Zero out gradients for all tracked parameters."""
        for param in self.params:
            param.grad = np.zeros_like(param.data)
