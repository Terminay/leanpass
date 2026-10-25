---
sidebar_position: 7
---

# Inspecting Gradients

After calling `backward()`, inspect a model's parameter gradients to understand what the loss is sending through the network or to diagnose a training problem.

```python
from leanpass import Tensor, nn

model = nn.MLP([2, 4, 1])
x = Tensor([[1.0, 2.0]], requires_grad=False)
target = Tensor([[1.0]], requires_grad=False)

prediction = model(x)
loss = nn.mse_loss(prediction, target)
loss.backward()

for index, parameter in enumerate(model.parameters()):
    if parameter.grad is None:
        print(f"Parameter {index}: no gradient")
        continue

    print(
        f"Parameter {index}: shape={parameter.data.shape}, "
        f"grad_mean={parameter.grad.mean():.6f}, "
        f"grad_max={abs(parameter.grad).max():.6f}"
    )
```

Gradients are stored as NumPy arrays in each parameter's `.grad` attribute. A gradient of `None` means no gradient reached that parameter during this backward pass. A gradient array full of zeros means the parameter participated but received zero-valued gradients.

Inspect gradients after `loss.backward()` and before the optimizer updates the parameters. Clear gradients before the next training pass with `optimizer.zero_grad()`; gradients accumulate across backward calls until cleared.

Gradient values can vary with the input, model initialization, and loss. Look for unexpected missing or non-finite values rather than expecting particular numbers.
