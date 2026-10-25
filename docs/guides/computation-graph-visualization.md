---
sidebar_position: 8
---

# Visualize a Computation Graph

LeanPass can describe the computation graph that produces a tensor. Use `visualize()` for a plain-text listing or `visualize_dot()` to generate Graphviz DOT that can be rendered as an image.

```python
from pathlib import Path

from leanpass import Tensor, nn

x = Tensor([[1.0, 2.0]], requires_grad=False)
model = nn.MLP([2, 4, 3])
logits = model(x)
target = Tensor([[0.0, 1.0, 0.0]], requires_grad=False)
loss = nn.cross_entropy_loss(logits, target)

print(loss.visualize())
Path("computation_graph.dot").write_text(loss.visualize_dot(), encoding="utf-8")
```

Install [Graphviz](https://graphviz.org/download/) so its `dot` command is available, then render the DOT file:

```bash
dot -Tsvg computation_graph.dot -o computation_graph.svg
```

Open `computation_graph.svg` in a browser or image viewer. The visualization starts from the tensor you call `visualize_dot()` on and follows its recorded operations back to the input tensors. For a training example, visualize the loss tensor to see the graph used to produce that loss.
