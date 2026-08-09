import sys
sys.path.insert(0, '.')
from leanpass import Tensor, nn, optim
import numpy as np

model = nn.MLP([2,4,1])
opt = optim.SGD(model.parameters(), lr=0.1)

print('model __dict__ keys and types:')
for k, v in model.__dict__.items():
    print(k, type(v))

params = model.parameters()
print('model.parameters() ->', params)
print('len=', len(params))

x = Tensor([[1.0, 1.0]], requires_grad=False)
y_true = Tensor([[2.0]], requires_grad=False)

pred = model(x)
loss = nn.mse_loss(pred, y_true)

print('param data before:')
for p in model.parameters():
    print(repr(p), p.data)

model.zero_grad()
loss.backward()
print('\nparam grads after backward:')
for p in model.parameters():
    print(repr(p), p.grad)

opt.step()
print('\nparam data after step:')
for p in model.parameters():
    print(repr(p), p.data)
