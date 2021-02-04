import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)

x_train = [[1], [2], [3]]
y_train = [[1], [2], [3]]
X = Variable(torch.Tensor(x_train))
Y = Variable(torch.Tensor(y_train))

#Hypothesis = XW + b
model = nn.Linear(1, 1, bias=True)
#cost criterion
criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(2001):
    optimizer.zero_grad()
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step % 20 == 0:
        print(step, cost.data.numpy(), model.weight.data.numpy(), model.bias.data.numpy())


predicted = model(Variable(torch.Tensor([[5]])))
print(predicted.data.numpy())
predicted = model(Variable(torch.Tensor([[2.5]])))
print(predicted.data.numpy())
predicted = model(Variable(torch.Tensor([[1.5], [3.5]])))
print(predicted.data.numpy())