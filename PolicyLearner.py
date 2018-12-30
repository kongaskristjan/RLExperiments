
import torch
from torch import nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        x = self.linear(x)
        return x


class PolicyLearner:
    def __init__(self, net):
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)

    def forward(self, x):
        x = self.net.forward(x)
        x = F.softmax(x)
        distr = torch.distributions.Categorical(probs=x)
        samples = distr.sample()
        return samples

    def learn(self, inputs, labels, reward):
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = reward * self.criterion(outputs, labels)
        loss = torch.sum(loss)
        loss.backward()
        self.optimizer.step()
        return loss
