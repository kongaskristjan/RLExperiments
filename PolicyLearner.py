
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
        #self.optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = torch.optim.Adam(net.parameters())

    def forward(self, x):
        x = self.net.forward(x)
        x = F.softmax(x)
        distr = torch.distributions.Categorical(probs=x)
        x = distr.sample()
        return x

    def learn(self, inputs, labels, correctness):
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = correctness * self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss
