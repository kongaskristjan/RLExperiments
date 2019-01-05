
import torch
from torch import nn

class PolicyLearner:
    def __init__(self, net):
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)

    def forward(self, x):
        x = self.net.forward(x)
        distr = torch.distributions.Categorical(probs=x)
        samples = distr.sample()
        return samples

    def learn(self, inputs, labels, rewards):
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = rewards * self.criterion(outputs, labels)
        loss = torch.sum(loss)
        loss.backward()
        self.optimizer.step()
        return loss
