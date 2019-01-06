
import Definitions
import torch
from torch import nn

class PolicyLearner:
    def __init__(self, net):
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)

    def forward(self, inputs):
        inputs = inputs.to(Definitions.device)
        outputs = self.net.forward(inputs)
        outputs = outputs.to("cpu")
        distr = torch.distributions.Categorical(probs=outputs)
        samples = distr.sample()
        return samples

    def learn(self, inputs, labels, rewards):
        self.optimizer.zero_grad()
        inputs = inputs.to(Definitions.device)
        outputs = self.net(inputs)
        outputs = outputs.to("cpu")
        loss = rewards * self.criterion(outputs, labels)
        loss = torch.sum(loss)
        loss.backward()
        self.optimizer.step()
        return loss
