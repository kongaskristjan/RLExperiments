
import Definitions
import torch
from torch import nn
from torch.nn import functional as F

class PolicyLearner:
    def __init__(self, net, confidenceMul=1):
        self.net = net
        nn.CrossEntropyLoss()
        self.criterion = TargetedMSELoss()
        self.optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
        self.confidenceMul = confidenceMul

    def setConfidenceMul(self, confidenceMul):
        self.confidenceMul = confidenceMul

    def forward(self, inputs):
        inputs = inputs.to(Definitions.device)
        outputs = self.net.forward(inputs)
        outputs = outputs.to("cpu")
        outputs *= self.confidenceMul
        probs = F.softmax(outputs)
        distr = torch.distributions.Categorical(probs=probs)
        samples = distr.sample()
        return samples

    def learn(self, inputs, labels, rewards):
        self.optimizer.zero_grad()
        inputs = inputs.to(Definitions.device)
        outputs = self.net(inputs)
        outputs = outputs.to("cpu")
        loss = self.criterion(outputs, labels, rewards)
        loss = torch.sum(loss)
        loss.backward()
        self.optimizer.step()
        return loss


class TargetedMSELoss:
    def __call__(self, outputs, labels, rewards):
        targetMask = torch.zeros(*outputs.shape, dtype=torch.float32)
        target = torch.zeros(*outputs.shape, dtype=torch.float32)
        for i in range(len(labels)):
            targetMask[i][labels[i]] = 1.0
            target[i][labels[i]] = rewards[i]
        loss = (outputs - target) ** 2
        loss = targetMask * loss
        loss = torch.sum(loss, dim=1)
        return loss
