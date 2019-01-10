
from torch import nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self, inputSpace, outputSpace, mul=1):
        super(Net, self).__init__()
        inputN = 1
        for i in inputSpace.shape:
            inputN *= i
        outputN = outputSpace.n
        hidden = 32
        self.linear1 = nn.Linear(inputN, hidden)
        self.linear2 = nn.Linear(hidden, outputN)
        self.mul = mul

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x *= self.mul
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
