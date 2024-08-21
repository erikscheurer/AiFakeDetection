import torch.nn as nn
import torch
import numpy as np


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coeff * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.iter_num = 0
        self.alpha = alpha
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        coeff = (
            2.0
            * (self.high - self.low)
            / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
            - (self.high - self.low)
            + self.low
        )
        return GradientReverse.apply(x, coeff)


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 8)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.gradient_reverse = GradientReversalLayer()

    def forward(self, x):
        x = self.gradient_reverse(x)
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        return x


# test gradient reversal
if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    gradient_reversal = GradientReversalLayer()
    gradient_reversal.iter_num = 100000
    y = gradient_reversal(x) * 2
    z = y.sum()
    y.retain_grad()
    z.retain_grad()
    z.backward()
    print(x.grad)
    print(y.grad)
    print(z.grad)
    print(y)
    print(z)
    print(x)
