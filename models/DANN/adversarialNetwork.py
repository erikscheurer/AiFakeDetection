import torch.nn as nn
import torch
import numpy as np

class GradientReversalLayer(torch.autograd.Function):
  def __init__(self, high_value=1.0):
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = high_value
    self.max_iter = 10000.0
    
  def forward(self, input):
    self.iter_num += 1
    output = input * 1.0
    return output

  def backward(self, gradOutput):
    self.coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha*self.iter_num / self.max_iter)) - (self.high - self.low) + self.low)
    return -self.coeff * gradOutput

class SilenceLayer(torch.autograd.Function):
  def __init__(self):
    pass
  def forward(self, input):
    return input * 1.0

  def backward(self, gradOutput):
    return 0 * gradOutput

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, 1024)
    self.ad_layer2 = nn.Linear(1024,1024)
    self.ad_layer3 = nn.Linear(1024, 1)
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
    x = self.sigmoid(x)
    return x
