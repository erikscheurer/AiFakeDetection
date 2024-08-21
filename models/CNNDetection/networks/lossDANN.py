import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def DANN(features, ad_net, grl_layer, use_gpu=True):
    ad_out = ad_net(grl_layer(features))
    batch_size = ad_out.size(0) // 2
    dc_target = Variable(
        torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    )
    if use_gpu:
        dc_target = dc_target.cuda()
    return nn.BCELoss(ad_out.view(-1), dc_target.view(-1))
