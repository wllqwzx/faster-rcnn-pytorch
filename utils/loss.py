import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def _smooth_l1_loss(pred_delta, target_delta, weight, sigma):
    #---------- debug
    assert isinstance(pred_delta, Variable)
    assert isinstance(target_delta, Variable)
    assert isinstance(weight, Variable)
    #---------- debug
    sigma2 = sigma * sigma
    diff = weight * (pred_delta - target_delta)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1.0 / sigma2)).float() # do not back propagat on flag
    flag = Variable(flag, requires_grad=False)
    res = flag*(sigma2 / 2.)*(abs_diff * abs_diff)  +  (1 - flag)*(abs_diff - 0.5 / sigma2)
    return res.sum()


def delta_loss(pred_delta, target_delta, anchor_label, sigma):
    """
    Args:
        pred_delta: (N,4)
        target_delta: (N,4)
        anchor_label: (N,)
    """
    #---------- debug
    assert isinstance(pred_delta, Variable)
    assert isinstance(target_delta, Variable)
    assert isinstance(anchor_label, Variable)
    assert pred_delta.shape == target_delta.shape
    assert pred_delta.shape[0] == anchor_label.shape[0]
    #---------- debug
    weight = torch.zeros(target_delta.shape)
    if torch.cuda.is_available():
        weight.cuda()

    pos_index = (anchor_label.data > 0).view(-1,1).expand_as(weight)
    weight[pos_index] = 1
    weight = Variable(weight)

    loss = _smooth_l1_loss(pred_delta, target_delta, weight, sigma)
    return loss

if __name__ == '__main__':
    pred_delta = Variable(torch.rand(15000, 4))
    target_delta = Variable(torch.rand(15000, 4))
    anchor_label = Variable((torch.rand(15000)*20).int())
    loss = delta_loss(pred_delta, target_delta, anchor_label, 1)
    print(loss) #=> Variable: FloatTensor od size 1.
    print("loss passed!")
