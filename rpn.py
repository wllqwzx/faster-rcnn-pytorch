import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

from utils.generate_anchor import generate_anchor
from utils.proposal_creator import ProposalCreator
from utils.anchor_target_creator import AnchorTargetCreator


class rpn(nn.Module):
    def __init__(self, in_channel, mid_channel, ratio=[0.5, 1, 2], anchor_size = [128, 256, 512]):
        super(rpn, self).__init__()

        self.K = len(ratio)*len(anchor_size)    # default: 9 : 9 ahcnors per spatial channel in feature maps

        self.mid_layer = nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1) 
        self.prob_layer = nn.Conv2d(mid_channel, 2*self.K, kernel_size=1, stride=1, padding=0)
        self.delta_layer = nn.Conv2d(mid_channel, 4*self.K, kernel_size=1, stride=1, padding=0)

        self.proposal_creator = ProposalCreator()
        self.anchor_target_creator = AnchorTargetCreator()

    def forward(self, features, image_size):
        """
        Batch size are fixed to one.
        features: (N-1, C, H, W)
        """
        #---------- debug
        assert isinstance(features, Variable)
        assert features.shape[0] == 1
        #---------- debug

        _, _, feature_height, feature_width = features.shape
        image_height, image_width = image_size[0], image_size[1]

        mid_features = F.relu(self.mid_layer(features))
        
        delta = self.delta_layer(mid_features)
        delta = delta.permute(0,2,3,1).view([feature_height*feature_width*self.K, 4])
        
        prob = F.softmax(self.prob_layer(mid_features))
        prob = prob.permute(0,2,3,1).view([feature_height*feature_width*self.K, 2])

        # ndarray: (feature_height*feature_width*K, 4)
        anchor = generate_anchor(feature_height, feature_width, image_size, self.ratio, self.anchor_size)
        #---------- debug
        assert isinstance(delta, Variable) and isinstance(prob, Variable) and isinstance(anchor, np.ndarray)
        assert delta.shape == (feature_height*feature_width*self.K, 4)
        assert prob.shape == (feature_height*feature_width*self.K, 2)
        #---------- debug
        return delta, prob, anchor

    def loss(self, delta, prob, anchor, gt_bbox, image_size):
        #---------- debug
        
        #---------- debug
        target_delta, anchor_label = self.anchor_target_creator.make_anchor_target(anchor, gt_bbox, image_size)




    def predict(self):
        pass

