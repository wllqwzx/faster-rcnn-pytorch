import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np


class rpn(nn.Module):
    def __init__(self, in_channel, mid_channel, ratio=[0.5, 1, 2], anchor_size = [128, 256, 512]):
        super().__init__()

        self.ratio = ratio
        self.anchor_size = anchor_size
        self.K = len(ratio)*len(anchor_size)    # default: 9 : 9 ahcnors per spatial channel in feature maps
        
        self.mid_layer = nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1) 
        self.prob_layer = nn.Conv2d(mid_channel, 2*self.K, kernel_size=1, stride=1, padding=0)
        self.delta_layer = nn.Conv2d(mid_channel, 4*self.K, kernel_size=1, stride=1, padding=0)

    def forward(self, feature_map, image_size):
        """
        TODO: batch size are fixed to one, will be generalized later!
        * **feature_map** : type `Variable`, shape `[batch=1, channel, height, width]`
        * **image_size** : type `ndarray`, shape `[2]`
        """
        if feature_map.shape[0] != 1:
            print("batch size should be 1")
            assert feature_map.shape[0] == 1
        batch, _, feature_height, feature_width = feature_map.shape
        image_height, image_width = image_size[0], image_size[1]

        mid_representation = F.relu(self.mid_layer(feature_map))    # [batch, mid_channel, feature_height, feature_width]
        
        deltas = self.delta_layer(mid_representation)   # Variable; [batch=1, 4K, feature_height, feature_width]
        deltas = deltas.permute(0,2,3,1).view([feature_height, feature_width, self.K, 4])
        
        probs = self.prob_layer(mid_representation)     # Variable; [batch=1, 2K, feature_height, feature_width]
        probs = probs.permute(0,2,3,1).view([feature_height, feature_width, self.K, 2])

        # ndarray; [feature_height, feature_width, K, 4]
        anchors = _generate_anchors(self.ratio, self.anchor_size, feature_height, feature_width, image_size)
        anchors = Variable(torch.from_numpy(anchors), requires_grad=False)   # transform anchors to Variable
        
        # anchor, roi, bbox     format: [xmin, ymin, xmax, ymax]; 
        # delta                 format: dx,dy,dh,dw
        rois = _adjust_anchors_to_rois(anchors, deltas) # type: Variable; [feature_height, feature_width, K, 4]
        
        rois = _roi_filter(rois, probs) # type: Variable; [num_left_rois, 4]
        return rois

def _generate_anchors(ratio, anchor_size, feature_height, feature_width, image_size):
    anchor_base = []
    for ratio_t in ratio:
        for anchor_size_t in anchor_size:
            h = anchor_size_t*math.sqrt(ratio_t)
            w = anchor_size_t*math.sqrt(1/ratio_t)
            anchor_base.append([-h/2, -w/2, h/2, w/2])
    anchor_base = np.array(anchor_base) # default shape: [9,4]

    K = len(ratio) * len(anchor_size)   # default: 9
    image_height = image_size[0]
    image_width = image_size[1]
    stride_x = image_height / feature_height
    stride_y = image_width / feature_width
    anchors = np.zeros([feature_height, feature_width, K, 4])
    for i in range(feature_height):
        for j in range(feature_width):
            x = i*stride_x + stride_x/2
            y = j*stride_y + stride_y/2
            shift = [x,y,x,y]
            anchors[i, j] = anchor_base+shift

    return anchors  # type: ndarray; default shape: [feature_height, feature_width, 9, 4]


def _adjust_anchors_to_rois(anchors, deltas):
    assert anchors.shape == deltas.shape
    assert isinstance(anchors, Variable)
    assert isinstance(deltas, Variable)

    feature_height, feature_width, K, _ = anchors.shape
    anchors_h = anchors[:,:,:,2::4] - anchors[:,:,:,0]  #[feature_height, feature_width, 9]
    anchors_w = anchors[:,:,:,3] - anchors[:,:,:,1]     #[feature_height, feature_width, 9]
    anchors_x = anchors[:,:,:,0] + anchors_h/2          #[feature_height, feature_width, 9]
    anchors_y = anchors[:,:,:,1] + anchors_w/2          #[feature_height, feature_width, 9]

    rois_x = anchors_x + anchors_h*deltas[:,:,:,0]  #[feature_height, feature_width, 9]
    rois_y = anchors_y + anchors_w*deltas[:,:,:,1]  #[feature_height, feature_width, 9]
    rois_h = anchors_h / torch.exp(deltas[:,:,:,2]) #[feature_height, feature_width, 9]
    rois_w = anchors_w / torch.exp(deltas[:,:,:,3]) #[feature_height, feature_width, 9]

    rois_x_min = (rois_x - rois_h / 2).view([feature_height, feature_width, K, 1])  #[feature_height, feature_width, 9, 1]
    rois_y_min = (rois_y - rois_w / 2).view([feature_height, feature_width, K, 1])  #[feature_height, feature_width, 9, 1]
    rois_x_max = (rois_x + rois_h / 2).view([feature_height, feature_width, K, 1])  #[feature_height, feature_width, 9, 1]
    rois_y_max = (rois_y + rois_w / 2).view([feature_height, feature_width, K, 1])  #[feature_height, feature_width, 9, 1]
    
    rois = torch.cat([rois_x_min, rois_y_min, rois_x_max, rois_y_max], dim=3)   #[feature_height, feature_width, 9, 4]
    return rois

def _roi_filter(rois, probs):
    assert isinstance(rois, Variable)
    assert isinstance(probs, Variable)
    # TODO
    return rois