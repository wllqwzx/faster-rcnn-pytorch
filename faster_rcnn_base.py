import torch
import torch.nn as nn
from torch.autograd import Variable


class FasterRCNNBase(nn.Module):
    def __init__(self, feature_extractor, rpn, head):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.rpn = rpn
        self.head = head
    
    def forward(self, x):
        img_size = x.shape[2:]
        feature = self.feature_extractor(x)
        _, _, rois, rois_indices, _ = self.rpn(feature, img_size)
        roi_locs, roi_probs = self.head(feature, rois, rois_indices)
        return roi_locs, roi_probs, rois_indices
    
    def _suppress(self, bbox_locs, bbox_probs):
        pass
    
    def predict(self):
        pass

