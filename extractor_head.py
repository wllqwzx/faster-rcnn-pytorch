import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils.roipooling import RoIPool

def get_vgg16_extractor_and_head(n_class, roip_size=7):
    vgg16_net = vgg16(pretrained=True)
    features = list(vgg16_net.features)[0:30]
    
    for layer in features[0:10]:    # freeze top 4 conv2d layers
        for p in layer.parameters():
            p.requires_grad = False
    extractor = nn.Sequential(*features)
    output_feature_channel = 512

    classifier = list(vgg16_net.classifier)
    del(classifier[6])  # delete last fc layer
    classifier = nn.Sequential(*classifier)     # classifier : (N,25088) -> (N,4096); 25088 = 512*7*7 = C*H*W
    head = _VGG16Head(n_class_bg=n_class+1, roip_size=roip_size, classifier=classifier)
    return extractor, head, output_feature_channel


class _VGG16Head(nn.Module):
    def __init__(self, n_class_bg, roip_size, classifier):
        """n_class_bg: n_class plus background = n_class + 1"""
        super().__init__()
        self.n_class_bg = n_class_bg
        self.roip_size = roip_size

        self.roip = RoIPool(roip_size, roip_size)
        self.classifier = classifier
        self.delta = nn.Linear(in_features=4096, out_features=n_class_bg*4)    # Note: predice a delta for each class
        self.prob = nn.Linear(in_features=4096, out_features=n_class_bg)
    
    def forward(self, feature_map, rois, image_size):
        """
        Args:
            feature_map: (N1=1,C,H,W)
            rois : (N2,4)
        """
        #---------- debug
        assert isinstance(feature_map, Variable)
        assert isinstance(rois, np.ndarray)
        assert len(feature_map.shape) == 4 and feature_map.shape[0] == 1    # batch size should be 1
        assert len(rois.shape) == 2 and rois.shape[1] == 4
        #---------- debug

        # this is important because rois are in image scale, we need to pass this ratio 
        # to roipooing layer to map roi into feature_map scale
        feature_image_scale = feature_map.shape[2] / image_size[0]  
        
        # meet roi_pooling's input requirement
        temp = np.zeros((rois.shape[0], 1), dtype=rois.dtype)
        rois = np.concatenate([temp, rois], axis=1) 

        roipool_out = self.roip(feature_map, rois, spatial_scale=feature_image_scale)

        roipool_out = roipool_out.view(roipool_out.size(0), -1) # (N, 25088)
        mid_output = self.classifier(roipool_out)   # (N, 4096)
        delta_for_class = self.delta(mid_output)    # (N, n_class_bg*4)
        prob = F.softmax(self.prob(mid_output))     # (N, n_class_bg)
        #---------- debug
        assert isinstance(delta_for_class, Variable) and isinstance(prob, Variable)
        assert delta_for_class.shape[0] == prob.shape[0] == rois.shape[0]
        assert delta_for_class.shape[1] == prob.shape[1] * 4 == self.n_class_bg * 4
        assert len(delta_for_class.shape) == len(prob.shape) == 2
        #---------- debug
        return delta_for_class, prob

    def loss(self):
        pass
    
    def predice(self):
        pass
    

def _get_resnet50_extractor_and_head():
    pass
