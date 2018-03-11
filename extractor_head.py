import torch
import torch.nn as nn
from torchvision.models import vgg16
from roipooling import roi_pooling
import torch.nn.functional as F

def get_vgg16_extractor_and_head(n_class, roip_size=7):
    vgg16_net = vgg16(pretrained=True)
    features = list(vgg16_net.features)[0:30]
    
    for layer in features[0:10]:    # freeze top 4 conv2d layers
        for p in layer.parameters():
            p.requires_grad = False
    extractor = nn.Sequential(*features)
    extractor_channel = 512 # extractor channel: 512

    classifier = list(vgg16_net.classifier)
    del(classifier[6])  # delete last fc layer
    classifier = nn.Sequential(*classifier)     # classifier : 25088 dim -> 4096 dim; 25088 = 7*7*512 = H*W*C(roip)
    head = _VGG16Head(n_class_bg=n_class+1, roip_size=roip_size, classifier=classifier)
    return extractor, head, extractor_channel


class _VGG16Head(nn.Module):
    def __init__(self, n_class_bg, roip_size, classifier):  # n_class_bg: n_class plus background
        super().__init__()
        self.n_class_bg = n_class_bg
        self.roip_size = roip_size

        self.classifier = classifier
        self.delta = nn.Linear(in_features=4096, out_features=n_class_bg*4)    # Note: each class per delta
        self.prob = nn.Linear(in_features=4096, out_features=n_class_bg)
    
    def forward(self, feature_map, rois, image_size):
        """
        * **feature_map** : type `Variable`, shape `[batch=1, channel, height, width]`
        * **rois** : type Variable, shape: `[num_left_rois, 4]`
        """
        feature_image_scale = feature_map.shape[2] / image_size[0]  #!!!
        
        temp = torch.zeros(rois.shape[0], 1)
        assert temp.type() == rois.type()
        rois = torch.cat([temp, rois], dim=1)   # meet roi_pooling's input requirement
        # roipool_out : [num_left_rois, channel, roip_size, roip_size]
        roipool_out = roi_pooling(feature_map, rois, size=(self.roip_size, self.roip_size), spatial_scale=feature_image_scale)

        roipool_out = roipool_out.view(roipool_out.size(0), -1) # [num_left_rois, roip_size*roip_size*channel=25088]
        mid_output = self.classifier(roipool_out)   # [num_left_rois, 4096]
        delta_for_class = self.delta(mid_output)    # [num_left_rois, n_class_bg*4]
        prob = F.softmax(self.prob(mid_output))     # [num_left_rois, n_class_bg]
        return delta_for_class, prob


def _get_resnet50_extractor_and_head():
    pass
