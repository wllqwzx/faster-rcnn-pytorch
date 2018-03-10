import torch
import torch.nn as nn
from torchvision.models import vgg16
from roipooling import RoIPooing2D

def get_extractor_and_head(n_class ,backbone="vgg16"):
    """
    * **backbone** : can choose from `vgg16`, `resnet50`...
    """
    if backbone == 'vgg16':
        extractor, head = _get_vgg16_extractor_and_head(n_class)
    elif backbone == 'resnet50':
        pass
    else:
        raise ValueError
    return extractor, head




def _get_vgg16_extractor_and_head(n_class, roip_size=7):
    model = vgg16(pretrained=False)
    features = list(model.features)[0:30]
    
    for layer in features[0:10]:    # freeze top 4 conv2d layers
        for p in layer.parameters():
            p.requires_grad = False
    extractor = nn.Sequential(*features)

    classifier = list(model.classifier)
    del(classifier[6])  # delete last fc layer
    classifier = nn.Sequential(*classifier)     # classifier : 25088 dim -> 4096 dim; 25088 = 7*7*512 = H*W*C(roip)
    head = _VGG16Head(n_class_bg=n_class+1, roip_size=roip_size, classifier=classifier)
    return extractor, head


class _VGG16Head(nn.Module):
    def __init__(self, n_class_bg, roip_size, classifier):  # n_class_fg: n_class plus background
        super().__init__()
        self.n_class_bg = n_class_bg
        self.roip_size = roip_size

        self.roi_pooling = RoIPooing2D(self.roip_size, self.roip_size)
        self.classifier = classifier
        self.delta = nn.Linear(in_features=4096, out_features=n_class_bg*4)    # Note: each class per delta
        self.prob = nn.Linear(in_features=4096, out_features=n_class_bg)
    
    def forward(self, feature_map, rois):
        """
        * **feature_map** : type `Variable`, shape `[batch=1, channel, height, width]`
        * **rois** : type Variable, shape: `[num_left_rois, K, 4]`
        """
        roipool_out = self.roi_pooling(feature_map, rois)       # [num_left_rois, channel, roip_size, roip_size]
        roipool_out = roipool_out.view(roipool_out.size(0), -1) # [num_left_rois, roip_size*roip_size*channel=25088]
        mid_output = self.classifier(roipool_out)   # [num_left_rois, 4096]
        delta = self.delta(mid_output)  # [num_left_rois, n_class_bg*4]
        prob = self.prob(mid_output)    # [num_left_rois, n_class_bg]
        return delta, prob


def _get_resnet50_extractor_and_head():
    pass
