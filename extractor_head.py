import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.roipooling import RoIPool
from utils.loss import delta_loss

def get_vgg16_extractor_and_head(n_class, roip_size=7):
    vgg16_net = vgg16(pretrained=False)
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
        super(_VGG16Head, self).__init__()
        self.n_class_bg = n_class_bg
        self.roip_size = roip_size

        self.roip = RoIPool(roip_size, roip_size)
        self.classifier = classifier
        self.delta = nn.Linear(in_features=4096, out_features=n_class_bg*4)    # Note: predice a delta for each class
        self.score = nn.Linear(in_features=4096, out_features=n_class_bg)
    
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

        rois = Variable(torch.FloatTensor(rois))
        if torch.cuda.is_available():
            rois = rois.cuda()
        roipool_out = self.roip(feature_map, rois, spatial_scale=feature_image_scale)

        roipool_out = roipool_out.view(roipool_out.size(0), -1) # (N, 25088)
        mid_output = self.classifier(roipool_out)   # (N, 4096)
        delta_per_class = self.delta(mid_output)    # (N, n_class_bg*4)
        score = self.score(mid_output)      # (N, n_class_bg)
        #---------- debug
        assert isinstance(delta_per_class, Variable) and isinstance(score, Variable)
        assert delta_per_class.shape[0] == score.shape[0] == rois.shape[0]
        assert delta_per_class.shape[1] == score.shape[1] * 4 == self.n_class_bg * 4
        assert len(delta_per_class.shape) == len(score.shape) == 2
        #---------- debug
        return delta_per_class, score

    def loss(self, score, delta_per_class, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi):
        """
        Args:
            score: (N,2)
            delta_per_class: (N,4*n_class_bg)
            target_delta_for_sample_roi: (N, 4)
            bbox_bg_label_for_sample_roi: (N,)
        """
        #---------- debug
        assert isinstance(score, Variable)
        assert isinstance(delta_per_class, Variable)
        assert isinstance(target_delta_for_sample_roi, Variable)
        assert isinstance(bbox_bg_label_for_sample_roi, Variable)
        #---------- debug
        n_sample = score.shape[0]
        delta_per_class = delta_per_class.view(n_sample, -1, 4)
        delta = delta_per_class[torch.arange(0, n_sample).long(), bbox_bg_label_for_sample_roi.data]

        head_delta_loss = delta_loss(delta, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi, 1)
        head_class_loss = F.cross_entropy(score, bbox_bg_label_for_sample_roi)

        return head_class_loss + head_delta_loss

    def predict(self):
        pass
    

def _get_resnet50_extractor_and_head():
    pass


if __name__ == '__main__':
    from utils.proposal_target_creator import ProposalTargetCreator
    extractor, head, output_feature_channel = get_vgg16_extractor_and_head(20, 7)
    
    features = Variable(torch.randn(1,512,50,50))
    rois = (np.random.rand(2000,4)+[0,0,1,1])*240
    gt_bbox = (np.random.rand(10,4) + [0,0,1,1])*240
    gt_bbox_label = np.random.randint(0,20,size=10)
    
    proposal_target_creator = ProposalTargetCreator()
    sample_roi, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi = proposal_target_creator.make_proposal_target(rois, gt_bbox, gt_bbox_label)
    
    # sample_roi = Variable(torch.FloatTensor(sample_roi))
    target_delta_for_sample_roi = Variable(torch.FloatTensor(target_delta_for_sample_roi))
    bbox_bg_label_for_sample_roi = Variable(torch.LongTensor(bbox_bg_label_for_sample_roi))

    delta_per_class, score = head.forward(features, sample_roi, image_size=(500,500))
    loss = head.loss(score, delta_per_class,target_delta_for_sample_roi,bbox_bg_label_for_sample_roi)
    print(loss)
    loss.backward()
