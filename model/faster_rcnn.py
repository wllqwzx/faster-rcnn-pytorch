import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from model.rpn import rpn
from model.extractor_head import get_vgg16_extractor_and_head
from model.utils.proposal_target_creator import ProposalTargetCreator

class _Faster_RCNN_Maker(nn.Module):
    def __init__(self, feature_extractor, rpn, head):
        super(_Faster_RCNN_Maker, self).__init__()
        self.feature_extractor = feature_extractor
        self.rpn = rpn
        self.head = head
        self.proposal_target_creator = ProposalTargetCreator()

    def forward(self):
        raise NotImplementedError("Do not call forward directly! Instead, calling .loss in traininig phase and .predict in inference phase!")


    def loss(self, image, gt_bbox, gt_bbox_label):
        """
        image: (N=1,3,H,W)
        gt_bbox: (N2,4)
        gt_bbox_label: (N2,)
        """
        if self.training == False:
            raise Exception("Do not call loss in eval mode, you should call .train() to set the model in train model!")
        #-------- debug
        assert isinstance(image, np.ndarray)
        assert isinstance(gt_bbox, np.ndarray)
        assert isinstance(gt_bbox_label, np.ndarray)
        assert image.shape[0] == 1
        assert gt_bbox.shape[0] == gt_bbox_label.shape[0]
        #-------- debug
        image = Variable(torch.FloatTensor(image))
        if torch.cuda.is_available():
            image = image.cuda()

        features = self.feature_extractor(image)
        image_size = image.shape[2:]

        # rpn loss
        delta, score, anchor = self.rpn.forward(features, image_size)
        rpn_loss = self.rpn.loss(delta, score, anchor, gt_bbox, image_size)

        #=====!!!!!
        # print("rpn delta mean:", delta.data.cpu().numpy().mean())

        # head loss:
        roi = self.rpn.predict(delta, score, anchor, image_size)
        sample_roi, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi = self.proposal_target_creator.make_proposal_target(roi, gt_bbox, gt_bbox_label)

        #=====!!!!!!
        # print("background:",(bbox_bg_label_for_sample_roi == 0).sum())
        # print("sample_roi number:", sample_roi.shape[0])

        delta_per_class, score = self.head.forward(features, sample_roi, image_size)

        #=====!!!!!
        # print("head delta mean:", delta_per_class.data.cpu().numpy().mean())

        head_loss = self.head.loss(score, delta_per_class, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi)

        return rpn_loss + head_loss



    def predict(self, image, prob_threshold=0.5):
        """
        image: (N=1,3,H,W)
        """
        #---------- debug
        assert isinstance(image, np.ndarray)
        #---------- debug
        if self.training == True:
            raise Exception("Do not call predict in training mode, you should call .eval() to set the model in eval mode!")
        image = Variable(torch.FloatTensor(image))
        if torch.cuda.is_available():
            image = image.cuda()

        features = self.feature_extractor(image)
        image_size = image.shape[2:]

        delta, score, anchor = self.rpn.forward(features, image_size)
        roi = self.rpn.predict(delta, score, anchor, image_size)
        #------!!!!!!
        # print("roi number:", roi.shape[0])

        delta_per_class, score = self.head.forward(features, roi, image_size)       
        bbox_out, class_out, prob_out = self.head.predict(roi, delta_per_class, score, image_size, prob_threshold=prob_threshold)
        
        return bbox_out, class_out, prob_out

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = 0.001
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]
        if False:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer



def faster_rcnn(n_class, backbone='vgg16', model_path=None):
    if backbone == 'vgg16':
        if model_path is None:
            vgg_pretrained = True
        else:
            vgg_pretrained = False

        extractor, head, feature_dim = get_vgg16_extractor_and_head(n_class, roip_size=7, vgg_pretrained=vgg_pretrained)
        rpn_net = rpn(in_channel=feature_dim, mid_channel=512,ratio=[0.5, 1, 2], anchor_size=[128, 256, 512])
        model = _Faster_RCNN_Maker(extractor, rpn_net, head)
        return model
    elif backbone == 'resnet50':
        raise ValueError("resnet50 has not been implemented!")
    else:
        raise ValueError("backbone only support vgg16, resnet50!")
    

if __name__ == '__main__':
    model = faster_rcnn(20, backbone='vgg16')
    if torch.cuda.is_available():
        model = model.cuda()
    image = np.random.rand(1,3,500,600)*255
    gt_bbox = (np.random.rand(10,4) + [0,0,1,1])*240
    gt_bbox_label = np.random.randint(0,20,size=10)

    model.train()
    loss = model.loss(image, gt_bbox, gt_bbox_label)
    loss.backward()
    print(loss)
 
    model.eval()
    bbox_out, class_out, prob_out = model.predict(image)
    print(bbox_out.shape)
    print(class_out.shape)
    print(prob_out.shape)
