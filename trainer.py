import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from chainercv.datasets import VOCBboxDataset
from chainercv.datasets import voc_bbox_label_names
from chainercv.visualizations import vis_bbox
import torch
from torchnet.meter import AverageValueMeter
from torchvision import transforms


from model.faster_rcnn import faster_rcnn
    

train_dataset = VOCBboxDataset(year='2007', split='train')
val_dataset = VOCBboxDataset(year='2007', split='val')
trainval_dataset = VOCBboxDataset(year='2007', split='trainval')
test_dataset = VOCBboxDataset(year='2007', split='test')

model = faster_rcnn(20, backbone='vgg16')
if torch.cuda.is_available():
    model = model.cuda()


# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, momentum=0.9)
optimizer = model.get_optimizer()

model.train()
avg_loss = AverageValueMeter()


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()

def adjust_learning_rate(optimizer, epoch, init_lr, lr_decay_factor=0.1, lr_decay_epoch=10):
    """Sets the learning rate to the initial LR decayed by lr_decay_factor every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch == 0:
        lr = init_lr * (lr_decay_factor ** (epoch // lr_decay_epoch))
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

for epoch in range(14):
    adjust_learning_rate(optimizer, epoch, 0.001, lr_decay_epoch=9)
    for i in range(100):
        img, bbox, label = train_dataset[i]
        img = pytorch_normalze(img/255)
        image_size = img.shape[1:]
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        loss = model.loss(img, bbox, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.cpu().data.numpy()[0]
        avg_loss.add(loss_value)
        print('[epoch:{}]  [batch:{}/{}]  [sample_loss:{:.4f}]  [avg_loss:{:.4f}]'.format(epoch, i, len(train_dataset), loss_value, avg_loss.value()[0]))



model.eval()
for i in range(100):
    img, _, _ = train_dataset[i]
    imgx = pytorch_normalze(img/255)
    imgx = imgx.reshape(1, imgx.shape[0], imgx.shape[1], imgx.shape[2])
    bbox_out, class_out, prob_out = model.predict(imgx, prob_threshold=0.7)
    vis_bbox(img, bbox_out, class_out, prob_out,label_names=voc_bbox_label_names)
    # plt.show()

    fig = plt.gcf()
    fig.set_size_inches(11, 5)
    fig.savefig('test_frcnn_'+str(i)+'.pdf', dpi=100)



