# faster-rcnn-pytorch
This is a pytorch implementation of faster RCNN.

## Demo

```python
import matplotlib.pyplot as plt
from chainercv.visualizations import vis_bbox
from chainercv.datasets import voc_bbox_label_names
from model.faster_rcnn import faster_rcnn


path = 'pretrained-model-path'
model = faster_rcnn(n_class=20, model_path=path).cuda()
model.eval()

img = plt.imread('image/example0.jpg')
img = img.transpose(2,0,1)
imgx = img/255
bbox_out, class_out, prob_out = model.predict(imgx, prob_threshold=0.95)
vis_bbox(img, bbox_out, class_out, prob_out,label_names=voc_bbox_label_names) 
plt.show()
```

## Train

This project uses the convenient api provided by chainercv to download and load the Pascal VOC2007 dataset, please install it before running the training script.

```bash
$pip3 install chainercv
$python3 trainer.py
```



## Reference



* [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
* [simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)
* [chainercv](https://github.com/chainer/chainercv/tree/master/chainercv/links/model/faster_rcnn)