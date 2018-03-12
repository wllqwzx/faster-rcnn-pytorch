import numpy as np

def delta2bbox(src_bbox, delta):
    """
    src_bbox: (N_bbox, 4)
    delta:  (N_bbox, 4)
    """
    #---------- debug
    assert src_bbox.shape == delta.shape
    assert isinstance(src_bbox, np.ndarray)
    assert isinstance(delta, np.ndarray)

    #----------
    src_bbox_h = src_bbox[:,2] - src_bbox[:,0]  
    src_bbox_w = src_bbox[:,3] - src_bbox[:,1]
    src_bbox_x = src_bbox[:,0] + src_bbox_h/2 
    src_bbox_y = src_bbox[:,1] + src_bbox_w/2 

    dst_bbox_x = src_bbox_x + src_bbox_h*delta[:,0] 
    dst_bbox_y = src_bbox_y + src_bbox_w*delta[:,1] 
    dst_bbox_h = src_bbox_h / np.exp(delta[:,2])
    dst_bbox_w = src_bbox_w / np.exp(delta[:,3])

    dst_bbox_x_min = (dst_bbox_x - dst_bbox_h / 2).reshape([-1, 1])
    dst_bbox_y_min = (dst_bbox_y - dst_bbox_w / 2).reshape([-1, 1])
    dst_bbox_x_max = (dst_bbox_x + dst_bbox_h / 2).reshape([-1, 1])
    dst_bbox_y_max = (dst_bbox_y + dst_bbox_w / 2).reshape([-1, 1])
    
    dst_bbox = np.concatenate([dst_bbox_x_min, dst_bbox_y_min, dst_bbox_x_max, dst_bbox_y_max], axis=1)   #(N_dst_bbox, 4)
    return dst_bbox

def bbox2delta(src_bbox, dst_bbox):
    """
    src_bbox: (N_bbox, 4)
    dst_bbox: (N_bbox, 4)
    """
    #---------- debug
    assert isinstance(src_bbox, np.ndarray)
    assert isinstance(dst_bbox, np.ndarray)
    assert src_bbox.shape == dst_bbox.shape
    #----------
    src_h = src_bbox[:, 2] - src_bbox[:, 0]
    src_w = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_h
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_w

    dst_h = dst_bbox[:, 2] - dst_bbox[:, 0]
    dst_w = dst_bbox[:, 3] - dst_bbox[:, 1]
    dst_ctr_x = dst_bbox[:, 0] + 0.5 * dst_h
    dst_ctr_y = dst_bbox[:, 1] + 0.5 * dst_w

    eps = np.finfo(src_h.dtype).eps
    height = np.maximum(src_h, eps)
    width = np.maximum(src_w, eps)

    dx = (dst_ctr_x - src_ctr_x) / height
    dy = (dst_ctr_y - src_ctr_y) / width
    dh = np.log(dst_h / height)
    dw = np.log(dst_w / width)

    dx = dx.reshape([-1,1])
    dy = dy.reshape([-1,1])
    dh = dh.reshape([-1,1])
    dw = dw.reshape([-1,1])
    delta = np.concatenate([dx, dy, dh, dw], axis=1)
    return delta


def bbox_iou(bbox1, bbox2):
    


if __name__ == '__main__':
    src_bbox = np.random.random((2500, 4)) + [0,0,1,1]
    delta = np.random.randn(2500,4)
    dst_bbox = delta2bbox(src_bbox, delta)
    assert dst_bbox.shape == src_bbox.shape
    print("delta2bbox passed")

    src_bbox = np.random.random((2500, 4)) + [0,0,1,1]
    dst_bbox = np.random.random((2500, 4)) + [0,0,1,1]
    delta = bbox2delta(src_bbox, dst_bbox)
    assert delta.shape == src_bbox.shape
    print("bbox2delta passed")