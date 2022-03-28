import cv2
import numpy as np
import torch
from model.segnet import SegNet
from tool.representers.seg_representor import SegDetectorRepresenter
import time


def resize_image(img, image_short_side):

    img_new = np.zeros((image_short_side, image_short_side, 3))
    h, w, _ = img.shape
    scale = min(image_short_side*1.0/h, image_short_side*1.0/w)
    img_resize = cv2.resize(img, (int(scale*w), int(scale*h)))
    img_new[0:img_resize.shape[0], 0:img_resize.shape[1]] = img_resize

    return img_new

def load_image(image_path, image_short_side):
    # RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
    img = cv2.imread(image_path)
    original_shape = img.shape[:2]
    img = resize_image(img, image_short_side)
    # img -= RGB_MEAN
    # img /= 255.
    img = (img / 255. - 0.588) / 0.193
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    h, w = original_shape[0], original_shape[1]
    resize_scale = min(image_short_side*1.0/h, image_short_side*1.0/w)
    return img, original_shape, resize_scale

def demo_visualize(image_path, output):
    boxes, _ = output
    boxes = boxes[0]
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_shape = original_image.shape
    pred_canvas = original_image.copy().astype(np.uint8)
    pred_canvas = cv2.resize(pred_canvas, (original_shape[1], original_shape[0]))

    for box in boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(pred_canvas, [box], True, (0, 255, 0), 2)

    return pred_canvas


if __name__=='__main__':
    weight_path = 'experiments/ccpd_baseSegNet_838_7502_765000.pth'
    image_path = 'mask.jpg'
    image_short_side = 640
    device = torch.device('cuda')
    seg_net = SegNet(device)
    seg_net.load_state_dict(torch.load(weight_path), strict=False)
    representer = SegDetectorRepresenter()# is_poly=True

    started = time.time()
    img, original_shape, resize_scale = load_image(image_path, image_short_side)
    batch = dict()
    batch['original_size'] = [original_shape]
    batch['filename'] = [image_path]

    batch['resize_scale'] = [resize_scale]
    with torch.no_grad():
        batch['image'] = img
        started = time.time()
        pred = seg_net.forward(batch)
        # finished = time.time()
        output = representer.represent(batch, pred)
        finished = time.time()
        vis_image = demo_visualize(image_path, output)
        cv2.imwrite( 'seg_demo_ccpd1_output.jpg', vis_image)
        print('elapsed time: {0}'.format(finished - started))
    # original_h, original_w, c = img.shape
    # img_in, scale = resize_img(img, (640,640))
    # img_in = torch.from_numpy(img_in).permute(2, 0, 1).unsqueeze(0)
    # input = {}
    # input['image'] = img_in
    # input['resize_scale'] = [scale]
    #
    # mask = np.zeros((1,640,640))
    # mask[0][0][0] = 1
    # mask = torch.from_numpy(mask).unsqueeze(0).float()
    # input['gt'] = mask


    #
    # loss, pred, metrics = net(input)
    # boxes_batch, scores_batch = representer.represent(input, pred)
    # print('max pred map probability:%s'%torch.max(pred))
    # print('pred map: %s'%pred)