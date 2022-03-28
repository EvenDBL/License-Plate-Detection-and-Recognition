import numpy as np
from shapely.geometry import Polygon
import cv2
from tool import utils
from tool.representers.seg_representor import SegDetectorRepresenter
from model.segnet import SegNet
import torch
import os
from data.seg_dataset import NormalizeImage
from torch.utils.data import DataLoader
from data.seg_dataset import CCPDDataset
from data.seg_dataset import CollectFN


class Eval():
    def __init__(self, valid_loader, iou_threshold=0.5, display=False):

        self.valid_loader = valid_loader
        self.loss_avg_valid = utils.averager()
        self.display = display

        self.iou_threshold = iou_threshold

    def eval(self, net):

        npos = 0
        tp = 0
        fp = 0

        has_got_display_data = False
        representer = SegDetectorRepresenter()
        with torch.no_grad():
            for batch in self.valid_loader:
                loss, pred = net(batch)
                self.loss_avg_valid.add(loss)

                boxes_batch, scores_batch = representer.represent(batch, pred)

                if not has_got_display_data and self.display:
                    display_img_map = self.get_display_data(batch, boxes_batch, scores_batch)

                    display_pre_heat_map = self.get_pre_heat_map(batch, pred)
                    display_img_map['heat_map'] = display_pre_heat_map

                    display_img_map['mask'] = self.get_first_mask(batch)

                    has_got_display_data = True

                targets_batch = batch['lines']

                '''
                 targets: [{'poly':  ,'text':  },{'poly':  ,'text':  },{'poly':  ,'text':  }]
                 poly_pre: i th image boxes_pre,   shape:[x, 4, 2]
                 scores_pre: i th image scores_pre,  shape:[x,]
                '''

                for targets, poly_pre, scores_pre in zip(targets_batch, boxes_batch, scores_batch):
                    npos += len(targets)
                    tp_, fp_ = self.counter(targets, poly_pre, scores_pre)
                    tp += tp_
                    fp += fp_

        precision = 0  if (tp + fp)==0 else tp / (tp + fp)
        recall = tp / npos
        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        loss_val = self.loss_avg_valid.val()
        self.loss_avg_valid.reset()

        print('total_pos:%d, tp:%d, fp:%d'%(npos, tp, fp))
        if self.display:
            return precision, recall, hmean, loss_val, display_img_map
        else:
            return precision, recall, hmean, loss_val, None

    def get_first_mask(self, batch):
        original_h, original_w = batch['original_size'][0]
        scale = batch['resize_scale'][0]
        targets = batch['lines'][0]  # [{},{},{}]
        mask = np.zeros((original_h, original_w, 3)).astype('uint8')
        for item in targets:
            poly = item['poly']  # [[],[],[],[]]
            poly_array = (np.array(poly)).astype('int')
            cv2.fillPoly(mask, [poly_array], (255, 255, 255))
        return mask

    def get_pre_heat_map(self, batch, pred):
        scale = batch['resize_scale'][0]

        pred_map = np.array(pred[0][0].detach().cpu())   # numpy array shape: 640x640
        in_h, in_w= pred_map.shape
        resize_h = int(np.ceil(in_h / scale))
        resize_w = int(np.ceil(in_w / scale))
        pred_map = cv2.resize(pred_map, (resize_w, resize_h))

        original_size = batch['original_size'][0]
        ori_h, ori_w = original_size

        pre_map_new = np.zeros((ori_h, ori_w))

        pre_map_new[0:ori_h, 0:ori_w] = pred_map[0:ori_h, 0:ori_w]

        return pre_map_new

    def get_display_data(self, batch, boxes_batch, scores_batch):

        display_img_map = {}
        scale = batch['resize_scale'][0]
        img_in = batch['image'][0]
        original_size = batch['original_size'][0]
        ori_h, ori_w = original_size

        img_in = NormalizeImage().restore(img_in)
        in_h, in_w, _ = img_in.shape
        resize_h = int(np.ceil(in_h/scale))
        resize_w = int(np.ceil(in_w/scale))
        img_original = np.zeros((ori_h, ori_w, 3)).astype('uint8')
        img_resize = cv2.resize(img_in, (resize_w, resize_h))

        img_original[0:ori_h, 0:ori_w] = img_resize[0:ori_h, 0:ori_w]

        display_img_map['image'] = img_original

        scores_ = scores_batch[0]
        polys_ = boxes_batch[0]
        img_poly = img_original.copy()
        mask_pre = np.zeros((ori_h, ori_w, 3)).astype('uint8')
        for i in range(len(scores_)):
            score = scores_[i]
            if score == 0.0:
                continue

            poly_points = polys_[i].astype('int')
            pre_poly = Polygon(poly_points)

            if not pre_poly.is_valid or not pre_poly.is_simple:
                continue

            cv2.polylines(img_poly, [poly_points], isClosed=True, color=(255, 0, 0), thickness=1)

            left_top = poly_points[0]
            text_location_x = max(min(ori_w-1, left_top[0]), 0)
            text_location_y = max(min(ori_h-1, left_top[1]-2), 0)
            cv2.putText(img_poly, str(score), (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.fillPoly(mask_pre, [poly_points], (255, 255, 255))

        display_img_map['img_with_poly'] = img_poly
        display_img_map['mask_pre'] = mask_pre
        return display_img_map

    def counter(self, targets, poly_pre, scores_pre):
        tp = 0
        fp = 0

        cover = set()
        for i in range(len(scores_pre)):
            score = scores_pre[i]
            if score==0:
                continue

            pre_poly_points = poly_pre[i]
            pre_poly = Polygon(pre_poly_points)

            if not pre_poly.is_valid or not pre_poly.is_simple:
                continue

            flag = False
            for gt_id, target in enumerate(targets):
                gt_poly_points = target['poly']     # list
                gt_poly_points = np.array(gt_poly_points)
                gt_poly = Polygon(gt_poly_points)

                #FIXME why the gt cannot be valid
                if not gt_poly.is_valid or not gt_poly.is_simple:
                    continue

                inter = self.get_intersection(pre_poly, gt_poly)
                union = self.get_union(pre_poly, gt_poly)

                if (inter * 1.0 / union) > self.iou_threshold:
                    if gt_id not in cover:
                        flag = True
                        cover.add(gt_id)
                        break
            if flag:
                tp += 1.0
            else:
                fp += 1.0
        return tp, fp

    def get_union(self, pD, pG):
        areaA = pD.area
        areaB = pG.area
        return areaA + areaB - self.get_intersection(pD, pG);

    def get_intersection(self, pD, pG):

        if not pD.intersects(pG):
            return 0.0
        return pD.intersection(pG).area

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    weight_path = 'experiments/aolpLeSegNet_35_100_5000.pth'
    device = torch.device('cuda')
    net = SegNet(device)
    net.load_state_dict(torch.load(weight_path))
    representer = SegDetectorRepresenter()
    ccpd_valid_dataset = CCPDDataset('/home/admin1/datasets/AOLP/Subset_LE/Image', '/home/admin1/datasets/AOLP/Subset_LE/AOLP_LE_test.txt', use_argument=False) # 暂时不能改成True
    valid_loader = DataLoader(ccpd_valid_dataset, 4, collate_fn=CollectFN())
    eval_operator = Eval(valid_loader, iou_threshold=0.5, display=False)
    print(eval_operator.eval(net))
