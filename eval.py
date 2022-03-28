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
import argparse
import yaml
from easydict import EasyDict as edict
import reg_lib.models.crnn as crnn
import reg_lib.config.lp_chars as alphabets
import time

class End2EndEval():
    def __init__(self, valid_loader, iou_threshold=0.5, display=False, name_label_dict={}):

        self.valid_loader = valid_loader
        self.loss_avg_valid = utils.averager()
        self.display = display

        self.name_label_dic = name_label_dict

        self.iou_threshold = iou_threshold

    def eval(self, seg_net, reg_net):
        seg_net.eval()
        reg_net.eval()
        npos = 0
        tp = 0
        fp = 0

        has_got_display_data = False
        representer = SegDetectorRepresenter()
        with torch.no_grad():
            i = 0
            for batch in self.valid_loader:
                loss, pred = seg_net(batch)
                self.loss_avg_valid.add(loss)

                boxes_batch, scores_batch = representer.represent(batch, pred)

                if not has_got_display_data and self.display:
                    display_img_map = self.get_display_data(batch, boxes_batch, scores_batch)

                    display_pre_heat_map = self.get_pre_heat_map(batch, pred)
                    display_img_map['heat_map'] = display_pre_heat_map

                    display_img_map['mask'] = self.get_first_mask(batch)

                    has_got_display_data = True

                targets_batch = batch['lines']
                img_path_list = batch['filename']
                img_ori_batch = batch['image_ori']

                reg_input_tensor, detected_index_flag = get_reg_input_tensor(img_ori_batch, boxes_batch)
                lp_regResult_list = self.reg_pipel(reg_net, reg_input_tensor)   # reg网络的输出结果，需要和输入的图像对齐
                alignment_reg_result = self.get_alignment_reg_result(detected_index_flag, lp_regResult_list)

                start = time.time()
                for targets, poly_pre, scores_pre, reg_result, img_path in zip(targets_batch, boxes_batch, scores_batch, alignment_reg_result, img_path_list):
                    img_name = img_path.split('/')[-1]
                    lp_char_label = self.name_label_dic[img_name]
                    npos += len(targets)
                    tp_, fp_ = self.counter(targets, poly_pre, scores_pre, reg_result, lp_char_label)
                    tp += tp_
                    fp += fp_
                finished = time.time()
                i += 1
                print('%dth batch dec and reg counter time consume: %5f'%(i, finished-start))
        precision = 0  if (tp + fp)==0 else tp / (tp + fp)
        recall = tp / npos
        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        loss_val = self.loss_avg_valid.val()
        self.loss_avg_valid.reset()

        print('total_pos:%d, tp:%d, fp:%d'%(npos, tp, fp))
        print('end2end precision: %3f'%(recall))
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

    def get_alignment_reg_result(self, detected_index_flag, lp_regResult_list):   # lp_regResult_list为None时，detected_index_flag=[-1, -1,...]
                                                                            # len(detected_index_flag)==batch_size
        alignment_lp_regResult = []
        index_regRes_raw = 0
        for i in range(len(detected_index_flag)):    # len(detected_index_flag) == batch_size
            index_flag = detected_index_flag[i]
            if index_flag != -1:
                alignment_lp_regResult.append(lp_regResult_list[index_regRes_raw])
                index_regRes_raw += 1
            else:
                alignment_lp_regResult.append('')
        return alignment_lp_regResult

    def counter(self, targets, poly_pre, scores_pre, reg_result, lp_char_label):
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
            if flag and reg_result==lp_char_label:
                tp += 1.0
            else:
                fp += 1.0
        return tp, fp

    def reg_pipel(self, reg_net, reg_input_tensor):

        # seg网络没有检测到目标
        if reg_input_tensor == None:
            return None

        reg_input_tensor = reg_input_tensor.cuda()
        # inference
        preds, stn_out = reg_net(reg_input_tensor)
        preds = preds.cpu()

        batch_size = reg_input_tensor.size(0)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)

        return sim_preds

    def get_union(self, pD, pG):
        areaA = pD.area
        areaB = pG.area
        return areaA + areaB - self.get_intersection(pD, pG);

    def get_intersection(self, pD, pG):

        if not pD.intersects(pG):
            return 0.0
        return pD.intersection(pG).area

def get_reg_input_tensor(image_ori_batch, batch_boxes):      # batch_boxe: [array(-1, 4, 2)]

    reg_input_tensor = None
    lp_area_list = []
    detected_index_flag = []
    for img_idx in range(len(image_ori_batch)):
        one_img_boxes = batch_boxes[img_idx]    # array, one image may has many targets
        one_img_ori = image_ori_batch[img_idx]  # tensor

        # 没有检测到目标
        if one_img_boxes.shape[0] == 0:
            detected_index_flag.append(-1)
            continue

        detected_index_flag.append(img_idx)
        box = np.array(one_img_boxes[0]).astype(np.int32)
        x1 = np.min(box[:, 0])
        y1 = np.min(box[:, 1])
        x2 = np.max(box[:, 0])
        y2 = np.max(box[:, 1])

        lp_area = (one_img_ori[y1:y2 + 1, x1:x2 + 1] / 255. - 0.588) / 0.193
        # ratio resize
        h, w, _ = lp_area.shape
        lp_area = lp_area.numpy()
        lp_area = cv2.resize(lp_area, (0, 0), fx=160 / w, fy=32 / h,interpolation=cv2.INTER_CUBIC)
        lp_area = np.reshape(lp_area, (config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W, 3))
        lp_area = torch.from_numpy(lp_area).permute(2, 0, 1).float()
        lp_area_list.append(lp_area)

    if lp_area_list != []:
        reg_input_tensor = torch.stack(lp_area_list)
    return reg_input_tensor, detected_index_flag

def get_ccpd_base_lpChar(txt_path, char_file):
    img_name_lpChars_dict = {}
    with open(char_file, 'rb') as file:
        char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}
    with open(txt_path, 'r', encoding='utf-8') as file:
        contents = file.readlines()
        for c in contents:
            imgname = c.strip()
            indices = c.split('-')[-3].split('_')
            province = char_dict[int(indices[0]) + 35]
            label_dig = indices[1:]
            label_char = ''.join([char_dict[int(idx) + 1] for idx in label_dig])
            label_str = province + label_char
            img_name_lpChars_dict[imgname] = label_str
    return img_name_lpChars_dict

def parse_arg():
    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='reg_lib/config/lp.yaml')
    parser.add_argument('--checkpoint', type=str, default='output/CCPD/crnn/2022-03-17-14-19-ccpd_base_0.996ac/checkpoints/checkpoint_64_acc_0.9950.pth',
                        help='the path to your checkpoints')
    parser.add_argument('--seg_weight', type=str,
                        default='experiments/ccpd_baseSegNet_838_7502_765000.pth',
                        )
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda')

    config, args = parse_arg()
    seg_weight_path = args.seg_weight
    reg_weight_path = args.checkpoint

    seg_net = SegNet(device)
    reg_net = crnn.get_crnn(config).to(device)

    representer = SegDetectorRepresenter()
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

    # 初始化模型
    seg_net.load_state_dict(torch.load(seg_weight_path))
    reg_net.load_state_dict(torch.load(reg_weight_path)['state_dict'])
    print('loading seg pretrained model from {0}'.format(seg_weight_path))
    print('loading reg pretrained model from {0}'.format(reg_weight_path))

    ccpd_valid_dataset = CCPDDataset('/home/admin1/datasets/CCPD2019/ccpd_base', '/home/admin1/datasets/CCPD2019/base_test.txt', use_argument=False)
    ccpd_base_test_txt = '/home/admin1/datasets/CCPD2019/base_test.txt'
    lp_char_file = 'reg_lib/dataset/txt/LP_char.txt'
    img_name_lpChars_dict = get_ccpd_base_lpChar(ccpd_base_test_txt, lp_char_file)

    valid_loader = DataLoader(ccpd_valid_dataset, 4, collate_fn=CollectFN())
    eval_operator = End2EndEval(valid_loader, iou_threshold=0.5, display=False, name_label_dict=img_name_lpChars_dict)
    eval_operator.eval(seg_net, reg_net)
