import cv2
import numpy as np
from torch.autograd import Variable
import reg_lib.config.lp_chars as alphabets
from PIL import Image, ImageDraw, ImageFont
import torch
from model.segnet import SegNet
from reg_lib.models.crnn import CRNN
from tool.utils import strLabelConverter
from tool.representers.seg_representor import SegDetectorRepresenter
import time
import os

class DemoOperator():
    def __init__(self, seg_net, reg_net, representer, converter):
        self.seg_net = seg_net
        self.reg_net = reg_net
        self.representer = representer
        self.converter = converter

        # 字体的格式
        ttf = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
        self.fontStyle = ImageFont.truetype(font=ttf, size=40, encoding="utf-8")

    def demoRun(self, img_path):
        if os.path.isdir(img_path):
            for img_file in os.listdir(img_path):
                reg_result = self.demoOneImg(os.path.join(img_path, img_file))
        else:
            reg_result = self.demoOneImg(img_path)

    # 单图片流
    def demoOneImg(self, img_path):
        img_name = img_path.split('/')[-1]
        total_time_start = time.time()
        img_seg_in, img_ori, original_shape, scale = self.resize_for_seg(img_path)
        #封装seg_net的数据
        batch = dict()
        batch['original_size'] = [original_shape]
        batch['resize_scale'] = [scale]
        batch['image'] = img_seg_in
        batch['img_ori'] = img_ori

        seg_start_time = time.time()
        with torch.no_grad():
            seg_pred = self.seg_net(batch)
        seg_time = time.time() - seg_start_time

        img_reg_in, boxes = self.resize_for_reg(batch, seg_pred)  # 一张图可能有多张车牌，boxes存储着检测框的坐标
        if img_reg_in.shape[0]==0:
            return '未检测到车牌'

        reg_time_start = time.time()
        with torch.no_grad():
            reg_preds, _ = self.reg_net(img_reg_in.cuda())
        reg_time = time.time() - reg_time_start

        batch_size = reg_preds.shape[1]
        preds_size = Variable(torch.IntTensor([reg_preds.size(0)] * batch_size))

        _, reg_preds = reg_preds.max(2)
        reg_preds = reg_preds.transpose(1, 0).contiguous().view(-1)
        sim_pred = self.converter.decode(reg_preds.data, preds_size.data, raw=False)
        print('reg results: {0}'.format(sim_pred))

        vis_image = self.demo_visualize(img_ori, boxes, sim_pred)
        cv2.imwrite('real_imgs_demo_out/%s' % img_name, vis_image)

        total_time_end = time.time()

        total_time = total_time_end - total_time_start
        load_img_time = total_time - seg_time - reg_time
        inference_time = total_time - load_img_time
        print('时间总消耗: %.3fs\n加载数据消耗: %.3fs\n推理消耗: %.3fs'%(total_time, load_img_time, inference_time))

        return sim_pred

    def demo_visualize(self, image_ori, boxes, reg_outs):
        # img_ori = (img_ori / 255. - 0.588) / 0.193
        # img_ori = (image_ori*0.193 + 0.588)*255
        original_image = image_ori
        ori_h, ori_w, _ = original_image.shape
        pred_canvas = original_image.copy().astype(np.uint8)
        pred_canvas = cv2.resize(pred_canvas, (ori_w, ori_h))

        for i in range(boxes.shape[0]):
            box = boxes[i]
            box = np.array(box).astype(np.int32)  # .reshape(-1, 2)
            cv2.polylines(pred_canvas, [box], True, (0, 255, 0), 2)  # 画框

            lp_str = reg_outs[i] if (isinstance(reg_outs, list)) else reg_outs
            left_down_point = box[3]
            text_location_x = max(min(ori_w - 1, left_down_point[0]), 0)
            text_location_y = max(min(ori_h - 1, left_down_point[1] - 2), 0)

            pred_canvas = Image.fromarray(pred_canvas)
            draw = ImageDraw.Draw(pred_canvas)
            draw.text((text_location_x, text_location_y), lp_str, (0, 0, 255), font=self.fontStyle )
            pred_canvas = np.array(pred_canvas)

        return pred_canvas

    def resize_for_seg(self, img_path):
        img_ori = cv2.imread(img_path)
        # img_ori = (img_ori / 255. - 0.588) / 0.193
        original_shape = img_ori.shape[:2]

        img_new = np.zeros((640, 640, 3))
        h, w, _ = img_ori.shape
        scale = min(640 * 1.0 / h, 640 * 1.0 / w)
        img_resize = cv2.resize(img_ori, (int(scale * w), int(scale * h)))
        img_new[0:img_resize.shape[0], 0:img_resize.shape[1]] = img_resize

        img_new = (img_new / 255. - 0.588) / 0.193

        img_seg_in = torch.from_numpy(img_new).permute(2, 0, 1).float().unsqueeze(0)

        return img_seg_in, img_ori, original_shape, scale

    def resize_for_reg(self, batch, seg_pred):
        output = self.representer.represent(batch, seg_pred)
        img_ori = batch['img_ori']   #
        lp_areas, boxes = self.get_lp_area(img_ori, output)
        if len(lp_areas)==0:
            return torch.tensor([]), boxes
        lp_area_batch = torch.stack(lp_areas)
        return lp_area_batch, boxes

    def get_lp_area(self, image_ori, output):

        batch_boxes, batch_scores = output  # [batch] 用list装的一个batch的图像，因为是单图片测试，所以只有一张图
        one_img_boxes = batch_boxes[0]  # one image may has many targets, 可能是空数组：array([])
        lp_area_list = []

        if one_img_boxes.shape[0] == 0:
            return lp_area_list, one_img_boxes

        for i in range(one_img_boxes.shape[0]):
            box = np.array(one_img_boxes[i]).astype(np.int32)
            x1 = np.min(box[:, 0])
            y1 = np.min(box[:, 1])
            x2 = np.max(box[:, 0])
            y2 = np.max(box[:, 1])
            lp_area = image_ori[y1:y2 + 1, x1:x2 + 1]

            # resize车牌区域，使得满足regNet的输入尺寸；转换成tensor
            h, w, _ = lp_area.shape
            lp_area = cv2.resize(lp_area, (0, 0), fx=160 / w, fy=32 / h, interpolation=cv2.INTER_CUBIC)
            lp_area = np.reshape(lp_area, (32, 160, 3)).astype(np.float32)

            lp_area = (lp_area / 255. - 0.588) / 0.193

            lp_area = lp_area.transpose([2, 0, 1])
            lp_area_tensor = torch.from_numpy(lp_area)
            lp_area_list.append(lp_area_tensor)

        return lp_area_list, one_img_boxes

if __name__=='__main__':
    seg_weight_path = 'experiments/end2end_goodWeights/clpdSegNet_900_final_778950.pth'
    reg_weight_path = 'experiments/end2end_goodWeights/checkpoint_2945_acc_0.9133.pth'
    lp_chars = alphabets.alphabet

    device = torch.device('cuda')
    seg_net = SegNet(device)
    reg_net = CRNN(imgH=32, nc=3, nclass=len(lp_chars) + 1, nh=256).to(device)
    representer = SegDetectorRepresenter()
    converter = strLabelConverter(lp_chars)

    reg_net.load_state_dict(torch.load(reg_weight_path)['state_dict'])
    seg_net.load_state_dict(torch.load(seg_weight_path), strict=False)

    load_end = time.time()
    # print('加载模型耗时：%.3fs' % (load_end - load_start))

    operator = DemoOperator(seg_net=seg_net, reg_net=reg_net, representer=representer, converter=converter)
    operator.demoRun('real_imgs')