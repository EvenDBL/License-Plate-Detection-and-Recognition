import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper

class SegDetectorRepresenter:

    def __init__(self, is_poly=False):
        self.threshold = 0.3
        self.max_candidates = 1000
        self.is_poly = is_poly
        self.min_size = 3
        self.box_thresh = 0.6

    def represent(self, batch, _pred):

        images = batch['image']
        pred = _pred
        scale_b = batch['resize_scale']

        segmentation = self.binarize(pred)

        boxes_batch = []
        scores_batch = []
        for batch_index in range(images.shape[0]):
            height, width = batch['original_size'][batch_index]
            scale = scale_b[batch_index]
            if self.is_poly:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index], width, height, scale)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height, scale)
            boxes_batch.append(boxes)
            scores_batch.append(scores)

        return boxes_batch, scores_batch

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height, scale):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''
        assert _bitmap.size(0) == 1
        bitmap = _bitmap.cpu().numpy()[0]  # The first channel
        pred = pred.cpu().detach().numpy()[0]

        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.01 * cv2.arcLength(contour, True)    # 求轮廓周长，
            approx = cv2.approxPolyDP(contour, epsilon, True)   # 求轮廓的近似值，epsilon为阈值，True表示轮廓闭合
            points = approx.reshape((-1, 2))

            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, points)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, unclip_ratio=1.0)
            if len(box) > 1:
                continue

            box = box.reshape(-1, 2)    # 多边形轮廓点
            if box.shape[0] < 4:
                continue

            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            # if sside < self.min_size + 2:
            #     continue

            box[:, 0] = np.clip(np.round(box[:, 0] / scale), 0, dest_width)  # 将目标的检测框位置，还原到原图位置
            box[:, 1] = np.clip(np.round(box[:, 1] / scale), 0, dest_height)
            boxes.append(box.astype('int'))
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height, scale):

        assert _bitmap.size(0) == 1
        bitmap = _bitmap.cpu().numpy()[0]  # The first channel
        pred = pred.cpu().detach().numpy()[0]

        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        # boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        # scores = np.zeros((num_contours,), dtype=np.float32)

        boxes_list = []
        scores_list = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            # box = points
            box, sside = self.get_mini_boxes(box)
            # if sside < self.min_size + 2:
            #     continue
            box = np.array(box)     # shape = [4, 2]

            box[:, 0] = np.clip(np.round(box[:, 0] / scale), 0, dest_width)          # 将目标的检测框位置，还原到原图位置
            box[:, 1] = np.clip(np.round(box[:, 1] / scale), 0, dest_height)
            # boxes[index, :, :] = box.astype('int')
            # scores[index] = score
            boxes_list.append(box.astype('int'))
            scores_list.append(score)

        boxes_array = np.array(boxes_list).reshape((-1, 4, 2))
        scores_array = np.array(scores_list)
        return boxes_array, scores_array

    def box_score_fast(self, pre, _box):
        h, w = pre.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(pre[ymin:ymax+1, xmin:xmax+1], mask)[0]

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def binarize(self, pred):
        return pred > self.threshold