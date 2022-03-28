import random
import numpy as np
import torch
from collections import OrderedDict
import imgaug
import imgaug.augmenters as iaa
import cv2
import math
from shapely.geometry import Polygon
import pyclipper

char2index_dict = { "a" : 0, "b" : 1, "c" : 2,"d" : 3,"e" : 4,"f" : 5,"g" : 6,"h" : 7,
                    "j" : 8,"k" : 9,"l" : 10,"m" : 11,"n" : 12,"p" : 13,"q" : 14,"r" : 15,"s" : 16,
                    "t" : 17,"u" : 18,"v" : 19,"w" : 20,"x":  21,"y" : 22,"z" : 23,"0" : 24,"1" : 25,
                    "2" : 26,"3" : 27,"4" : 28,"5" : 29,"6" : 30,"7" : 31,"8" : 32,"9" : 33,
                    "皖": 34,"沪": 35,"津": 36,"渝": 37,"冀": 38,"晋": 39,"蒙": 40,
                    "辽": 41, "吉": 42,"黑": 43,"苏": 44,"浙": 45,"京": 46,"闽": 47,
                    "赣": 48,"鲁": 49,"豫": 50,"鄂": 51,"湘": 52,"粤": 53,"桂": 54,
                    "琼": 55,"川": 56,"贵": 57,"云": 58,"西": 59,"陕": 60,"甘": 61,
                    "青": 62,"宁": 63,"新": 64
}

index2char_dict = { 0:'a',1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'j', 9:'k',
                    10:'l', 11:'m', 12:'n', 13:'p', 14:'q', 15:'r', 16:'s', 17:'t', 18:'u',
                    19:'v', 20:'w', 21:'x', 22:'y', 23:'z', 24:'0', 25:'1', 26:'2', 27:'3',
                    28:'4', 29:'5', 30:'6', 31:'7', 32:'8', 33:'9',
                    34:"皖",35:"沪",36:"津",37:"渝",38:"冀",39:"晋",40:"蒙",
                    41:"辽", 42:"吉",43:"黑",44:"苏",45:"浙",46:"京",47:"闽",
                    48:"赣",49:"鲁",50:"豫",51:"鄂",52:"湘",53:"粤",54:"桂",
                    55:"琼",56:"川",57:"贵",58:"云",59:"西",60:"陕",61:"甘",
                    62:"青",63:"宁",64:"新"
}

class CCPDDataset():

    def __init__(self, data_dir, data_list, use_argument=True):
        self.size = (640, 640)
        self.data_dir = data_dir
        self.data_list = data_list
        self.use_argument = use_argument
        self.image_paths = []
        self.gt_paths = []
        self.get_all_samples()

        # self.processes = [AugmentDetectionData(), RandomCropData(), ShrinkGt(), NormalizeImage()] # ExposureImage(),
        self.processes = [ ShrinkGt(), NormalizeImage()]#

    def get_all_samples(self):

        if 'CCPD' in self.data_list:
            with open(self.data_list, 'r') as fid:
                image_list = fid.readlines()
            self.image_names = [image_name.strip('\n') for image_name in image_list]
            self.image_paths = [self.data_dir+'/'+ image_name for image_name in self.image_names]
        elif 'CLPD' in self.data_list:
            with open(self.data_list, 'r') as fid:
                rows = fid.readlines()
            self.image_names = [row.strip('\n').split(' ')[0].split('/')[1] for row in rows]
            self.image_paths = [self.data_dir+'/CLPD_1200/' + image_name for image_name in self.image_names]
        elif 'AOLP' in self.data_list:
            with open(self.data_list, 'r') as fid:
                rows = fid.readlines()
            self.image_names = [row.strip('\n').split(' ')[0] for row in rows]
            self.image_paths = [self.data_dir+'/' + image_name for image_name in self.image_names]
        elif 'ALPR' in self.data_list:
            with open(self.data_list, 'r') as fid:
                rows = fid.readlines()
            self.image_names = [row.strip('\n').split(' ')[0] for row in rows]
            self.image_paths = self.image_names
        self.num_samples = len(self.image_paths)
        self.targets = self.load_ann()

        assert len(self.image_paths) == len(self.targets)

    def get_lp_str(self, image_name):
        index_lp = image_name.split('-')[4].split('_')[1:]
        province = index2char_dict[int(index_lp[0]) + 34]
        char_lp_list = []
        for index in index_lp:
            char_lp_list.append(index2char_dict[int(index)])
        char_lp = ''.join(char_lp_list)
        lp_str = province + char_lp
        return lp_str

    def get_4locations_as_np(self, image_name):
        fourPoints = image_name.split('-')[3].split('_')
        locations = []
        for point in fourPoints:
            locations.append(point.split('&')[0])
            locations.append(point.split('&')[1])
        points = np.array(list(map(float, locations))).reshape((-1, 2)).tolist()
        return points

    def load_ann(self):
        res = []
        if 'CCPD' in self.data_list:
            for image_name in self.image_names:
                lines = []
                item = {}
                poly = self.get_4locations_as_np(image_name)
                lp_str = self.get_lp_str(image_name)
                item['poly'] = poly
                item['text'] = lp_str
                lines.append(item)
                res.append(lines)
        elif 'CLPD' in self.data_list or 'AOLP' in self.data_list or 'ALPR' in self.data_list :
            with open(self.data_list, 'r') as fid:
                rows = fid.readlines()
            loc4points_total = [row.strip('\n').strip().split(' ')[1:-1] for row in rows]
            texts = [row.strip('\n').split(' ')[-1] for row in rows]
            assert len(texts)==len(loc4points_total)
            for i in range(len(texts)):
                loc4points = loc4points_total[i]
                lines = []
                item = {}
                poly = np.array(loc4points).astype('float')
                poly = poly.reshape(-1, 2)
                poly = poly.tolist()
                item['poly'] = poly
                item['text'] = texts[i]
                lines.append(item)
                res.append(lines)
        return res

    def resize_img(self, img, size):
        h, w = size
        original_h, original_w, c = img.shape
        img_new = np.zeros((h, w, c), img.dtype)
        scale_h = h / original_h
        scale_w = w / original_w
        scale = min(scale_h, scale_w)
        new_h = int(original_h * scale)
        new_w = int(original_w * scale)
        img_new[:new_h, :new_w] = cv2.resize(img, (new_w, new_h))

        return img_new, scale

    def gen_gt(self, targets, scale):

        h, w = self.size
        gt = np.zeros((h, w, 1))

        for item in targets:
            poly = item['poly']  # [[],[],[],[]]
            poly_array = (np.array(poly) * scale).astype('int')
            cv2.fillPoly(gt, [poly_array], (1))

        gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).long()
        return gt_tensor

    def __getitem__(self, index):
        if index >= self.num_samples:
            raise StopIteration

        data = {}

        image_path = self.image_paths[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        h, w, _ = img.shape
        original_size = [h, w]
        img_new, scale = self.resize_img(img, self.size)
        resize_scale = scale
        filename = image_path
        one_image_target = self.targets[index]
        one_image_gt = self.gen_gt(one_image_target, scale)

        image = img_new
        lines = one_image_target
        gt = one_image_gt

        data['image'] = image
        data['gt'] = gt
        data['lines'] = lines
        data['filename'] = filename
        data['resize_scale'] = resize_scale
        data['original_size'] = original_size

        # 数据增强
        data['image_ori'] = img
        if self.use_argument:
            for data_process in self.processes:
                data = data_process(data)
        else:
            data = ShrinkGt()(data)
            data = NormalizeImage()(data)
        return data

    def __len__(self):
        return len(self.image_paths)

class RandomCropData():

    def __init__(self):
        self.size = [640, 640]
        self.max_tries = 10
        self.min_crop_side_ratio = 0.1

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        img = data['image']

        all_care_polys = [line['poly'] for line in data['polys_augment']]
        crop_x, crop_y, crop_w, crop_h = self.crop_area(img, all_care_polys)
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        padimg = np.zeros((self.size[1], self.size[0], img.shape[2]), img.dtype)
        padimg[:h, :w] = cv2.resize(img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h)) # padimg[:h+1, :w+1] = cv2.resize(img[crop_y:crop_y + crop_h+1, crop_x:crop_x + crop_w+1], (w, h)) 待尝试
        img = padimg

        lines = []
        gt = np.zeros((self.size[0], self.size[1], 1))

        for line in data['polys_augment']:
            poly = ((np.array(line['poly']) -(crop_x, crop_y)) * scale).tolist()
            lines.append({**line, 'poly': poly})
            poly_array = np.array(poly).astype('int')
            cv2.fillPoly(gt, [poly_array], (1))
        gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).long()
        data['polys_augment'] = lines
        data['gt'] = gt_tensor
        data['image'] = img
        return data

    def is_poly_in_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def is_poly_outside_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i-1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, max_size):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, img, polys):
        h, w, _ = img.shape
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h

class NormalizeImage():
    # RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])

    def __call__(self, data):
        data['image'] = self.process(data['image'])
        return data

    @classmethod
    def process(self, data):
        image = data
        # image -= self.RGB_MEAN
        # image /= 255.
        image = (image / 255. - 0.588) / 0.193
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    @classmethod
    def restore(self, image):
        image = image.permute(1, 2, 0).to('cpu').numpy()
        # image = image * 255.
        # image += self.RGB_MEAN
        image = (image * 0.193+0.588)*255
        image = image.astype(np.uint8)
        image = image[:, :, ::-1]  # BGR2RGB
        return image

class ShrinkGt():
    def __init__(self):
        self.shrink_ratio = 0.4

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        old_gt = data['gt']

        _, h, w = old_gt.shape
        new_gt = np.zeros((h, w, 1))
        if 'polys_augment' in data:
            for line in data['polys_augment']:   #经增强之后的坐标信息，即旋转、缩放、随机裁剪操作，
                poly = np.array(line['poly']).astype('int')
        else:
            for line in data['lines']:     # 没有经过resize的坐标信息，即最原始的坐标信息
                scale = data['resize_scale']
                poly = (np.array(line['poly']) * scale).astype('int')    # 与缩放因子resize_scale相乘，把坐标信息转换到resize后的图像上

        polygon_shape = Polygon(poly)
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked = padding.Execute(-distance)

        shrinked = np.array(shrinked[0]).reshape(-1, 2)
        cv2.fillPoly(new_gt, [shrinked.astype('int')], (1))

        gt_tensor = torch.from_numpy(new_gt).permute(2, 0, 1).long()
        data['gt'] = gt_tensor

        return data

class ExposureImage():

    def __init__(self):
        pass

    def __call__(self, data):
        data['image'] = self.process(data['image'])
        return data

    def process(self, image_ori):

        image = cv2.cvtColor(image_ori , cv2.COLOR_BGR2HSV)
        gamma = random.uniform(0, 2)
        image[:, :, 2] = gamma * image[:, :, 2]
        image[:, :, 2][image[:, :, 2] > 255] = 255 # 调整亮度

        gamma = random.uniform(0, 2)
        image[:, :, 1] = gamma * image[:, :, 1]
        image[:, :, 1][image[:, :, 1] > 255] = 255 # 调整对比度
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image

class AugmentDetectionData():

    def __init__(self):
        self.augmenter_args =  [['Fliplr', 0.5],
                                {'cls': 'Affine', 'rotate': [-10, 10]},
                                ['Resize', [0.5, 3.0]]]
        self.keep_ratio = False
        self.only_resize = False
        self.augmenter = self.build(self.augmenter_args)

    def __call__(self, data):
        return self.process(data)

    def resize_image(self, image):
        origin_height, origin_width, _ = image.shape
        resize_shape = self.augmenter_args[0][1]
        height = resize_shape['height']
        width = resize_shape['width']
        if self.keep_ratio:
            width = origin_width * height / origin_height
            N = math.ceil(width / 32)
            width = N * 32
        image = cv2.resize(image, (width, height))
        return image

    def process(self, data):
        image = data['image_ori']
        shape = image.shape

        aug = self.augmenter.to_deterministic()
        if self.only_resize:
            data['image'] = self.resize_image(image)
        else:
            data['image'] = aug.augment_image(image)
        self.may_augment_annotation(aug, data, shape)
        data.update(shape=shape[:2])
        return data

    def may_augment_annotation(self, aug, data, shape):
        line_polys = []
        for line in data['lines']:
            if self.only_resize:
                new_poly = [(p[0], p[1]) for p in line['poly']]
            else:
                new_poly = self.may_augment_poly(aug, shape, line['poly'])
            line_polys.append({
                'poly': new_poly,
                'text': line['text'],
            })
        data['polys_augment'] = line_polys
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly

    def build(self, args, root=True):
        if args is None:
            return None
        elif isinstance(args, (int, float, str)):
            return args
        elif isinstance(args, list):
            if root:
                sequence = [self.build(value, root=False) for value in args]
                return iaa.Sequential(sequence)
            else:
                return getattr(iaa, args[0])(*[self.to_tuple_if_list(a) for a in args[1:]])
        elif isinstance(args, dict):
            if 'cls' in args:
                cls = getattr(iaa, args['cls'])
                return cls(**{k: self.to_tuple_if_list(v) for k, v in args.items() if not k == 'cls'})
            else:
                return {key: self.build(value, root=False) for key, value in args.items()}
        else:
            raise RuntimeError('unknown augmenter arg: ' + str(args))

    def to_tuple_if_list(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj

class CollectFN():
    def __init__(self):
        pass

    def __call__(self, batch):
        data_dict = OrderedDict()
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                data_dict[k].append(v)
        data_dict['image'] = torch.stack(data_dict['image'], 0)
        data_dict['gt'] = torch.stack(data_dict['gt'], 0)
        return data_dict