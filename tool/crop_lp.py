from PIL import Image
import os
import csv
import numpy as np

class CropLp():
    def __init__(self, original_path, save_path):
        self.original_path = original_path
        self.save_path = save_path

    def cropAndSave(self):
        images_name = os.listdir(self.original_path)
        i = 0
        total = len(images_name)
        for image_name in images_name:
            image_path = os.path.join(self.original_path, image_name)

            twoPoints = image_name.split('-')[2].split('_')
            leftUp = twoPoints[0].split('&')
            rightDown = twoPoints[1].split('&')

            image = Image.open(image_path)
            lp_area = image.crop((int(leftUp[0]), int(leftUp[1]), int(rightDown[0]), int(rightDown[1])))

            lp_save_path = os.path.join(self.save_path, image_name)
            lp_area.save(lp_save_path)

            i += 1
            print('[%d/%d]'%(i, total))

    def cropAndSaveClpd(self, csv_path):

        # 读取csv文件
        f = csv.reader(open(csv_path, 'r', encoding='gbk'))
        for line in f:
            # 获取文件名
            file_name = line[0]

            if file_name=='path':
                continue
            file_name = line[0].split('/')[1]
            points_loc = line[1:-1]
            points_loc_array = np.array(points_loc).astype('int')

            # 获取左上、右下坐标
            poly = points_loc_array.reshape(-1, 2)
            box = [[min(poly[:, 0]), min(poly[:, 1])], [max(poly[:, 0]), max(poly[:, 1])]]

            # 读取图片，并截取区域
            image = Image.open(os.path.join(self.original_path, file_name))
            lp_area = image.crop((box[0][0], box[0][1], box[1][0], box[1][1]))

            lp_save_path = os.path.join(self.save_path, file_name)
            lp_area.save(lp_save_path)

    def cropAndSaveAolpAc(self, txt_path):
        file_name_list = os.listdir(txt_path)
        for file_name in file_name_list:
            file = open(os.path.join(txt_path, file_name))
            point_loc = []
            for line in file.readlines():
                line = line.strip('\n')
                point_loc.append(int(float(line)))

            img_name = file_name.split('.')[0]+'.jpg'
            if '_' in img_name:
                continue
            # 读取图片，并截取区域
            image = Image.open(os.path.join(self.original_path, img_name))

            if point_loc[0]<point_loc[2]:
                x1 = point_loc[0]
                x2= point_loc[2]
            else:
                x1 = point_loc[2]
                x2 = point_loc[0]
            if point_loc[1]<point_loc[3]:
                y1 = point_loc[1]
                y2 = point_loc[3]
            else:
                y1 = point_loc[3]
                y2 = point_loc[1]

            # lp_area = image.crop((point_loc[0], point_loc[1], point_loc[2], point_loc[3]))
            lp_area = image.crop((x1, y1, x2, y2))

            lp_save_path = os.path.join(self.save_path, img_name)
            lp_area.save(lp_save_path)

if __name__=='__main__':
    cropper = CropLp('/home/admin1/datasets/AOLP/Subset_LE/Image', '/home/admin1/datasets/AOLP/Subset_LE/lp_area')
    cropper.cropAndSaveAolpAc('/home/admin1/datasets/AOLP/Subset_LE/groundtruth_localization')