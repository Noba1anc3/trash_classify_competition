# -*- coding=utf-8 -*-

# 包括:
#     1. 裁剪(需改变bbox)
#     2. 平移(需改变bbox)
#     3. 改变亮度
#     4. 加噪声
#     5. 旋转角度(需要改变bbox)
#     6. 镜像(需要改变bbox)
#     7. cutout
# 注意:
#     random.seed(),相同的seed,产生的随机数是一样的!!

import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure
from xml_helper import parse_xml


def show_pic(image, bboxes=None):
    """
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
    """

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)),
                      (0, 255, 0), 3)

    cv2.namedWindow('pic', 0)
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200, 800)  # 可视化的图片大小
    cv2.imshow('pic', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class DataAugmentForObjectDetection:
    def __init__(self, rotation_rate=0.5, max_rotation_angle=5,
                 crop_rate=0.5, shift_rate=0.5, change_light_rate=0.7,
                 add_noise_rate=0.6, flip_rate=0.6, cutout_rate=0.6,
                 cut_out_length=80, cut_out_holes=2, cut_out_threshold=0.5):

        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold

    @staticmethod
    def _addNoise(image):
        """
        输入:
            image:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        """

        return random_noise(image, mode='gaussian', clip=True)

    @staticmethod
    def _changeLight(image):
        flag = random.uniform(0.5, 1.5)  # flag>1为调暗,小于1为调亮
        return exposure.adjust_gamma(image, flag)

    @staticmethod
    def _cutout(image, bboxes, length=100, n_holes=1, threshold=0.5):
        """
        原版本：https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        Randomly mask out one or more patches from an image.
        Args:
            image : a 3D numpy array,(h,w,c)
            bboxes : 框的坐标
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        """

        length = int((image.shape[0] + image.shape[1])/15)

        def cal_iou(boxA, boxB):
            """
            boxA, boxB为两个框，返回iou
            boxB为bouding box
            """

            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union
            iou = interArea / float(boxAArea + boxBArea - interArea)

            # return the intersection over union value
            return iou

        # 得到h和w
        if image.ndim == 3:
            h, w, c = image.shape
        else:
            _, h, w, c = image.shape

        for n in range(n_holes):

            chongdie = True  # 看切割的区域是否与box重叠太多

            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0, h)
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                chongdie = False
                for box in bboxes:
                    if cal_iou([x1, y1, x2, y2], box) > threshold:
                        chongdie = True
                        break

            image[y1:y2, x1:x2, :] = 0.

        return image

    # 旋转
    @staticmethod
    def _rotate_img_bbox(image, bboxes, angle=5, scale=1.):
        """
        参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        输入:
            image:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        """
        # ---------------------- 旋转图像 ----------------------
        w = image.shape[1]
        h = image.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_image = cv2.warpAffine(image, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx + rw
            ry_max = ry + rh
            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

        return rot_image, rot_bboxes

    # 裁剪
    @staticmethod
    def _crop_img_bboxes(image, bboxes):
        """
        裁剪后的图片要包含所有的框
        输入:
            image:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        """

        # ---------------------- 裁剪图像 ----------------------
        w = image.shape[1]
        h = image.shape[0]

        x_min = w
        x_max = 0
        y_min = h
        y_max = 0

        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max  # 包含所有目标框的最小框到右边的距离
        d_to_top = y_min  # 包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max  # 包含所有目标框的最小框到底部的距离

        # 随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # ---------------------- 裁剪bbox ----------------------
        # 裁剪后的bbox坐标计算

        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min])

        return crop_img, crop_bboxes

    # 平移
    @staticmethod
    def _shift_pic_bboxes(image, bboxes):
        """
        参考:https://blog.csdn.net/sty945/article/details/79387054
        平移后的图片要包含所有的框
        输入:
            image:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            shift_img:平移后的图像array
            shift_bboxes:平移后的bounding box的坐标list
        """

        # ---------------------- 平移图像 ----------------------
        w = image.shape[1]
        h = image.shape[0]

        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0

        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最大左移动距离
        d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
        d_to_top = y_min  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left - 1) / 2, (d_to_right - 1) / 2)
        y = random.uniform(-(d_to_top - 1) / 2, (d_to_bottom - 1) / 2)

        M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # ---------------------- 平移boundingbox ----------------------
        shift_bboxes = list()

        for bbox in bboxes:
            shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y])

        return shift_img, shift_bboxes

    # 镜像
    @staticmethod
    def _filp_pic_bboxes(image, bboxes):
        """
            参考:https://blog.csdn.net/jningwei/article/details/78753607
            平移后的图片要包含所有的框
            输入:
                image:图像array
                bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            输出:
                flip_img:平移后的图像array
                flip_bboxes:平移后的bounding box的坐标list
        """
        # ---------------------- 翻转图像 ----------------------
        import copy

        flip_img = copy.deepcopy(image)

        if random.random() < 0.8:  # 0.8的概率水平翻转，0.2的概率垂直翻转
            horizon = True
        else:
            horizon = False

        h, w, _ = image.shape

        if horizon:  # 水平翻转
            flip_img = cv2.flip(flip_img, 1)  # 1是水平，-1是水平垂直
        else:
            flip_img = cv2.flip(flip_img, 0)

        # ---------------------- 调整boundingbox ----------------------
        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]

            if horizon:
                flip_bboxes.append([w - x_max, y_min, w - x_min, y_max])
            else:
                flip_bboxes.append([x_min, h - y_max, x_max, h - y_min])

        return flip_img, flip_bboxes

    def dataAugment(self, image, bboxes):
        """
        图像增强
        输入:
            image:图像array
            bboxes:该图像的所有框坐标
        输出:
            image:增强后的图像
            bboxes:增强后图片对应的box
        """

        change_num = 0  # 改变的次数
        noise = False

        while change_num < 1:  # 默认至少有一种数据增强生效
            if random.random() < self.crop_rate:  # 裁剪
                image, bboxes = self._crop_img_bboxes(image, bboxes)
                change_num += 1
                print('裁剪')

            if random.random() < self.rotation_rate:  # 旋转
                angle = random.sample([90, 180, 270], 1)[0]
                scale = random.uniform(0.7, 0.8)
                image, bboxes = self._rotate_img_bbox(image, bboxes, angle, scale)
                change_num += 1
                print('旋转')

            if random.random() < self.shift_rate:  # 平移
                image, bboxes = self._shift_pic_bboxes(image, bboxes)
                change_num += 1
                print('平移')

            if random.random() < self.change_light_rate:  # 改变亮度
                image = self._changeLight(image)
                change_num += 1
                print('亮度')

            if random.random() < self.add_noise_rate:  # 加噪声
                image = self._addNoise(image)
                change_num += 1
                print('加噪声')
                noise = True

            if random.random() < self.cutout_rate:  # cutout
                image = self._cutout(image, bboxes, length=self.cut_out_length,
                                     n_holes=self.cut_out_holes,
                                     threshold=self.cut_out_threshold)

                change_num += 1
                print('cutout')

            if random.random() < self.flip_rate:  # 翻转
                image, bboxes = self._filp_pic_bboxes(image, bboxes)
                change_num += 1
                print('翻转')

        return image, bboxes, noise


if __name__ == '__main__':
    from test import change_xml

    dataAug = DataAugmentForObjectDetection()

    pic_root_path = './train/VOC2007/JPEGImages'
    xml_root_path = './train/VOC2007/Annotations'

    count = 0
    overall = 0

    for parent, _, files in os.walk(pic_root_path):
        for file in files:
            overall += 1
            print('------')
            print(file)
            if random.random() < 0.5:
                count += 1
                pic_path = os.path.join(parent, file)
                xml_path = os.path.join(xml_root_path, file[:-4] + '.xml')

                coords = parse_xml(xml_path)
                coords = [coord[:4] for coord in coords]

                img = cv2.imread(pic_path)
                # show_pic(img, coords)

                auged_img, auged_bboxes, noise = dataAug.dataAugment(img, coords)

                height = auged_img.shape[0]
                width = auged_img.shape[1]

                imgname = './aug_train/VOC2007/JPEGImages/' + 'new_' + file
                xmlname = file[:-3] + 'xml'

                if noise:
                    cv2.imwrite(imgname, auged_img*255)
                else:
                    cv2.imwrite(imgname, auged_img)

                change_xml(xmlname, height, width, auged_bboxes)

                # show_pic(auged_img, auged_bboxes)

            else:
                pass

            print(count, overall)

    print(count)
