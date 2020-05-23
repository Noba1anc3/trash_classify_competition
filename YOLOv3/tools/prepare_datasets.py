# 将原始数据集的训练集和验证集分别组织成新的文件形式
# target_trainval_dir/
#     images/
#         0001.jpg
#         0002.jpg
#         0003.jpg
#     labels/
#         0001.txt
#         0002.txt
#         0003.txt
#
# 将trainval文件夹打包并命名为trainval.zip, 上传到OBS中以备使用。
#
# 生成的txt文件为同名图片的标注，文件中的每一行为一个bbox
# 格式为：  class_id x_center y_center width height   其中中心坐标和宽高都为小数形式

import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil


def generate_label_txt(ori_anno_dir, target_anno_dir, class_names):
    """
    # 生成txt标注文件
    """

    for xml_filename in tqdm(os.listdir(ori_anno_dir)):  # 对于每一个xml，生成对应的txt

        xml_path = os.path.join(ori_anno_dir, xml_filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        content = ''
        for obj in root.iter('object'):

            # 类别号
            class_id = obj.find('name').text
            class_id = class_names.index(class_id)

            # 坐标
            xmlbox = obj.find('bndbox')
            box = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                   int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)]
            x_center = round(((box[0] + box[2]) / 2) / w, 6)
            y_center = round(((box[1] + box[3]) / 2) / h, 6)
            width = round((box[2] - box[0]) / w, 6)
            height = round((box[3] - box[1]) / h, 6)

            # 即将写入文件的内容
            content += str(class_id) + ' ' + ' '.join(str(v) for v in [x_center, y_center, width, height]) + '\n'

            # 错误标注处理：坐标越界或宽高小于等于0
            if box[2] > w or box[3] > h:
                print('Image with annotation error:', xml_path)
            if box[0] < 0 or box[1] < 0:
                print('Image with annotation error:', xml_path)
            if box[2] <= box[0] or box[3] <= box[1]:
                print('Image with annotation error:', xml_path)

        # 写入对应txt文件
        txt_anno_path = os.path.join(target_anno_dir, xml_filename)
        with open(txt_anno_path, 'w') as f:
            f.writelines(content)


def creat_new_datasets(ori_dir, target_dir):

    ori_images_dir = os.path.join(ori_dir, 'VOC2007', 'JPEGImages')
    ori_anno_dir = os.path.join(ori_dir, 'VOC2007', 'Annotations')
    classes_txt_path = os.path.join(ori_dir, 'train_classes.txt')

    # 目标路径
    target_images_dir = os.path.join(target_dir, 'images')
    target_anno_dir = os.path.join(target_dir, 'labels')

    # 目标路径创建
    if not os.path.exists(target_anno_dir):
        os.makedirs(target_anno_dir)

    # 获取所有类别名
    with open(classes_txt_path, 'r', encoding='utf-8') as f:
        class_names = f.read().splitlines()

    # 拷贝图片
    print('copying images...')
    shutil.copytree(ori_images_dir, target_images_dir)
    print('images copied\n')

    # 生成txt标注文件
    print('generating train labels...')
    generate_label_txt(ori_anno_dir, target_anno_dir, class_names)
    print('train labels generated\n')


if __name__ == "__main__":
    ori_trainval_dir = '/home/j_m/Desktop/trainval_'
    target_trainval_dir = '/home/j_m/Desktop/trainval'

    creat_new_datasets(ori_trainval_dir, target_trainval_dir)
