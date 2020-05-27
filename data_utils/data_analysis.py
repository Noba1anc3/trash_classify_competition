import os
import xml.etree.ElementTree as ET

classes = '../PyTorch-YOLOv3/cache/datasets/dataset/train_classes.txt'
anno_dir = '../PyTorch-YOLOv3/cache/datasets/dataset/VOC2007/Annotations/'

filelist = os.listdir(anno_dir)

with open(classes, 'r') as f:
    classes = f.readlines()

for i in range(len(classes)-1):
    classes[i] = classes[i][:-1]

classes_num = []
for i in range(len(classes)):
    classes_num.append([classes[i], 0, 0, 0])


def takeSecond(elem):
    return elem[1]


def takeThird(elem):
    return elem[2]


def takeFourth(elem):
    return elem[3]


def cls_num():
    for filename in filelist:

        xml_path = anno_dir + filename

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            bndbox_name = obj.find('name').text
            class_index = classes.index(bndbox_name)
            classes_num[class_index][1] += 1

    summ = 0
    for i in range(len(classes_num)):
        summ += classes_num[i][1]
    for i in range(len(classes_num)):
        classes_num[i][2] = str((classes_num[i][1]/summ)*100)[:5] + '%'
    classes_num.sort(key=takeSecond, reverse=True)

    for item in classes_num:
        print(item[0] + ' : ' + str(item[1]))


def cls_area():
    for filename in filelist:

        xml_path = anno_dir + filename

        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        size = float(size.find('width').text) * float(size.find('height').text)

        for obj in root.findall('object'):
            bndbox_name = obj.find('name').text

            bndbox = obj.find('bndbox')
            bbox_width = float(bndbox.find('xmax').text) - float(bndbox.find('xmin').text)
            bbox_height = float(bndbox.find('ymax').text) - float(bndbox.find('ymin').text)

            bbox_size = bbox_height*bbox_width
            box_img_ratio = bbox_size/size

            class_index = classes.index(bndbox_name)
            classes_num[class_index][1] += 1
            classes_num[class_index][2] += bbox_size
            classes_num[class_index][3] += box_img_ratio

    for i in range(len(classes_num)):
        classes_num[i][2] /= classes_num[i][1]
        classes_num[i][3] /= classes_num[i][1]

    classes_num.sort(key=takeThird, reverse=True)
    for i in range(len(classes_num)):
        print(classes_num[i][0] + ' : ' + str(classes_num[i][2]))

    print('\n')

    classes_num.sort(key=takeFourth, reverse=True)
    for i in range(len(classes_num)):
        print(classes_num[i][0] + ' : ' + str(classes_num[i][3]))


def box_num():
    box_in_img = []
    for i in range(80):  # 最多79个框
        box_in_img.append([i, 0, 0])

    for filename in filelist:

        xml_path = anno_dir + filename

        tree = ET.parse(xml_path)
        root = tree.getroot()

        objs = len(root.findall('object'))
        box_in_img[objs][1] += 1

    summ = 0
    for i in range(len(box_in_img)):
        summ += box_in_img[i][1]
    for i in range(len(box_in_img)):
        box_in_img[i][2] = str((box_in_img[i][1]/summ)*100)[:5] + '%'
    box_in_img.sort(key=takeSecond, reverse=True)
    print(box_in_img)


cls_num()
# box_num()
# cls_area()
