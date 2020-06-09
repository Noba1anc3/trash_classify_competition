import os
import xml.etree.ElementTree as ET

log_file = './log/log1.log'

with open(log_file, 'r') as f:
    log_file = f.readlines()


def takeSecond(elem):
    return elem[1]


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

    print(classes_num)


def box_num():


cls_num()
# box_num()

