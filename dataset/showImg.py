import os
import cv2
import xml.etree.ElementTree as ET

txt_path = './1.txt'
anno_dir = './data/VOC2007/Annotations'
img_dir = './data/VOC2007/JPEGImages'

resolution = [1920, 1080]
resolution = [resolution[0]*0.6, resolution[1]*0.6]

with open(txt_path, 'r') as f:
    all_filenames = f.read().splitlines()

for filename in all_filenames:

    info = filename

    xml_path = os.path.join(anno_dir, filename + '.xml')

    tree = ET.parse(xml_path)
    root = tree.getroot()

    bboxes = []

    for obj in root.findall('object'):
        bndbox_name = obj.find('name').text
        info += ' ' + bndbox_name

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        bboxes.append([xmin, ymin, xmax, ymax, bndbox_name])

    img_path = os.path.join(img_dir, filename + '.jpg')
    img = cv2.imread(img_path)

    height, width = img.shape[0], img.shape[1]
    ratio = [width / resolution[0], height / resolution[1]]
    ratio = max(ratio[0], ratio[1])
    if ratio > 1:
        width = int(width / ratio)
        height = int(height / ratio)

    print(info)

    cv2.namedWindow('anno', 0)
    cv2.resizeWindow('anno', width, height)
    for idx, bbox in enumerate(bboxes):
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
        cv2.putText(img, str(idx), (bbox[0]+5, bbox[3]-5), cv2.FONT_HERSHEY_SIMPLEX, 1
                    , (0, 0, 255), thickness=2)

    cv2.imshow('anno', img)
    cv2.waitKey()
