train_dataset = './data/VOC2007/ImageSets/Main/val.txt'

pic_root_path = './data/VOC2007/JPEGImages/'
xml_root_path = './data/VOC2007/Annotations/'

newpic_root_path = './val/VOC2007/JPEGImages/'
newxml_root_path = './val/VOC2007/Annotations/'

import os
from shutil import copyfile

with open(train_dataset, 'r') as f:
    train_dataset = f.readlines()

for index in range(len(train_dataset)-1):
    train_dataset[index] = train_dataset[index][:-1]

count = 0
for parent, _, files in os.walk(pic_root_path):
    for file in files:
        if file[:-4] in train_dataset:
            count += 1
            img = pic_root_path + file
            xml = xml_root_path + file[:-4] + '.xml'
            newimg = newpic_root_path + file
            newxml = newxml_root_path + file[:-4] + '.xml'
            copyfile(img, newimg)
            copyfile(xml, newxml)
            print(count)
