import xml.etree.ElementTree as ET

old_anno_dir = './train/VOC2007/Annotations/'
new_anno_dir = './aug_train/VOC2007/Annotations/'


def change_xml(filename, height, width, box):
    updateTree = ET.parse(old_anno_dir + filename)
    root = updateTree.getroot()

    Filename = root.find("filename")
    Filename.text = 'new_' + Filename.text

    Height = root.find("size").find("height")
    Width = root.find("size").find("width")

    Height.text = str(height)
    Width.text = str(width)

    for index in range(len(root.findall('object'))):
        obj = root.findall('object')[index]

        bbox = obj.find('bndbox')
        xmin = bbox.find('xmin')
        ymin = bbox.find('ymin')
        xmax = bbox.find('xmax')
        ymax = bbox.find('ymax')

        xmin.text = str(int(box[index][0]))
        ymin.text = str(int(box[index][1]))
        xmax.text = str(int(box[index][2]))
        ymax.text = str(int(box[index][3]))

    updateTree.write(new_anno_dir + 'new_' + filename, encoding="utf-8")

