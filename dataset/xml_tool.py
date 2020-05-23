import xml.etree.ElementTree as ET

old_anno_dir = './train/VOC2007/Annotations/'
new_anno_dir = './aug_train/VOC2007/Annotations/'


# 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def parse_xml(xml_path):
    """
    输入：
        xml_path: xml的文件路径
    输出：
        从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()

    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(box[0].text)
        y_min = int(box[1].text)
        x_max = int(box[2].text)
        y_max = int(box[3].text)
        coords.append([x_min, y_min, x_max, y_max, name])

    return coords


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

