
size_map = []

with open('./a.md', 'r') as f:
    train = f.readlines()

with open('./cls_num.md', 'r') as f:
    annonum = f.readlines()

with open('./cls_anno_ratio.md', 'r') as f:
    annoratio = f.readlines()

with open('./cls_anno_size.md', 'r') as f:
    annosize = f.readlines()

for index in range(len(annoratio)):
    annoitem = annoratio[index]
    anno_cls = annoitem.split(':')[0].strip()
    anno_size = annoitem.split(':')[1].replace('\n', '')

    size_map.append([anno_cls, anno_size, 0])


for index in range(len(train)):
    trainitem = train[index]
    item_cls = trainitem.split("|")[0].strip()
    item_map = trainitem.split("|")[1].replace('\n', '')

    for index in range(len(size_map)):
        if size_map[index][0] == item_cls:
            size_map[index][2] = item_map

print(size_map)
