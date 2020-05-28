
size_map = []

with open('../map.log', 'r') as f:
    train = f.readlines()

with open('../cls_anno_ratio.md', 'r') as f:
    annosize = f.readlines()

with open('../cls_anno_size.md', 'r') as f:
    annoratio = f.readlines()

for index in range(len(annosize)):
    annoitem = annosize[index]
    anno_cls = annoitem.split(':')[0].strip()
    anno_size = annoitem.split(':')[1].replace('\n', '')

    size_map.append([anno_cls, anno_size, 0])


for index in range(len(train)):
    trainitem = train[index]
    item_cls = trainitem.split("|")[2].strip()
    item_map = trainitem.split("|")[3].strip()

    for index in range(len(size_map)):
        if size_map[index][0] == item_cls:
            size_map[index][2] = item_map

for item in size_map:
    print(item)
