## 2020 Huawei Cloud Cup Shenzhen Open Data Appliance Innovation Competition
[Competition Link](https://competition.huaweicloud.com/information/1000038439/introduction)

## Dataset
[Download](https://bhpan.buaa.edu.cn:443/link/9C3EC040AFD877740AA818AD4918038E)

## Data Cleansing
- Delete data which annotation bbox's location out of range
- Delete data which annotation bbox's xmin < xmax, or ymin < ymax
- Delete data which without annotation bbox

## Dataset Split
Training Set : Testing Set = 9 : 1

- Origin Training Set: 13468 Images
- Origin Testing Set: 1496 Images

After artificial check, 25 images of testing set is deleted.

- Training Set: 13468 Images
- Testing Set: 1471 Images

## Data Numerical Characteristics
### Classes
```
['陶瓷器皿', 2662, '10.40%'],  ['剩饭剩菜',　　1415, '5.529%']
['插头电线', 1277, '4.990%'],  ['污损塑料',　　1237, '4.834%']
['果皮果肉', 1052, '4.111%'],  ['洗护用品',　　1030, '4.025%']
['菜帮菜叶',  984, '3.845%'],  ['玻璃器皿', 　　878, '3.431%']
['筷子', 　　 817, '3.192%'],  ['过期药物', 　　710, '2.774%']
['塑料器皿',  666, '2.602%'],  ['毛绒玩具', 　　655, '2.559%']
['纸盒纸箱',  623, '2.434%'],  ['调料瓶',　　 　543, '2.122%']
['锅',　　　  531, '2.075%'],  ['鱼骨', 　　　　525, '2.051%']
['砧板', 　　 466, '1.821%'],  ['鞋',　　　 　　463, '1.809%']
['蛋壳', 　　 458, '1.789%'],  ['食用油桶',　　 446, '1.742%']
['干电池', 　 445, '1.739%'],  ['包',　　　　　 442, '1.727%']
['塑料玩具',  439, '1.715%'],  ['烟蒂',　　　　 433, '1.692%']
['软膏',　　  418, '1.633%'],  ['易拉罐',　　　 414, '1.617%']
['大骨头', 　 413, '1.613%'],  ['充电宝',　　 　410, '1.602%']
['茶叶渣', 　 409, '1.598%'],  ['金属食品罐',　 400, '1.563%']
['旧衣服', 　 394, '1.539%'],  ['枕头', 　　　　382, '1.492%']
['饮料瓶',　  371, '1.449%'],  ['酒瓶', 　　　　346, '1.352%']
['金属器皿',  341, '1.332%'],  ['一次性快餐盒', 329, '1.285%']
['塑料衣架',  325, '1.270%'],  ['金属厨具', 　　284, '1.109%']
['快递纸袋',  248, '0.969%'],  ['污损用纸', 　　227, '0.887%']
['花盆',　　  211, '0.824%'],  ['书籍纸张', 　　206, '0.805%']
['牙签', 　　 156, '0.609%'],  ['垃圾桶', 　　　108, '0.422%']

```

### Bbox in Image

```
[1, 10356, '69.20%'],  [2, 2786, '18.61%']
[3,   939, '6.275%'],  [4,  355, '2.372%']
[5,   199, '1.329%'],  [6,   81, '0.541%']
[7,    48, '0.320%'],  [8,   30, '0.200%']
[9,    24, '0.160%'],  [12,  15, '0.100%']
                ... ,  [79,   1, '0.006%']
```

## Data Visual Characteristics
- Occlusion
- Part of thing in the image
- Image is a part of thing
- Low resolution
- Class unbalanced

## Data Anchors Calculation
clusters:
```
 [0.21759225 0.47786703]
 [0.36999975 0.25625028]
 [0.52727695 0.537109  ]
 [0.74499975 0.9161846 ]
 [0.789062   0.35      ]
 [0.957031   0.65478175]
 [0.364      0.85      ]
 [0.094727   0.073242  ]
 [0.177474   0.166992  ]
```

clusters * 416:
```
 [ 91. 199.]
 [154. 107.]
 [219. 223.]
 [310. 381.]
 [328. 146.]
 [398. 272.]
 [151. 354.]
 [ 39.  30.]
 [ 74.  69.]
```

Average IoU: 71.90%  
Average IoU with VOC clusters: 64.36%

Ratios:  
 [0.43, 0.46, 0.81, 0.98, 1.06, 1.29, 1.44, 1.46, 2.25]

## Data Augmentation
- Augmentate Ratio : 0.5
- Crop Ratio : 0.5
- Shift Ratio : 0.5
- Rotate Ratio : 0.5
- Flip Ratio : 0.6
    - Horizontal Flip Ratio : 0.8
    - Vertical Flip Ratio : 0.2
- Change Light Ratio : 0.7
    - Range : [0.5, 1.5]
- Add noise Ratio : 0.6
    - Mode : Gaussian
- Cutout Ratio : 0.6
    - Cutout Size : (ImgHeight + ImgWidth) / 15
    - Cutout Hole : 2
    - Cutout IoU Threshold : 0.5

## Train Yolo-v3 Model
[Documentation](https://github.com/Noba1anc3/trash_classify_competition/tree/master/PyTorch-YOLOv3/README.md)
