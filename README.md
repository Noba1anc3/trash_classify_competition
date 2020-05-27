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
['陶瓷器皿', 2662, '10.40%']  ['剩饭剩菜', 1415, '5.529%']  ['插头电线',　　1277, '4.990%']
['污损塑料', 1237, '4.834%']  ['果皮果肉', 1052, '4.111%']  ['洗护用品',　　1030, '4.025%']
['菜帮菜叶',  984, '3.845%']  ['玻璃器皿',  878, '3.431%']  ['筷子',　　　　 817, '3.192%']
['过期药物',  710, '2.774%']  ['塑料器皿',  666, '2.602%']  ['毛绒玩具',　　 655, '2.559%']
['纸盒纸箱',  623, '2.434%']  ['调料瓶',　  543, '2.122%']  ['锅', 　　　　　531, '2.075%']
['鱼骨', 　　 525, '2.051%']  ['砧板', 　　 466, '1.821%']  ['鞋',　　　　　 463, '1.809%']
['蛋壳', 　　 458, '1.789%']  ['食用油桶',　446, '1.742%']  ['干电池', 　　　445, '1.739%']
['包',　　　  442, '1.727%']  ['塑料玩具',  439, '1.715%']  ['烟蒂',　　　　 433, '1.692%']
['软膏',　　  418, '1.633%']  ['易拉罐', 　 414, '1.617%']  ['大骨头', 　　　413, '1.613%']
['充电宝',　　410, '1.602%']  ['茶叶渣', 　 409, '1.598%']  ['金属食品罐',　 400, '1.563%']
['旧衣服', 　 394, '1.539%']  ['枕头', 　　 382, '1.492%']  ['饮料瓶',　　　 371, '1.449%']
['酒瓶', 　　 346, '1.352%']  ['金属器皿',  341, '1.332%']  ['一次性快餐盒', 329, '1.285%']
['塑料衣架',  325, '1.270%']  ['金属厨具',　284, '1.109%']  ['快递纸袋',　　 248, '0.969%']
['污损用纸',  227, '0.887%']  ['花盆',　　  211, '0.824%']  ['书籍纸张', 　　206, '0.805%']
['牙签', 　　 156, '0.609%']  ['垃圾桶',  　108, '0.422%']

```
![](http://m.qpic.cn/psc?/fef49446-40e0-48c4-adcc-654c5015022c/90yfO.8bOadXEE4MiHsPn.FLsGecOSQpLRrJDJCMHGD3ARPTnSkqUVeeoa9sP6fkquzHgBVoWjTn5v8WxRRF4g!!/b&bo=ygN5AsoDeQIDCSw!&rf=viewer_4)

<!-- ![](https://s1.ax1x.com/2020/05/27/tEI6OO.png) -->

### Average Class Annotation Size

| ID | Name         | Size       | ID | Name      | Size       | ID | Name      | Size       | ID | Name      | Size       |
|:--:|:------------:|:----------:|:--:|:---------:|:----------:|:--:|:---------:|:----------:|:--:|:---------:|:----------:|
| 1  | 书籍纸张     | 1418859    | 2  | 金属厨具   | 886125     | 3  | 砧板      | 586395     | 4  | 快递纸袋   | 313754     |
| 5  | 垃圾桶       | 295087     | 6  | 包         | 287510     | 7  | 菜帮菜叶  | 260559     | 8  | 充电宝     | 256266     |
| 9  | 污损塑料     | 243065     | 10 | 塑料衣架   | 221071     | 11 | 塑料玩具  | 214609     | 12 | 枕头       | 213060     |
| 13 | 果皮果肉     | 206472     | 14 | 旧衣服     | 205153     | 15 | 锅        | 192131     | 16 | 大骨头     | 187760     |
| 17 | 塑料器皿     | 179405     | 18 | 调料瓶     | 176388     | 19 | 茶叶渣    | 175455     | 20 | 筷子       | 173675     |
| 21 | 一次性快餐盒 | 155888     | 22 | 纸盒纸箱   | 149912     | 23 | 污损用纸   | 149409    | 24 | 酒瓶       | 146719     |
| 25 | 毛绒玩具     | 144384     | 26 | 鞋         | 143123     | 27 | 洗护用品  | 141550     | 28 | 鱼骨       | 131911     |
| 29 | 花盆         | 129661    | 30 | 牙签       | 122917      | 31 | 饮料瓶    | 122624     | 32 | 金属器皿   | 111677     |
| 33 | 金属食品罐   | 104832    | 34 | 蛋壳       | 101802      | 35 | 陶瓷器皿   | 101420     | 36 | 食用油桶   | 100835    |
| 37 | 易拉罐       | 97242     | 38 | 玻璃器皿   | 89586       | 39 | 插头电线   | 87257      | 40 | 剩饭剩菜   | 76435     |
| 41 | 过期药物     | 70445     | 42 | 软膏       | 68024       | 43 | 干电池    | 46207      | 44 | 烟蒂       | 41917     |

![](http://m.qpic.cn/psc?/fef49446-40e0-48c4-adcc-654c5015022c/U9VSE8DftkGCrX.UXUSpm.QhT*rJAEav5HdWFjX9r2rgCRK7dstM0qeFUGzw9PB32BM5OqscGMhyWnSDy49uWXr*7jamOaH6Jlhc6gkSRes!/b&bo=yQN9AskDfQIDGTw!&rf=viewer_4)

<!-- ![](https://s1.ax1x.com/2020/05/27/tEI20e.md.png) -->

### Average Class Annotation Ratio

| ID | Name         | Ratio      | ID | Name      | Ratio      | ID | Name       | Ratio      | ID | Name      | Ratio      |
|:--:|:------------:|:----------:|:--:|:---------:|:----------:|:--:|:----------:|:----------:|:--:|:---------:|:----------:|
| 1  | 旧衣服       | 0.8224     | 2  | 包        | 0.750      | 3  | 快递纸袋    | 0.743      | 4  | 茶叶渣    |  0.727     |
| 5  | 砧板         | 0.715      | 6  | 大骨头    | 0.663      | 7  | 菜帮菜叶    | 0.654      | 8  | 枕头      | 0.648      |
| 9  | 果皮果肉     | 592        | 10 | 锅        | 0.588      | 11 | 纸盒纸箱    | 0.578      | 12 | 塑料衣架  | 0.576      |
| 13 | 毛绒玩具     | 0.568      | 14 | 金属食品罐 | 0.498     | 15 | 鱼骨        |  0.493     | 16 | 酒瓶      | 0.488      |
| 17 | 鞋           | 0.485      | 18 | 充电宝    | 0.460      | 19 | 塑料器皿    | 0.455      | 20 | 垃圾桶    | 0.453      |
| 21 | 食用油桶     |  0.450     | 22 | 过期药物  | 0.443      | 23 | 一次性快餐盒 | 0.437     | 24 | 调料瓶    | 0.432       |
| 25 | 塑料玩具     | 0.408      | 26 | 书籍纸张  |  0.408     | 27 | 蛋壳        | 0.401      | 28 | 干电池    | 0.392      |
| 29 | 花盆         | 0.365      | 30 | 污损塑料  | 0.352      | 31 | 金属器皿    | 0.346      | 32 | 易拉罐    | 0.340      |
| 33 | 软膏         | 0.325      | 34 | 玻璃器皿  |  0.318     | 35 | 洗护用品    | 0.318      | 36 | 陶瓷器皿  | 0.305      |
| 37 | 插头电线     | 0.303      | 38 | 饮料瓶    | 0.257      | 39 | 牙签        | 0.254      | 40 | 筷子      | 0.254      |
| 41 | 烟蒂         | 0.253      | 42 | 金属厨具  | 0.179      | 43 | 污损用纸    | 0.173      | 44 | 剩饭剩菜  | 0.156      |

![](http://m.qpic.cn/psc?/fef49446-40e0-48c4-adcc-654c5015022c/U9VSE8DftkGCrX.UXUSpmxT8HVkUBF5dhYLijGU08wy3wyChzT0q9LioduSIVt9jN3Z8oQL6S*ZFKdfTOWYezW3ULXC74FwYofuJsCsCisk!/b&bo=0QN9AtEDfQIDGTw!&rf=viewer_4)

<!-- ![](https://s1.ax1x.com/2020/05/27/tEIvhn.md.png) -->

### Average Class Annotation Size & Ratio
![](http://m.qpic.cn/psc?/fef49446-40e0-48c4-adcc-654c5015022c/U9VSE8DftkGCrX.UXUSpm0zxPbyPBJF.G0S1ZlnAZSDJ2vSV*AwP1Pdd9tsoNkSuElCZhWzK8BcDb6WCS2FtHlGOUtIEGMnQg86pDJVt4Ys!/b&bo=0gN8AtIDfAIDGTw!&rf=viewer_4)

<!-- ![](https://s1.ax1x.com/2020/05/27/tEom1x.md.png) --->

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
[Documentation](https://github.com/Noba1anc3/trash_classify_competition/tree/master/log_analysis/README.md)

## YoLo-v3 Evaluation Logs
[Documentation](https://github.com/Noba1anc3/trash_classify_competition/tree/master/PyTorch-YOLOv3/README.md)

