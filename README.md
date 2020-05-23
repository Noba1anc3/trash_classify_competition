## 2020 Huaweiyun Cup Shenzhen Open Data Appliance Innovation Competition
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

## Data Character
- Occlusion
- Part of thing in the image
- Image is a part of thing
- Low resolution
- Class unbalanced

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
