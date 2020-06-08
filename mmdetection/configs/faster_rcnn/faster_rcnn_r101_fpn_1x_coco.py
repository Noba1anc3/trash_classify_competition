_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
model = dict(pretrained='/cache/model/resnet101-5d3b4d8f.pth', backbone=dict(depth=101))
