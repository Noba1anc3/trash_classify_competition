## mmdetection

## Customized Task
- mmdet/core/evaluation/class_names.py : voc_classes()
- configs/_base_/schedules/schedule_*.py : total_epochs - 12                     # epochs
- configs/_base_/datasets/vov0712.py : classes ? 
- configs/_base_/datasets/vov0712.py : data_root 
- configs/_base_/datasets/voc0712.py : data['samples_per_gpu']                   # batchsize
- configs/_base_/datasets/vov0712.py : data['train']['times']
- configs/_base_/datasets/vov0712.py : data['train']['dataset']['ann_file']
- configs/_base_/datasets/vov0712.py : data['train']['dataset']['img_prefix']

## To whl
As compile not permitted in the deploy phase on Huawei ModelArts Platform, we need to pack mmdetection to whl file

```
python setup.py install
python setup.py develop
python setup.py bdist_wheel
```

File: mmdet-2.0.0+unknown-cp36-cp36m-linux_x86_64.whl is packed on ModelArts Platform

### model
- [cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth](https://bhpan.buaa.edu.cn:443/link/355651E5E64263BA8BBEA717A309B4AD)
