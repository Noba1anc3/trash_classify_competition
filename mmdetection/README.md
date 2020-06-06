## mmdetection

### Customized Task
- mmdet/datasets/voc.py
  - CLASSES
- mmdet/core/evaluation/class_names.py
  - voc_classes()
- configs/base/models/cascade_rcnn_r50_fpn.py
  - pretrained='torchvision://resnet50',
  - model['roi_head']['bbox_head']['num_classes']
- configs/base/schedules/schedule_*.py
- configs/base/datasets/vov0712.py
  - data_root 
  - data['samples_per_gpu']
  - data['train']['times']
  - data['train']['dataset']['ann_file']
  - data['train']['dataset']['img_prefix']

### To whl
As compile not permitted in the deploy phase on Huawei ModelArts Platform, we need to pack mmdetection to whl file

```
python setup.py install
python setup.py develop
python setup.py bdist_wheel
```

File: mmdet-2.0.0+unknown-cp36-cp36m-linux_x86_64.whl is packed on ModelArts Platform

### Requirements
- With whl: pip-requirements.txt
- Without whl (Online Develop): pip-requirements-without-whl.txt

### model
#### 44 cls
- [resnet50-19c8e357.pth](https://bhpan.buaa.edu.cn:443/link/D74B0212071B1C26482F1689B6294626)
- [resnet101-5d3b4d8f.pth](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)
