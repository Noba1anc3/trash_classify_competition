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

### Run on Colab

#### Requirements

##### Instruction
```
pip-requirements-colab.txt
```

##### mmcv
In case of eruption in colab kernel, mmcv is not installed by pip in requirements, but already in the folder mmcv.  
checkpoint and log file will output into Google Drive automatically, instead of `work_dir` set in train_colab.py.

#### Train
```
python train_colab.py
```

### Run on Huawei-ModelArts

#### To whl
As compile not permitted in the deploy phase on Huawei ModelArts Platform, we need to pack mmdetection to whl file

```
python setup.py install
python setup.py develop
python setup.py bdist_wheel
```

File: mmdet-2.0.0+unknown-cp36-cp36m-linux_x86_64.whl is packed on ModelArts Platform

#### Requirements
- With whl: `pip-requirements.txt`
- Without whl (Online Develop): `pip-requirements-without-whl.txt`

#### Train
Delete mmcv folder and then run the python file.
```
python train-modelarts.py
```

### pretrained model
- [resnet50-19c8e357.pth](https://bhpan.buaa.edu.cn:443/link/D74B0212071B1C26482F1689B6294626)
- [resnet101-5d3b4d8f.pth](https://bhpan.buaa.edu.cn:443/link/D9537C99C11CE4B645B9EE50E4923485)
- [resnext101_64x4d-ee2c6f71.pth](https://bhpan.buaa.edu.cn:443/link/52DD4E7BA15D4509E3631700D1D2B84F)
