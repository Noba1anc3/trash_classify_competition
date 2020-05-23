# PyTorch-YOLOv3-ModelArts
在华为云ModelArts平台部署PyTorch版本的Yolo-v3目标检测网络，实现模型训练、在线预测及参赛发布。
- source code: https://github.com/eriklindernoren/PyTorch-YOLOv3

## Preparation
### Convert Official Dataset's Annotation to New Standard Annotation
```
$ cd PyTorch-YOLOv3/tools
$ python prepare_datasets.py
```

## Usage
### Download Pretrained Weights
```
$ cd weights/
$ bash download_weights.sh
```

### Train
```
$ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]
```

### Test
```
$ python test.py --weights_path weights/yolov3.weights
```

### Inference
```
$ python detect.py --image_folder images/
```

### Tensorboard
Track training progress in Tensorboard:
- Initialize training
- Run the command below
- Go to http://localhost:6006/
```
$ tensorboard --logdir='logs' --port=6006
```

### 创建自定义模型的cfg文件
```
$ cd PyTorch-YOLOv3-ModelArts/config
$ bash create_custom_model.sh <num-classes> #此处已创建，即yolov3-44.cfg
```

## 在ModelArts平台上训练
1.将新数据集打包成压缩文件，替换原始数据集压缩包；

2.训练集和测试集的图片路径默认保存在config/train.txt和valid.txt中，每一行代表一张图片，默认按8：2划分。注意每行图片的路径为虚拟容器中的地址，自己重新划分训练集时只需要修改最后的图片名称，千万不要更改路径！

2.如果使用预训练模型，请提前将其上传到自己的OBS桶中，并添加参数

`--pretrained_weights = s3://your_bucket/{model}`。

此处的model可以是官方预训练模型（yolov3.weights或darknet53.conv.74），也可以是自己训练过的PyTorch模型（.pth）。

3.训练过程中，学习率等参数默认不进行调整，请依个人经验调整

4.其余流程同大赛指导文档。


## Credit

### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
[[Project Webpage]](https://pjreddie.com/darknet/yolo/)
[[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
