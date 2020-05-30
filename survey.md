## 目标
- 高准确率，高推理效率，低资源占用
- 简化预处理，后处理

## 数据分析
- 数据量小
- 类别多且不均衡
- 图片小，分辨率低，垃圾尺度多样
- 特征粒度有粗有细

## 模型筛选
- ResNet
- ResNeXt
    - se_ResNeXt_50_32x4d
    - se_ResNeXt_101_32x4d
    - ResNeXt_50_32x8d
    - **ResNeXt_101** (Weak-Supervised Trained on Instagram and finetune on ImageNet)
        - ResNeXt_101_32x8d_wsl (weak supervised learning)
        - ResNeXt_101_32x16d_wsl
        - ResNeXt_101_32x32d_wsl
- Senet154
- PnasNet
- DPN-107
- **[EfficientNet b7/b5/b4/b3](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)**

## Adaptive Head
- Classical
    - Average Pooling -> Dense -> Softmax
    - 丢失局部代表性特征
    - 泛化性一般
- Customized
    - Concat(AvgPool, MaxPool) -> Flatten -> BatchNorm -> Dropout -> Linear -> ReLU -> BatchNorm -> Dropout -> Linear
    - MaxPooling兼顾局部代表性特征
    - Batch Norm & Dropout提升头部泛化能力

## 数据清洗,增强,扩充
- AutoAugment
- [self_define](http://imgaug.readthedocs.io/en/latest/index.html)
    - 几何变换: shift, rotate, flip, scale, [crop](https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/ops/random_ops.py#L290-L331)
    - Color Jittering: 亮度，饱和度，对比度
    - Random Noise
    - RICAP
        - Random Image Cropping and Patching Data Augmentation for Deep CNNs
    - Cutout
        - [Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896) 
        - [Code](https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/ops/random_ops.py#L290-L331)
    - [Mixup](https://www.jianshu.com/p/d22fcd86f36d)
    - CutMix
        - CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
- 数据扩充
    - 日常生活图片/爬取网络图片/使用[开源数据](https://github.com/Yangget/garbage_classify_expand)
    - 监督 / 半监督
    - 差别性图像扩充
        - 数据量少
        - 分类效果差 (混淆矩阵)
    - Pseudo Labelling
        - Hard Label / Soft Label
        - 借助弱监督方式引入外部数据集中高质量数据: 解决了自行扩展数据集带来的测试偏移

## 损失函数
- Cross Entropy
- **Focal Loss**
    - Focal Loss for Dense Object Detection
    - 解决one-stage目标检测中正负样本比例严重失衡的问题
    - 降低了大量简单负样本在训练中所占的权重,也可理解为一种困难样本挖掘

## Learning Rate Schedule
- ReduceLROnPlateau
- Hierarchical Warmup
    - MultiStep LR
    - Inception LR

## 优化器
- SGD
- AdaBoost
- Adam
- RMSProp
- Rectified Adam
    - Adam、RMSProp这些算法虽然收敛速度很快,但往往会掉入局部最优解的“陷阱”;
    - 原始的SGD方法虽然能收敛到更好的结果,但是训练速度太慢。
    - RAdam收敛速度快,也不容易掉入局部最优解,收敛结果对学习率的初始值非常不敏感。
    - [Code](https://github.com/LiyuanLucasLiu/RAdam)

## 超参数
- img_size
- learning rate
- Weight Decay
- Batch Size

## 提升策略
- K-Fold Cross Validation
- Upsampling
- **Dropout**
- **CBAM (Convolutional Block Attention Module)**
    - 首层卷积/特征池化层
    - 减小处理高维输入数据的计算负担,通过结构化的选取输入的子集,降低数据维度。
    - 让任务处理系统更专注于找到输入数据中显著的与当前输出相关的有用信息,从而提高输出的质量
- **Label Smoothing Regularization**
    - 迫使模型往增大正确分类概率并且同时减小错误分类概率的方向前进
    - [Reference](https://blog.csdn.net/sinat_36618660/article/details/100166957)
    - [Rethinking the Inception Architecture for ComputerVision](https://arxiv.org/pdf/1512.00567.pdf)
    - [Code](https://github.com/tensorflow/cleverhans/blob/master/cleverhans_tutorials/mnist_tutorial_tf.py)
- OHEM (Online Hard Example Mining)
    - topK loss backward

## 展望
- AutoML
- Multi-teacher Knowledge Distillation
- 对网络所有残差模块添加时间和空间注意力机制
- 对模型进行量化剪枝，保证精度的同时提升速度
- 联合loss: arcface_loss / triplet loss / focal loss
- 面对落地应用的情况，增加反馈机制，对模型进行在线训练

