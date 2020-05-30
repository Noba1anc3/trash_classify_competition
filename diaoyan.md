
## 数据分析
- 数据量小
- 类别多且不均衡
- 图片小，分辨率低，垃圾尺度多样

## 模型
- ResNet
- ResNeXt
  - se_ResNeXt_50_32x4d
  - se_ResNeXt_101_32x4d
  - ResNeXt_50_32x8d
  - **ResNeXt_101** (Instagram Weights Transferred)
    - ResNeXt_101_32x8d_wsl (weak supervised learning)
    - ResNeXt_101_32x16d_wsl
    - ResNeXt_101_32x32d_wsl
- Senet154
- PnasNet
- DPN-107
- **EfficientNet b7/b5/b4/b3**

## 数据清洗，增强，扩充
- 数据扩充
  - 监督，半监督
  - 差别性图像扩充
- AutoAugment
- self_define
  - 几何变换: shift, rotate, flip, scale, random crop
  - Color Jittering: 亮度，饱和度，对比度
  - Random Noise
  - RICAP: random image cropping and patching
  - Cutout
  - Mixup: 线性插值
  - Cutmix

## 损失函数
- Cross Entropy
- Focal Loss

## 超参数
- img_size
- Learning Rate Schedule
  - ReduceLROnPlateau
  - Hierarchical Warmup
    - MultiStep LR
    - Inception LR
- Optimizer
  - AdaBoost
  - Adam
  - RAdam
  - SGD
- Activation Function
  - Linear
  - ReLU
  - Sigmoid
  - Tanh
- Batch Size
- Dropout


## 优化
- Batch Normalization
- K-Fold Cross Validation
- **Label Smoothing**
- Upsampling
- **Dropout**
- CBAM (Convolutional Block Attention Module)
- OHEM (Online Hard Example Mining)
- Multi-teacher Knowledge Distillation
- Weight Decay

## 流程
- 简化预处理，后处理
