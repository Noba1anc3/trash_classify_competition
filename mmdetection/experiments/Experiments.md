## Baseline
### mAP Table

|    Backbone     | Faster-RCNN   | Cascade-RCNN |
| :-------------: | :-----------: | :----------: |
|    R-50-FPN     | 70.01 / 61.36 | 70.9 / 58.65 |
|    R-101-FPN    | 71.21 / 61.46 | 72.1 /  TBD  |
| X-101-32x4d FPN |               | 67.5 / 54.70 |

Note:  
- Task Different: Mask RCNN / Cascade Mask RCNN
- Low Performance: RPN, Fast RCNN, SSD, RetinaNet, double_head_rcnn, PAFPN

### Configs
#### Cascade-RCNN-R-50-FPN
- Train
    - batch = 4
    - Optimizer
        - SGD(momentum = 0.9, weight_decay = 1e-4)
        - config = grad_clip=dict(max_norm=35, norm_type=2)
    - lr config
        - initial = 5e-3
        - step = 8,11
        - warmup = linear
        - warmup_iter = 2000
        - warmup_ratio = 1e-3
    - best_epoch=12/20
- Inference
    - resize = 650*650
    - time = 3h

#### Faster-RCNN-R-50-FPN
- Train
    - batch = 8
    - Optimizer
        - SGD(momentum = 0.9, weight_decay = 1e-4)
        - config = grad_clip=None
    - lr config
        - initial = 1e-2
        - step = 8,11
        - warmup = linear
        - warmup_iter = 500
        - warmup_ratio = 1e-3
    - best_epoch=17/20
- Inference
    - resize = 1000*600
    - time = 2h32m

#### Faster-RCNN-R-101-FPN
- Train
    - batch = 8
    - Optimizer
        - SGD(momentum = 0.9, weight_decay = 1e-4)
        - config = grad_clip=None
    - lr config
        - initial = 1e-2
        - step = 8,11
        - warmup = linear
        - warmup_iter = 500
        - warmup_ratio = 1e-3
    - best_epoch=13/17
- Inference
    - resize = 1000*600
    - nms_pre=500/700
    - nms_post=500/700
    - max_num=500/700
    - time = 2h43m/2h53m
    - score = 61.39/61.46

## Optimize
### Optimizer
Baseline : Faster-RCNN ResNet50 FPN  
LR WarmUp : Linear, iter=500, ratio=1e-3

Ex-01. Adam & Amsgrad

| Optimizer | Ini lr | Lr Decay | Total | 1st   | Before_1 | After_1 | Before_2 | After_2 | Last  | Best       |
| --------- | ------ | -------- | ----- | ----- | -------- | ------- | -------- | ------- | ----- | ---------- |
| Adam      | 2E-4   | [12, 16] | 18    | 20.77 | 51.33    | 58.73   | 60.72    | 61.50   | 61.50 | 61.50 / 17 |
| Amsgrad   | 2E-4   | [8, 11]  | 15    | 25.4  | 52.44    | 61.89   | 63.33    | 64.17   | 64.62 | 64.67 / 13 |
| Adam      | 5E-4   | None     | 6     | 5.3   | NaN      | NaN     | NaN      | NaN     | 25.8  | 25.80 / 06 |
| Amsgrad   | 5E-4   | None     | 3     | 8.46  | NaN      | NaN     | NaN      | NaN     | 21.93 | 21.93 / 03 |

Ex-02. Different lr-decay

| Optimizer | Ini lr | Lr Decay | Total | 1st   | Before_1 | After_1 | Last  | Best       |
| --------- | ------ | -------- | ----- | ----- | -------- | ------- | ----- | ---------- |
| Amsgrad   | 1E-4   | [8]      | 14    | 36.19 | 59.48    | 66.40   | 67.90 | 68.18 / 13 |
| Amsgrad   | 1E-4   | [10]     | 14    | 36.19 | 60.38    | 67.24   | 68.64 | 68.64 / 14 |

Ex-03. Best Initial LR

| Optimizer | Ini lr | 1st   |
| --------- | ------ | ----- |
| Amsgrad   | 1E-4   | 32.11 |
| Amsgrad   | 3E-4   | 42.86 |
| Amsgrad   | 5E-4   | 42.33 |
| Amsgrad   | 7E-4   | 37.77 |

Ex-04. Adam Best

Optimizer : Amsgrad  
Ini lr : 5E-4

| Lr Decay    | Total | 1st   | Before_1 | After_1 | Before_2 | After_2 | Before_3 | After_3 | Last  | Best       |
| ----------- | ----- | ----- | -------- | ------- | -------- | ------- | -------- | ------- | ----- | ---------- |
| [8, 11, 14] | 17    | 42.86 | 65.53    | 69.5    | 70.55    | 71.2    | 70.88    | 71.14   | 71.29 | 71.29 / 17 |

### BBox-Head Loss
#### IoU, GIoU, Bounded-IoU Loss
- Baseline : Faster-RCNN ResNet50 FPN
- Lr : 1e-5
- Lr WarmUp : Linear, iter=500, ratio=1e-3
- Optimizer : Adagrad

| Loss        | Epoch 1 | Epoch 2  | Epoch 3  | Epoch 4  | Epoch 5  | Epoch 6  |
| ----------- | ------- | -------- | -------- | -------- | -------- | -------- |
| Baseline    | 42.33   | 52.20    | 53.63    | 57.28    | 60.10    | 62.00    |
| IoU         | 40.87   | 50.33    | 56.74    | 57.48    | 59.45    | 62.38    |
| GIoU        | 37.68   | 50.25    | 52.45    | 58.83    | 61.51    | 60.98    |
| Bounded-IoU | 39.25   | 49.12    | 52.68    | 55.47    | 60.16    | 57.48    |

#### IoU Loss
| Lr Decay    | Total | 1st   | Before_1 | After_1 | Before_2 | After_2 | Before_3 | After_3 | Last  | Best       |
| ----------- | ----- | ----- | -------- | ------- | -------- | ------- | -------- | ------- | ----- | ---------- |
| [8, 11, 14] | 17    | 40.87 | 65.25    | 69.76   | 70.41    | 70.44   | 70.59    | 70.62   | 70.57 | 70.62 / 13 |

### OHEM
- Baseline : Faster-RCNN ResNet50 FPN
- Lr : 1e-5
- Lr WarmUp : Linear, iter=500, ratio=1e-3
- Optimizer : Adagrad

| Name     | Lr Decay | Total | 1st   | Before_1 | After_1 | Before_2 | After_2 | Last  | Best       |
| -------- | -------- | ----- | ----- | -------- | ------- | -------- | ------- | ----- | ---------- |
| Baseline | [8, 11]  | 14    | 42.86 | 65.53    | 69.50   | 70.55    | 71.20   | 71.20 | 71.20 / 14 |
| OHEM     | [8, 11]  | 14    | 39.98 | 64.39    | 69.80   | 70.23    | 70.25   | 70.60 | 70.61 / 13 |

### Focal Loss
- Baseline : Cascade-RCNN ResNet50 FPN
- Lr : 1e-5
- Lr WarmUp : Linear, iter=500, ratio=1e-3
- Optimizer : SGD(momentum=0.9, weight_decay=0.0001)
- Lr Decay : [8, 11]
- Score : 44.2 (16/16)
- Note : Focal loss usually used in one-stage algorithm, it doesnt work on two-stage algorithm. For the imbalanced problem, it is beyond the scope of the codebase usage.
