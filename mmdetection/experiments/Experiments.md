## mAP Table

|    Backbone     | Faster-RCNN   | Cascade-RCNN |
| :-------------: | :-----------: | :----------: |
|    R-50-FPN     | 70.01 / 61.36 | 70.9 / 58.65 |
|    R-101-FPN    |               |              |
| X-101-32x4d FPN |               |              |
| X-101-64x4d FPN |               |              |

Note:  
- Task Different: Mask RCNN / Cascade Mask RCNN
- Low Performance: RPN, Fast RCNN, SSD, RetinaNet, double_head_rcnn, PAFPN

## Configs
## Cascade-RCNN-R-50-FPN
- Train
    - batch = 4
    - Optimizer = SGD(momentum = 0.9, weight_decay = 1e-4)
    - lr config
        - initial = 2e-2
        - step = 8,11
        - warmup = linear
        - warmup_iter = 500
        - warmup_ratio = 1e-3
    - best_epoch=12/20
- Inference
    - resize = 650*650
    - time = 3h

## Faster-RCNN-R-50-FPN
- Train
    - batch = 8
    - Optimizer = SGD(momentum = 0.9, weight_decay = 1e-4)
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
