## mAP Table

|    Backbone     | Faster-RCNN   | Cascade-RCNN |
| :-------------: | :-----------: | :----------: |
|    R-50-FPN     | 70.01 / 61.36 | 70.9 / 58.65 |
|    R-101-FPN    | 71.21 /       | 72.1 /  TBD  |
| X-101-32x4d FPN |               | 67.5 / 54.70 |

Note:  
- Task Different: Mask RCNN / Cascade Mask RCNN
- Low Performance: RPN, Fast RCNN, SSD, RetinaNet, double_head_rcnn, PAFPN

## Configs
## Cascade-RCNN-R-50-FPN
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

## Faster-RCNN-R-50-FPN
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

## Faster-RCNN-R-101-FPN
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
    - time = 
