## Logs
### Loss
- Start Loss : 235
- Epoch-1 Loss : 66
- Epoch-2 Loss : 39
- Epoch-3 Loss : 28
- Epoch-4 Loss : 24
- Epoch-5 Loss : 20
- Epoch-6 Loss : 19
- Epoch-7 Loss : 19
- Epoch-8 Loss : 17
- Epoch-9 Loss : 15
- Epoch-10 Loss : 15
- Epoch-11 ~ 27 Loss : 13 ~ 14

### Evaluation Result
- Without Pretrained Model

| ID | Time | Name                                  | Epoch | Loss | Best mAP       | Purpose | Analysis | Beizhu |
|:--:|:----:|:-------------------------------------:|:-----:|:----:|:--------------:|:-------:|:--------:|:------:|
| 01 | 0525 | 不使用预训练模型                       | 20    | 17.1 | 0.0515/19      |         |          |        |
| 02 | 0524 | 预训练　+ 一阶段50轮 freeze_body_1 = 0 | 13    | 9.48 | 2.35/3         |         |          |        |
| 03 | 0525 | 预训练　+ 一阶段50轮 freeze_body_1 = 1 | 50    | 1.66 | 22.87/46       |         |          |        |
| 04 | 0525 | 预训练　+ 一阶段50轮 freeze_body_1 = 2 | 28    | 12.6 | 29.9/23        |         |          |        |
