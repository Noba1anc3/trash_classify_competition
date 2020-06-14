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

| Lr Decay     | Total | 1st   | Before_1 | After_1 | Before_2 | After_2 | Before_3 | After_3 | Last  | Best       |
| ------------ | ----- | ----- | -------- | ------- | -------- | ------- | -------- | ------- | ----- | ---------- |
| [8, 11, 14]  | 17    | 42.86 | 65.53    | 69.5    | 70.55    | 71.2    | 70.88    | 71.14   | 71.29 | 71.29 / 17 |

