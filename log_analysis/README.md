## Logs
### Loss
- Start Loss : 235

### Evaluation Result

| ID | Time | Name                         | Epoch | Loss | Best mAP       | Purpose                         | Analysis              | Notes  |
|:--:|:----:|:----------------------------:|:-----:|:----:|:--------------:|:-------------------------------:|:---------------------:|:------:|
| 01 | 0525 | 不使用预训练模型              | 20    | 17.1 | 0.0515/19      | 观察直接训练的效果               | 直接训练提升很慢       | |
| 02 | 0524 | 预训练　+ 不冻结              | 13    | 9.48 | 2.35/3         | 比较不同冻结方案的差异           | 直接使用预训练不可取　  | |
| 03 | 0525 | 预训练　+ 冻结backbone        | 50    | 1.66 | 22.87/46       | 比较不同冻结方案的差异           | 有一定提升             | |
| 04 | 0525 | 预训练　+ 除三个检测层都冻结   | 28    | 12.6 | 29.9/23        | 比较不同冻结方案的差异           | 这是最好的冻结方案     | 初始map约为8~9   |
| 05 | 0526 | 同04 + 关闭多尺度训练　　　 　 | 27    | 12.9 | 27.7/12        | 观察多尺度训练带来的提升         | 有一定提升，但并不明显 | |
| 06 | 0526 | 同04　+ lr_1 = 1e-4           | 40    | 17.7 | 26.7/40        | 调低学习率，实验能否突破原有成绩 | 无法提升原有成绩       | |
| 07 | 0526 | 04最好的模型之上微调           | 12    | 2.85 | 27.74/10       | 观察微调最好模型能否突破原有成绩 | 无法提升原有成绩       | |
| 08 | 0526 | 同04(20轮) + 20轮微调　　　　　| 36    | 11.9 | 27.38/29　     | 实验冻结预训练+解除冻结开始微调  | 无法观察到效果提升     | |
| 08 | 0526 | 同04 + dataset_anchors        | 27    | 12.9 | 27.65/23       | 使用数据集宽高,观察能否带来提升  | 无法观察到效果提升     | 初始map约为10~12 |

### Analysis on the best model trained by Experiment-04 with mAP 29.9 on Evaluation Set
#### anno num per cls - mAP per cls
![](http://m.qpic.cn/psc?/fef49446-40e0-48c4-adcc-654c5015022c/90yfO.8bOadXEE4MiHsPn9NqWq4M1xjDX2dbcIyQVbx4wD53Fi7jb4WmTrd4pqXSwVO3YuuEpYn8Ol2rsZ3TqQ!!/b&bo=JwNtAicDbQIDCSw!&rf=viewer_4)

![](https://s1.ax1x.com/2020/05/27/tE6RGF.md.png)

Seems no relevance between anno num and mAP

#### anno size per cls - mAP per cls
![](http://m.qpic.cn/psc?/fef49446-40e0-48c4-adcc-654c5015022c/U9VSE8DftkGCrX.UXUSpm4A3MxgRGWaqdEC6qZjKpamqWyIRhvSjvZsZ8E050y9WY*syn8lDHfFMaNfnIqdDRJYrKEFsVvsWullhs3ocC*o!/b&bo=IQNpAiEDaQIDGTw!&rf=viewer_4)

[]:![](https://s1.ax1x.com/2020/05/27/tE6zqI.md.png)

Seems no relevance between anno size and mAP

#### anno ratio per cls - mAP per cls
![](http://m.qpic.cn/psc?/fef49446-40e0-48c4-adcc-654c5015022c/U9VSE8DftkGCrX.UXUSpm3sPsNqzkXE8YT.mDGBjZppdv65OJkOjXlk1xBNicgt9Hcuavs9pirFDAEaKS9B7*9.ZCG483bAlo0XwKKFwa74!/b&bo=LgNvAi4DbwIDGTw!&rf=viewer_4)

[]:![](https://s1.ax1x.com/2020/05/27/tEcCIf.md.png)

Seems some relevance between anno ratio and mAP

