
STAD-GWR: Continual Learning for Depression Video Diagnosis via Spatial-Temporal Attention Distillation and Gradient-Weight-Aware Regularization

## 数据集准备
1. 数据预处理：[preprocess.py](https://github.com/nextDeve/Depression-detect-ResNet)，从AVEC2013及AVEC2014一系列视频帧图像中裁剪人脸。图像应该是已经从视频中导出的帧。要直接运行程序，存储图像的路径格式应为`dataset_path/video_001/00001.jpg、dataset_path/video_001/00002.jpg`。
2. 将AVEC2013以及AVEC2014原始数据集按照刺激试验拆分出每个任务的视频片段(可使用videosplit.py进行自动处理，但需检查并手动调整最终视频片段数量)，并且按照每个任务的视频片段生成对应的标签文件，存放在data/AVEC2013和data/AVEC2014文件夹下。结构如下：
    ```shell
    ├── datasets
    │   ├── avec13
    │   │   ├── Training
    │   │   │   ├── task1
    │   │   │   │   ├── 203_1.mp4
    │   │   │   │   ├── ...
    │   │   │   │   ├── ...
    │   │   ├── Development
    │   │   │   ├── task1
    │   │   │   │   ├── 204_1.mp4
    │   │   │   │   ├──...
    │   │   ├── Testing
    │   │   │   ├── task1
    │   │   │   │   ├── 203_2.mp4
    │   │   │   │   ├──...
    ```

## 运行代码
  ```python
>>> python3 -u src/main.py --network STA --approach ours --num-tasks 10 --nepochs 500 --log disk --batch-size 5 --gpu 2 --exp-name dummy_functional_exp --lr 0.001 --seed 1 --lamb 1.0 --lr-patience 20 --plast_mu 1.0 --pool-along spatial
```
* `--approach`: 选择模型运行的方法。
* `--network`: 选择模型的网络结构。
* `--num-tasks`: 持续学习的任务数量。
* `--nepochs`: 每个任务训练的最大epoch数。
* `--log`: 结果保存路径。可选值为`disk`或`none`。
* `--batch-size`: 训练的批量大小。
* `--gpu`: 选择使用的GPU编号。
* `--exp-name`: 实验名称，用于在结果保存路径中创建子文件夹。
* `--lr`: 学习率。
* `--seed`: 随机种子。
* `--lamb`: 蒸馏损失的权重。
* `--lr-patience`: 学习率衰减的耐心值。


