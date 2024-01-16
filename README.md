# mmseg_segformer_rice

本代码基于mmSegmentation 0.x版本分支实现，建议先基于mmSegmentation 0.x版本分支安装好mmseg环境后再使用

其中Segformer部分的代码已经更新成了使用SIERRA上采样的形式

Rice数据集已经从原始的mask标注格式统一制作成了符合VOC标准的数据集格式，便于直接训练使用

训练使用的config文件为 ‘configs/segformer/my_model_segformer_b1.py’

#### 使用单卡 GPU 训练

```shell
python tools/train.py ${CONFIG_FILE} [可选参数]
```

如果你想用单卡GPU训练我们自己的segformer模型，运行以下代码即可（xxx为自己指定的pth文件保存路径）：

```shell
python tools/train.py configs/segformer/my_model_segformer_b1.py --work-dir work_dirs/xxx/
```

#### 使用多卡 GPU 训练

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [可选参数]
```

可选参数可以为:

- `--work-dir ${工作路径}`: 在配置文件里重写工作路径文件夹
- `--resume-from ${检查点文件}`: 继续使用先前的检查点 (checkpoint) 文件（可以继续训练过程）
- `--load-from ${检查点文件}`: 从一个检查点 (checkpoint) 文件里加载权重（对另一个任务进行精调）
- `--deterministic`: 选择此模式会减慢训练速度，但结果易于复现

如果你想用多卡GPU训练我们自己的segformer模型，运行以下代码即可（这里我用了4卡 ,xxx为自己指定的pth文件保存路径）：

```shell
bash tools/dist_train.sh configs/segformer/my_model_segformer_b1.py 4 --work-dir work_dirs/xxx/ --deterministic
```

### 模型的测试和mIoU评估
例子:

假设您已经获得检查点pth文件至文件夹 `work_dirs/xxx/` 里。

1. 测试 segformer 并可视化结果。按下任何键会进行到下一张图

   ```shell
   python tools/test.py configs/segformer/my_model_segformer_b1.py work_dirs/xxx/iter_4000_best.pth --show
   ```

2. 测试 segformer 并保存画出的图以便于之后的可视化。（xxxxx为指定保存的show文件夹路径）

   ```shell
   python tools/test.py configs/segformer/my_model_segformer_b1.py work_dirs/xxx/iter_4000_best.pth --show-dir show_dirs/xxxxx
   ```

1. 测试 segformer 并评估 mIoU

   ```shell
   python tools/test.py configs/segformer/my_model_segformer_b1.py work_dirs/xxx/iter_4000_best.pth --eval mIoU
   ```



