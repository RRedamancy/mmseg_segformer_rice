# mmseg_segformer_rice

本代码基于mmSegmentation 0.x版本分支实现，建议先基于mmSegmentation 0.x版本分支安装好mmseg环境后再使用

其中Segformer部分的代码已经更新成了使用SIERRA上采样的形式

Rice数据集已经从原始的mask标注格式统一制作成了符合VOC标准的数据集格式，便于直接训练使用

训练使用的config文件为 ‘configs/segformer/my_model_segformer_b1.py’

### 前期准备

1. 安装好mmseg的0.x版本
   
2. 准备数据集。
   
   由于mmseg需要的数据集必须是标准标注格式，如coco，voc，ade20k等格式才行，所以需要将我们自己的二值标注数据集转化成标准格式。
 
   这里我选择使用VOC标准格式，尺寸为512*512.真实图像格式为JPG格式，标注的mask二值图像的格式为PNG格式（且二值图像的标签值为0和254）.（不能是255，注意！）
   
   我们可以通过运行rice-seg-voc/convert.py来将我们自己的数据集转化为对应格式。
   
4. 将数据集置放于正确的位置。
   
   第一步：在rice-seg-voc目录下建立如下三个文件夹
   ![image](https://github.com/RRedamancy/mmseg_segformer_rice/assets/100562008/fe408431-7c8e-46ea-91bf-5125633a91f8)

   第二步：将待测试数据的真实图像放在rice-seg-voc/JPEGImages文件夹中

   第三步：将待测试数据的标签图像放在rice-seg-voc/SegmentationClass文件夹中

   第四步：在rice-seg-voc\ImageSets\Segmentation文件夹中放置val.txt文件

   注意：val.txt文件中是待测试数据的名称，例如：![image](https://github.com/RRedamancy/mmseg_segformer_rice/assets/100562008/8ed8a288-c59a-4508-814d-6053c954dcea)

   这样，我们的数据就准备好了。

5.准备训练好的模型文件（.pth）

   模型文件可以自己训练，也可以用我准备好的。
   将准备好的work_dirs.zip文件在根目录下解压，就可以在根目录下看到一个work——dirs文件夹了，里面存有已经训练好的segformer模型。
   
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



