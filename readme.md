# 一、实验环境

```
from __future__ import print_function
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import copy
```

​	上述代码是实验中所有需要导入的包的 $import$ 语句汇总，在实验开始之前，需要对实现实验所需的包进行安装，下面对需要所有包进行解释。

+ $torch$ 是实现 $PyTorch$ 进行风格转换必不可少的包，需安装。
+ $torch.nn$ 和 $torch.optim$ 是 $torch$ 下的模块，是用来构建和训练网络的以及进行高效梯度下降的模块。
+ $PIL$ 是实现图像处理的模块，需安装。
+ $matplotlib$ 是 $Python$ 中进行绘图的库，需安装。
+ $torchvision$ 是 $PyTorch$ 的一个图形库，它服务于 $PyTorch$ 深度学习框架的，主要用来构建计算机视觉模型，需安装。
+ $torchvision.models$：包含常用的模型结构（含预训练模型），例如 $AlexNet$、$VGG$、$ResNet$ 等。
+ $torchvision.transforms$：常用的图片变换，例如裁剪、旋转等。
+ $copy$：对模型进行深度拷贝。

​	实验需要安装的依赖包：依次执行下列语句。

```
pip install torch
#PIL名称已经换成pillow
pip install pillow
pip install matplotlib
pip install torchvision
```

# 二、数据集下载

​	实验采用$vangogh2photo$数据集，该数据集包含梵高画作风格图像和风景风格图像，梵高画作集由$755$幅$256×256$大小的图像组成，选取其中4张图像作为风格图像；风景风格图像集由$6287$幅$256×256$大小的图像组成，选取其中$100$张图像作为内容图像集，同时为该内容图像集进行重新编号。实验中将内容图像集数据作为训练数据中的源域图像，存放在文件夹 \<images\> 和 \<content\> 内，将梵高画作集和外加的3张艺术风格图像作为目标域，存放在文件夹 \<images\> 的 \<style\> 内，源域与目标域的图像数目分别是$100$张与$7$张。作为源域的图像主要是各种自然景物包括动物等在内的真实世界图像，而作为目标域的图像来自于多幅艺术风格图像，其中也包括梵高的抽象画作。本实验只需要从源域与目标域中随机选择图像进行匹配训练即可。

# 三、运行方式

​	训练文件 $style.py$ 可以对所有风格图片和所有风景内容图片进行遍历训练，最终将图片保存至文件夹 \<images\> 的 \<output\> 内。直接使用 $vscode$ 打开，点击右上角运行该 $python$ 文件即可。

# 四、实验结果

<img src="file:///C:/Users/laiping/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg" alt="img" style="zoom: 67%;" />

<img src="file:///C:/Users/laiping/AppData/Local/Temp/msohtmlclip1/01/clip_image004.jpg" alt="img" style="zoom:67%;" />

<img src="file:///C:/Users/laiping/AppData/Local/Temp/msohtmlclip1/01/clip_image006.jpg" alt="img" style="zoom:67%;" />

<img src="file:///C:/Users/laiping/AppData/Local/Temp/msohtmlclip1/01/clip_image008.jpg" alt="img" style="zoom:67%;" />

<img src="file:///C:/Users/laiping/AppData/Local/Temp/msohtmlclip1/01/clip_image010.jpg" alt="img" style="zoom:67%;" />

<img src="file:///C:/Users/laiping/AppData/Local/Temp/msohtmlclip1/01/clip_image012.jpg" alt="img" style="zoom:67%;" />

<img src="file:///C:/Users/laiping/AppData/Local/Temp/msohtmlclip1/01/clip_image014.jpg" alt="img" style="zoom:67%;" />
