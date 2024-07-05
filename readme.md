# UNet语义分割冠状血管
本项目使用U-Net对XCAD数据集上的冠状血管造影图像进行语义分割
出自于北京邮电大学计算机学院夏令营9组 柴铠琰

## 目录结构
```angular2html
|-- U-Net
    |-- README.md
    |-- best_model
    |-- Model
    |   |-- best_model.pth
    |   |-- best_model_150.pth
    |   |-- ceshi.py
    |   |-- dataset.py
    |   |-- evaluation_metrics_200.png
    |   |-- predict.py
    |   |-- train.py
    |   |-- training_loss_200.png
    |   |-- unet_model.py
    |   |-- unet_parts.py
    |-- XCAD
    |   |-- test
    |   |   |-- images
    |   |   |-- masks
    |   |   |-- res
    |   |   |-- res_predict_150
    |   |   |-- res_predict_50
    |   |-- train
    |       |-- images
    |       |-- masks
```

## 项目文件说明
### Model文件夹
> best_model.pth文件保存了模型训练的参数，可用于预测

> best_model_150.pth文件保存了模型训练150轮的参数，可用于预测

> clear_res.py用于清理模型预测生成的图像，帮助实现重复实验快速进行

> dataset.py用于加载数据，并对图像进行预处理 

> predict.py用于调用保存的模型参数进行预测

> trian.py用于训练

> unet_parts.py定义了U_Net网络的双卷积层、最大池化等结构模块，增强代码复用性

> unet_model.py定义了U-Net网络

> training_loss.png保存了训练的loss，用于模型评估

> evaluation_metrics.png保存了P值、R值和F1分数，用于模型评估

### XCAD文件夹
#### trian文件夹
保存了训练图像和mask
#### test文件夹
```angular2html
|-- test
    |   |   |-- images #测试集图像
    |   |   |-- masks  #测试集图像对应的mask
    |   |   |-- res    #模型预测输出
    |   |   |-- res_predict_150  #训练150轮的模型预测输出
    |   |   |-- res_predict_50   #训练50轮的模型预测输出
```

