## DCGAN：Deep Convolutional Generative Adversarial Networks模型在pytorch当中的实现
---

### 目录
1. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [预测步骤 How2predict](#预测步骤)
5. [训练步骤 How2train](#训练步骤)
6. [参考资料 Reference](#Reference)

## 所需环境
pytorch==1.2.0    

## 文件下载
为了验证模型的有效性，我使用了**花的例子**进行了训练。    
训练好的生成器与判别器模型[Generator_Flower.pth](https://github.com/bubbliiiing/dcgan-pytorch/releases/download/v1.0/Generator_Flower.pth)、[Discriminator_Flower.pth](https://github.com/bubbliiiing/dcgan-pytorch/releases/download/v1.0/Discriminator_Flower.pth)可以通过百度网盘下载或者通过GITHUB下载    
权值的百度网盘地址如下：    
链接: https://pan.baidu.com/s/1AMh52TauVT7nyn874BCAgg 提取码: dubv  

花的数据集可以通过百度网盘下载：   
链接: https://pan.baidu.com/s/1ITA1Lw_K28B3nbNPnI3_Kw 提取码: 11yt  

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，直接运行predict.py，在终端点击enter，即可生成图片，生成图片位于results/predict_out/predict_1x1_results.png，results/predict_out/predict_5x5_results.png。    
### b、使用自己训练的权重 
1. 按照训练步骤训练。    
2. 在dcgan.py文件里面，在如下部分修改model_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件**。    
```python
_defaults = {
    #-----------------------------------------------#
    #   model_path指向logs文件夹下的权值文件
    #-----------------------------------------------#
    "model_path"        : 'model_data/Generator_Flower.pth',
    #-----------------------------------------------#
    #   卷积通道数的设置
    #-----------------------------------------------#
    "channel"           : 64,
    #-----------------------------------------------#
    #   输入图像大小的设置
    #-----------------------------------------------#
    "input_shape"       : [64, 64],
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True,
}
```
3. 运行predict.py，在终端点击enter，即可生成图片，生成图片位于results/predict_out/predict_1x1_results.png，results/predict_out/predict_5x5_results.png。    

## 训练步骤
1. 训练前将期望生成的图片文件放在datasets文件夹下（参考花的数据集）。  
2. 运行根目录下面的txt_annotation.py，生成train_lines.txt，保证train_lines.txt内部是有文件路径内容的。  
3. 运行train.py文件进行训练，训练过程中生成的图片可查看results/train_out文件夹下的图片。  
