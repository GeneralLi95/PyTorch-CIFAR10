# PyTorch-CIFAR

利用 PyTorch 在 CIFAR10 数据集上实现多种神经网络方法。

## 实验记录:

 Epoch | 1 | 2 | 4 |8 |16
 :--- :|:---: |:---:|:---:|:---:|:---:
 LeNet|48.09%|54.10%|60.82%|60.93%|62.78%
 VGG16|59.31%|69.14%|79.33%|83.44%
 ResNet18|66.25%|78.35%|81.04%|81.08%|83.72%|
 
 
 ## 论文链接:
 
 模型|层数 | 论文链接 |发表时间|Google学术引用数(2019.12)
 :---:|:---:| :---:|:---:|:---:
 LeNet|5|[Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)|1998|23110
VGG|16|[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf%20http://arxiv.org/abs/1409.1556.pdf) |ICLR 2015|31319
ResNet|50|[Deep residual learning for image recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)|CVPR2016|35470|


## 致谢

* [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
* 