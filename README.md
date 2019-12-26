# PyTorch-CIFAR

利用 PyTorch 在 CIFAR10 数据集上实现多种神经网络方法。

## 实验记录:
**lr = 0.001, batch_size = 128, epoch = 300, GTX 2080 Ti** 

 model|best_acc
 ---|---
 LeNet|74.27%|
 VGG11|88.88%|
 VGG13|90.98%|
 VGG16|90.80%|
 ResNet18|92.19%|
 
 

 
 
 ## 论文链接:
 
 模型|层数 | 论文链接 |发表时间|Google学术引用数(2019.12)
 :---: |:---:| :---:|:---:|:---:
 LeNet|5|[Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)|1998|23110
VGG|11\13\16|[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf%20http://arxiv.org/abs/1409.1556.pdf) |ICLR 2015|31319
ResNet|18\34\50\101\152|[Deep residual learning for image recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)|CVPR2016|35470|


## 致谢

* [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
* [pytorch/Deep Learning with PyTorch: A 60 Minute Blitz > Training a Classifier 
](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)

