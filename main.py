#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:yaoli 
@file: main.py.py 
@time: 2019/12/03 
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import argparse

from models import *

# 0. 从 shell 指定参数
parser = argparse.ArgumentParser(description='PyTorch CIFAR10')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default=False, action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 载入并标准化 CIFAR10 数据
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane','car','bird','cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. 定义卷积神经网络

#net = LeNet()
net = VGG('VGG16')

# 使用GPU时 当经过torch.nn.DataParallel(net)  后  net.__name__会报错， 所以提前指定 model_name代替该值
model_name = net.__name__
print(model_name + ' is ready!')


# 是否使用 GPU
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# 从断点继续训练或者重新训练
start_epoch = 0
best_acc = 0

if args.resume==True:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint/'+ model_name), 'Error : no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+ model_name+'/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1


# 3. 定义损失函数和优化器

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# 4. 训练神经网络
def train(epoch):
    running_loss = 0.0
    net.train()    # 这条代码似乎也不需要...
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)  # 在使用cpu的时候这条行代码自动忽略

        # 清零梯度缓存 zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印统计数据 print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


# 5. 测试网络
def test(epoch):
    global best_acc
    net.eval()  # 这条语句似乎也不需要..
    # dataiter = iter(testloader)
    # images, labels = dataiter.next()
    # outputs = net(images)

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        for i in range(10):
            correct += class_correct[i]
            total += class_total[i]
            # 输出每类识别准确率
            # print("Accuracy of %5s : %2d %%" % (
            #     classes[i], 100 * class_correct[i] / class_total[i]))
        acc = 100 * correct / total
        # 输出总准确率
        print("Accuracy of whole dataset: %.2f %%" % acc)


    # save checkpoint
    if acc > best_acc:
        print('Acc > best_acc, Saving net, acc')
        state = {
            'net':net.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        if not os.path.isdir('checkpoint/'+model_name):
            os.mkdir('checkpoint/'+model_name)
        torch.save(state, './checkpoint/'+model_name+'/ckpt.pth')
        best_acc = acc
        print('Saving success!')



for epoch in range(start_epoch, start_epoch+8):
    train(epoch)
    test(epoch)