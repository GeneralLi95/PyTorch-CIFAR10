#!usr/bin/env python  
# -*- coding:utf-8 _*-
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
from utils import get_progress_bar, update_progress_bar

# 0. 从 shell 指定参数
parser = argparse.ArgumentParser(description='PyTorch CIFAR10')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default=False, action='store_true', help='resume from checkpoint')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用的 GPU 编号，0 是 name，不是 number
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 载入并标准化 CIFAR10 数据
# 1. Load and normalizing the CIFAR10 training and test datasets using torchvision
# data augmentation 数据增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. 定义卷积神经网络
# 2. Define a Convolution Neural Network

# net, model_name = LeNet(), 'LeNet'
# net, model_name = VGG('VGG11'), 'VGG11'
# net, model_name = VGG('VGG113'), 'VGG13'
# net, model_name = VGG('VGG16'), 'VGG16'
# net, model_name = ResNet18(), 'ResNet18'
# net, model_name = ResNet34(), 'ResNet34'
# net, model_name = ResNet50(), 'ResNet50'
# net, model_name = ResNet101(), 'ResNet101'
net, model_name = ResNet152(), 'ResNet152'

print(model_name + ' is ready!')

# 是否使用 GPU
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    print("Let's use", torch.cuda.device_count(), "GPUs")
    cudnn.benchmark = True

# 从断点继续训练或者重新训练
start_epoch = 0
best_acc = 0

if args.resume == True:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint/' + model_name), 'Error : no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + model_name + '/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1

# 3. 定义损失函数和优化器
# 3. Define a loss function

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# 4. 训练神经网络
# 4. Train the network on the training data

def train(epoch):
    running_loss = 0.0
    net.train()  # 这条代码似乎也不需要...
    correct = 0
    total = 0
    progress_bar_obj = get_progress_bar(len(trainloader))
    print('Epoch', epoch, 'Train')
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

        # 打印统计数据 print statistics  在 batch_size 不为 4 的情况下打印不出计数，改用 kuangliu 的 progress_bar
        # running_loss += loss.item()
        # if i % 2000 == 1999:
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        update_progress_bar(progress_bar_obj, index=i, loss=(running_loss / (i + 1)), acc=100. * (correct / total),
                            c=correct, t=total)


# 5. 测试网络
# 5. Test the network on the test data

def test(epoch):
    global best_acc
    net.eval()  # 这条语句似乎也不需要..
    # dataiter = iter(testloader)
    # images, labels = dataiter.next()
    # outputs = net(images)

    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    correct = 0
    total = 0
    test_loss = 0
    # progress_bar_obj = get_progress_bar(len(testloader))
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # update_progress_bar(progress_bar_obj, index=i, loss=(test_loss / (i + 1)), acc=100. * (correct / total),
            #                   c=correct, t=total)
            # c = (predicted == labels).squeeze()
            # for i in range(4):
            #     label = labels[i]
            #     class_correct[label] += c[i].item()
            #     class_total[label] += 1

            # for i in range(10):
            #     correct += class_correct[i]
            #     total += class_total[i]

            # 输出每类识别准确率
            # print("Accuracy of %5s : %2d %%" % (
            #     classes[i], 100 * class_correct[i] / class_total[i]))
        acc = 100 * correct / total
        # 输出总准确率
        print()
        print("Accuracy of whole dataset: %.2f %%" % acc)

    # save checkpoint

    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/' + model_name):
            os.mkdir('checkpoint/' + model_name)
        torch.save(state, './checkpoint/' + model_name + '/ckpt.pth')
        best_acc = acc
        print('Acc > best_acc, Saving net, acc')


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
