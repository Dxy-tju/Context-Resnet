'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ContextResNet', 'contextresnet20', 'contextresnet32', 'contextresnet44', 'contextresnet56', 'contextresnet110', 'contextresnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

import torch.nn as nn
import torch
import numpy as np

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, stride, padding):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        # self.conv1 = nn.Conv2d(in_channels=2*nHidden, out_channels=2*nHidden,
        #                        kernel_size=3, stride=stride, padding=padding, bias=False)
        # self.relu = nn.ReLU()

    def forward(self, input):
        output, _ = self.rnn(input)
        # print(recurrent.shape)
        # [8, 8, 128]
        # output = output.permute(2,0,1)
        # recurrent = torch.unsqueeze(recurrent, dim=0)
        # output = self.conv1(recurrent)
        # output = self.relu(output)
        return output

class SeqNet(nn.Module):
    
    def __init__(self, nIn, nHidden, nOut, stride, padding):
       super(SeqNet, self).__init__()
       self.rnn_topdown = BidirectionalLSTM(nIn, nHidden, stride, 1)
       self.rnn_leftright = BidirectionalLSTM(nIn, nHidden, stride, 1)
       self.conv1 = nn.Conv2d(in_channels=4*nHidden, out_channels=nOut,
                               kernel_size=1, stride=1, padding=padding, bias=False)
       self.bn1 = nn.BatchNorm2d(nOut)
       self.relu = nn.ReLU()

    def forward(self, input):
        input = np.squeeze(input);
        # print(input.shape) [64, 32, 32]
        input = input.permute(1,2,0)
        # print(input.shape) [32, 32, 64]

        out1 = self.rnn_topdown(input)
        # print("out1",out1.shape) [32, 32, 128]
        # 输入逆时针旋转90度
        out2 = self.rnn_leftright(torch.rot90(input, -1, [0, 1]))
        # print("out2",out2.shape) [32, 32, 128]
        
        
        out2 = torch.rot90(out2, 1, [0, 1])
        output = torch.cat([out1, out2],2)
        # output shape [32, 32, 256]
        
        output = output.permute(2,0,1)
        output = torch.unsqueeze(output, dim=0)
        
        output = self.conv1(output)
        output = self.bn1(output)
        output = self.relu(output)
        return output


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ContextResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ContextResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.context_layer = SeqNet(16, 16, 16, 1, 0)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.context_layer(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def context_resnet20():
    return ContextResNet(BasicBlock, [3, 3, 3])


def context_resnet32():
    return ContextResNet(BasicBlock, [5, 5, 5])


def context_resnet44():
    return ContextResNet(BasicBlock, [7, 7, 7])


def context_resnet56():
    return ContextResNet(BasicBlock, [9, 9, 9])


def context_resnet110():
    return ContextResNet(BasicBlock, [18, 18, 18])


def context_resnet1202():
    return ContextResNet(BasicBlock, [200, 200, 200])
