import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo


class BasicBlock(nn.Module):

    def __init__(self, planes):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1)
        # parameters initialization
        nn.init.normal(self.conv1.weight, mean=0, std=0.01)
        nn.init.constant(self.conv1.bias, 0)
        nn.init.normal(self.conv2.weight, mean=0, std=0.01)
        nn.init.constant(self.conv2.bias, 0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out += residual
        return out


class LSTMNet(nn.Module):

    def __init__(self, block, layers):
        super(LSTMNet, self).__init__()

        self.conv1_a = nn.Conv2d(3,  32, 3, stride=1)
        self.bn1_a = nn.BatchNorm2d(32)
        self.conv1_b = nn.Conv2d(32, 64, 3, stride=1)
        self.bn1_b = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.conv2   = nn.Conv2d(64,  128, 3, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.conv3   = nn.Conv2d(128, 256, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.conv4   = nn.Conv2d(256, 512, 3, stride=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.layer4 = self._make_layer(block, 512, layers[3])

        self.fc5_new = nn.Linear(512*5*4, 512)
        
        self.fc8_final = nn.Linear(512, 7)

        # parameters initialization #
        layers = [self.conv1_a,self.conv1_b,self.conv2,self.conv3,self.conv4]

        self.InitParam(layers)


    def InitParam(self,layers):
        for layer in layers:
            nn.init.xavier_uniform(layer.weight)
            nn.init.constant(layer.bias, 0)


    def _make_layer(self, block, planes, blocks):

        layers = []
        for i in range(0, blocks):
          layers.append(block(planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = F.relu(self.bn1_a(self.conv1_a(x)))
        x_pool1b = F.max_pool2d(F.relu(self.bn1_b(self.conv1_b(x))),2, stride=2)

        x = self.layer1(x_pool1b)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))),2, stride=2)

        x = self.layer2(x)
        x_pool3 = F.max_pool2d(F.relu(self.bn3(self.conv3(x))),2, stride=2)
        x = self.layer3(x_pool3)
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))),2, stride=2)
        x = self.layer4(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc5_new(x)
        
        # x1 = x1.view(1,-1,512)
        # x1, hn1 = self.lstm1(x1, (self.h1, self.c1))
        x = self.fc8_final(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]   
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

