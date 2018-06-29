import sys
import os
import time
import math
import torch
import torch.nn as nn
import numpy as np
import random
from PIL import Image
from collections import OrderedDict
from torch.autograd import Variable
from torchvision import models
from VGG_Face_torch import VGG_Face_torch


loss_function = nn.CrossEntropyLoss()

class AccumulatedAccuracyMetric():
    """
    Works with classification model
    """
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.loss = 0

    def __call__(self, outputs, target):
        pred = outputs.data.cpu().max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.cpu().view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        self.loss += loss_function(outputs, target)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.loss = 0

    def value(self):
        return float(self.correct) / self.total, float(self.loss)
    
    def correct_number(self):
        return self.correct




def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                #print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]
    return z
      

def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))



def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res



def data_augmentation(img ,shape, jitter, hue, saturation, exposure):
    oh = img.height  
    ow = img.width
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2
    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)

    if flip: 
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)

    img = random_distort_image(sized, hue, saturation, exposure)
    
    return img

def blurry_image(img):

    img1 = 255 - img
    img1 = np.uint8(img1/float(np.max(img1))*255)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            # if img1[i][j] > 126 :
            #     img1[i][j] = 255
            if img1[i][j] > 50 :
              # img1[i][j] = img1[i][j]+30
                img1[i][j] = 255
            else:
                img1[i][j] = (math.log(1+img1[i][j])) * 256/math.log(256)

    img = img1/float(255)

    return img


class VGG_Net(nn.Module):
    def __init__(self, model):
        super(VGG_Net, self).__init__()

        self.pre_model = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Linear(4096, 7)

    def forward(self, x):
        x = self.pre_model(x)
        x = self.classifier(x)

        return x


def Initial(model):

    model_vggface = VGG_Face_torch
    model_emotion = VGG_Net(model_vggface)
    model_emotion.load_state_dict(torch.load('best7164.model'))
    model_before_dict = model_emotion.state_dict()

    # model = FERANet()

    table_emotion = [0,2,5,7,10,12,14,17,19,21,24,26,28]
    print('start loading vgg16 pre-fera2013 model...')
    idx = 0
    idxx = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and idx < len(table_emotion):
            weight_name = "pre_model."+str(table_emotion[idx])+".weight"
            bias_name   = "pre_model."+str(table_emotion[idx])+".bias"
            assert m.weight.size() == model_before_dict[weight_name].size()
            assert m.bias.size() == model_before_dict[bias_name].size()
            m.weight.data = model_before_dict[weight_name]
            m.bias.data = model_before_dict[bias_name]
            idx = idx+1
        if isinstance(m, nn.Linear) and idxx < 1:
            weight_name = "pre_model.32.1.weight"
            bias_name   = "pre_model.32.1.bias"
            assert m.weight.size() == model_before_dict[weight_name].size()
            assert m.bias.size() == model_before_dict[bias_name].size()
            m.weight.data = model_before_dict[weight_name]
            m.bias.data = model_before_dict[bias_name]
            idxx = idxx+1

    print('finish loading vgg16 pre-fera2013 model!')
    return model



def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.5**(epoch // lr_decay_epoch))
    #     lr = init_lr #* 1 / (1 + )
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer


def cyclic_scheduler(optimizer,epoch):

    k = epoch // 25

    if 25*k < epoch <= 25*k+10 and k >=1 :
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001

    return optimizer