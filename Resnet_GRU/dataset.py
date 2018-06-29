#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
from PIL import Image
from utils import *
from torchvision import transforms



class listDataset(Dataset):

    def __init__(self, root, length = None, shuffle=True,  train=False, dataset = None ,debug = False):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        if shuffle:
            random.shuffle(self.lines)

        self.nSamples  = len(self.lines[:10]) if debug  else len(self.lines)

        self.train = train
        self.length = length
        self.dataset = dataset


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()


        if self.train == True:
            img, label_index = self.load_data_label(imgpath)
            img   = torch.from_numpy(img).float()
            label_index = torch.LongTensor([label_index])
        else:
            img, label_index = self.load_test_data_label(imgpath)
            # img   = torch.from_numpy(img).float()
            label_index = torch.LongTensor([label_index])

        return (img ,label_index)


    def load_data_label(self,imgpath):

        classes = self.get_classes()

        seq = np.zeros((112, 96, 3, self.length), dtype=np.float32)

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        jitter = 0.2
        hue = 0.1
        saturation = 1.5 
        exposure = 1.5

        img_path = imgpath.split(" ")[0]
        label_path = imgpath.split(" ")[1]
        label_index = int(classes.index(label_path))

        video_length = len(os.listdir(img_path))
        imgs = os.listdir(img_path)
        imgs.sort()

        if video_length >= self.length:

            select_frame = sorted(random.sample(range(video_length), self.length))
            for m in range(self.length):
                img_file = os.path.join(img_path, imgs[select_frame[m]])
                img =  Image.open(img_file).convert('RGB')
                img = data_augmentation(img, (96,112), jitter, hue, saturation, exposure)
                img = np.array(img)

                seq[:, :, 0, m] = (img[:,:,0]/255.-mean[0])/std[0]
                seq[:, :, 1, m] = (img[:,:,1]/255.-mean[1])/std[1]
                seq[:, :, 2, m] = (img[:,:,2]/255.-mean[2])/std[2]

        else:
            for k in range(self.length):
                if k+1 <= video_length:
                    img_file = os.path.join(img_path,imgs[k])
                else:
                    img_file = os.path.join(img_path,imgs[video_length-1])

                img =  Image.open(img_file).convert('RGB')
                img = data_augmentation(img, (96,112), jitter, hue, saturation, exposure)
                img = np.array(img)

                seq[:, :, 0, k] = (img[:,:,0]/255.-mean[0])/std[0]
                seq[:, :, 1, k] = (img[:,:,1]/255.-mean[1])/std[1]
                seq[:, :, 2, k] = (img[:,:,2]/255.-mean[2])/std[2]

                
        data = np.transpose(seq, (3,2,0,1))
        return data ,label_index


    def load_test_data_label(self, imgpath, filter_size=16, stride=8):

        data_transforms = transforms.Compose([
            transforms.Resize((112,96)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        classes = self.get_classes()

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        img_path = imgpath.split(" ")[0]
        label_path = imgpath.split(" ")[1]
        label_index = int(classes.index(label_path))

        video_length = len(os.listdir(img_path))
        imgs = os.listdir(img_path)
        imgs.sort()

        output = []
        if video_length >= filter_size:
            for i in range(0, video_length, stride):
                if i + filter_size <= video_length:
                    inputs = []
                    frames_subset = imgs[i: i+filter_size]
                    for frame in frames_subset:
                        frame = os.path.join(img_path, frame)
                        img = Image.open(frame)
                        inputs.append(data_transforms(img).unsqueeze(0))
                    output_subset = torch.cat(inputs).unsqueeze(0)
                    output.append(output_subset)

        else:
            inputs = []
            for k in range(filter_size):
                if k+1 <= video_length:
                    frame = os.path.join(img_path, imgs[k])
                    img = Image.open(frame)
                    inputs.append(data_transforms(img).unsqueeze(0))
                else:
                    frame = os.path.join(img_path, imgs[video_length-1])
                    img = Image.open(frame)
                    inputs.append(data_transforms(img).unsqueeze(0))
            output_subset = torch.cat(inputs).unsqueeze(0)
            output.append(output_subset)

        output = torch.cat(output)

        return output, label_index
    

    def get_classes(self):
        classes = []
        for line in open('TrainTestlist/'+self.dataset+'/classInd.txt'):
            classes.append(line.strip().split()[1])
        return classes
