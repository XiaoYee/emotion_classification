import os
import os.path as osp
import argparse
import random

parser = argparse.ArgumentParser(description='Emotiw dataset list producer')

args = parser.parse_args()

train = "/home/quxiaoye/disk/FR/Emotiw2018/data/Train_AFEW_all/Emotiw-faces"
test  = "/home/quxiaoye/disk/FR/Emotiw2018/data/Val_AFEW/Emotiw-faces"

train_path = osp.join(train)
test_path  = osp.join(test)

Face_category = open("./TrainTestlist/Emotiw/Emotiw_TRAIN.txt","w")
Face_category_test = open("./TrainTestlist/Emotiw/Emotiw_VAL.txt","w")


train_img_folders = os.listdir(train_path)
train_img_folders.sort()

for i in range(len(train_img_folders)):
    path_folder = osp.join(train_path,train_img_folders[i])
    emotion_folders = os.listdir(path_folder)
    emotion_folders.sort()
    for emotion_folder in emotion_folders:
        path_write = osp.join(path_folder,emotion_folder)
        Face_category.write(path_write+" "+train_img_folders[i]+"\n")

Face_category.close()


test_img_folders = os.listdir(test_path)
test_img_folders.sort()

for i in range(len(test_img_folders)):
    path_folder = osp.join(test_path,test_img_folders[i])
    emotion_folders = os.listdir(path_folder)
    emotion_folders.sort()
    for emotion_folder in emotion_folders:
        path_write = osp.join(path_folder,emotion_folder)
        Face_category_test.write(path_write+" "+test_img_folders[i]+"\n")

Face_category_test.close()
  
