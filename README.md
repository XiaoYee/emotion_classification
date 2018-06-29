Emotion Recognition
==== 
the purpose of this repository is to explore facial expression recognition with deep learning

Environment:
====
python 2.7 
pytorch 0.3.0

Models:
====
1. VGG Face finetune for image-based expression recognition

2. VGG+GRU for video-based expression recognition

3. Resnet+GRU for video-based expression recognition

Models Detail:
====

1. VGG Face
____
We use [VGG face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) for finetune, FER2013 dataset for classification.
First of all, we convert caffe model to Pytorch model through [tool](https://github.com/fanq15/caffe_to_torch_to_pytorch)
The FER2013 database consists of 35889 images: 28709 for training, 3589 for public test, and 3589 for private test. 
We use training data and public test data for training, and evaluate the model performance with private test data.
Here we provide [processed FER2013 dataset](https://drive.google.com/drive/folders/1f17xgwvGaUpgXYBssocUNXDBgga-b3qp?usp=sharing)
and [pytorch model converted from caffe](https://drive.google.com/drive/folders/1f17xgwvGaUpgXYBssocUNXDBgga-b3qp?usp=sharing)




