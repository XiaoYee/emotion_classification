Emotion Recognition
==== 
the purpose of this repository is to explore facial expression recognition with deep learning

Environment:
====
python 2.7 
pytorch 0.3.0
GTX 1080

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
First of all, we convert caffe model to Pytorch model through [tool](https://github.com/fanq15/caffe_to_torch_to_pytorch),we provide converted [pytorch model](https://drive.google.com/drive/folders/1f17xgwvGaUpgXYBssocUNXDBgga-b3qp?usp=sharing)
The FER2013 database consists of 35889 images: 28709 for training, 3589 for public test, and 3589 for private test. 
We use training data and public test data for training, and evaluate the model performance with private test data.
Here we provide [processed FER2013 dataset](https://drive.google.com/drive/folders/1f17xgwvGaUpgXYBssocUNXDBgga-b3qp?usp=sharing)

        Usage:
First download FER2013 dataset(need to unzip) and pytorch model(VGG_Face_torch.pth), save to folder `VGG_Finetune`.
Your file looks like this:
```
VGG_Finetune
│   train.py
│   VGG_Face_torch.py
│   VGG_Face_torch.pth 
│
└───train
│   │   0
│   │   1
│   
└───test
    │   0
    │   1
```
        Then `python train.py`,you can begin to train. Please note if you have memory problem, reduce batchsize.
        You can also finetune your own data, 0~6 in train and test mean different expressions. You can also use to train other classification problem.
        After you train and eval, you should get `71.64%` precision.

2. VGG+GRU
____


