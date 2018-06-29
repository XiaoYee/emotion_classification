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
1. `VGG Face`

We use [VGG face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) for finetune, FER2013 dataset for classification.
First of all, we convert caffe model to Pytorch model through [tool](https://github.com/fanq15/caffe_to_torch_to_pytorch),we provide converted [pytorch model](https://drive.google.com/drive/folders/1f17xgwvGaUpgXYBssocUNXDBgga-b3qp?usp=sharing)
The FER2013 database consists of 35889 images: 28709 for training, 3589 for public test, and 3589 for private test. 
We use training data and public test data for training, and evaluate the model performance with private test data.
Here we provide [processed FER2013 dataset](https://drive.google.com/drive/folders/1f17xgwvGaUpgXYBssocUNXDBgga-b3qp?usp=sharing)

Usage:
____
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

2. `VGG+GRU`

We propose this model especially for Emotiw audio-video challenge. It can also be used to handle other video classification problems, such as 
action classification, you can just change the folder as we do.

Usage:
____

First change getTraintest_Emotiw.py in `TrainTestlist` line 10~11 for video path. Your videos should look like this: 

```
Emotiw-faces
│  
└───Angry
│   
└───Disgust
    │   
    └───002138680 
        │ 
        └───images0001.jpg
```
If you don't trian seven classes, you need to change the `classInd.txt` to determine and `VGG_gru.py` fully connect number.

To run this code, you also need [pretrained model best7164.model](https://drive.google.com/drive/folders/1f17xgwvGaUpgXYBssocUNXDBgga-b3qp?usp=sharing)
you can also get this model by training on FER2013 dataset as above.

Then `python train.py`, you can begin to train. 

Performance:
____
all the performance train and test on Emotiw2018 train and validation(just 379 video) partition. 

Model        |VGG+LSTM(lr1e-4)| VGG+LSTM |  VGG+LSTM cell(dropout 0.8)| VGG+GRU | 
--------     | --------       | -------- |  --------                  |-------- | 
precision    | 47.76%         |  49.08%  |   49.87%                   |  51.19% |



3. `Resnet+GRU`

We propose this model also for Emotiw challenge. This resnet is not the original Resnet, but proposed initially for Face Recognition. So it has pretrained 
model in face dataset. It first use in [center loss](https://github.com/ydwen/caffe-face/blob/caffe-face/face_example/face_train_test.prototxt), but we 
adjust it for facial expression recognition by adding BN, conv dropout, and short connect.

To run this code, you also need [pretrained model best_resnet.model](https://drive.google.com/drive/folders/1f17xgwvGaUpgXYBssocUNXDBgga-b3qp?usp=sharing)
Then change getTraintest_Emotiw.py as `VGG+GRU`. Lastly `python train.py`, you can begin to train.

Performance:
____
Model          |Resnet+GRU(lr1e-4)| Resnet+GRU |  Resnet+GRU+conv drop |
--------       | --------         | --------   |  --------             | 
precision      | 46.97%           |  48.29%    |        50.13%         |


