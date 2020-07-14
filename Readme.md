This project solved cfar10 challenge using Convolutional Neural Network. Any pool request is appreciated.
The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is in 5 batches.
1.	Preprocess.py
For the first step I concatenate all 5 batches to one. Then normalized images.(Convert all pixels between 0 and 1) and we apart 10% of images for validation. 
2.	Train.py
In this file we create a CNN model and train it by datasets and finally we save the model.

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 30, 30, 64)        1792      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 13, 13, 128)       73856     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 128)         0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 4, 4, 256)         295168    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 2, 2, 256)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 1, 1, 1024)        1049600   
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0         
_________________________________________________________________
dense (Dense)                (None, 256)               262400    
_________________________________________________________________
dense_1 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_2 (Dense)              (None, 256)               33024     
_________________________________________________________________
dense_3 (Dense)              (None, 10)                2570      
=================================================================
Total params: 1,751,306
Trainable params: 1,751,306
Non-trainable params: 0
3.	Test the model by trst data.

Result: the accuracy is 98% .
