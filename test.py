from time import time
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from PIL import Image

def load_test_data():
    with open('preprocess_test.p', mode='rb') as file:
        # note the encoding type is 'latin1'
        (features, labels)=pickle.load(file,encoding='latin1')
    return features,labels
#-----------------------------------------------------------------------------------------------------------------------
# load test data
test_features,test_labels = load_test_data()

#open trained Model
m=keras.models.load_model('cifar10.model')

#evaluate test data
score =m.evaluate(test_features,test_labels)

#print score
print(score)