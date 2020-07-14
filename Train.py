from time import time
import pickle
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow_core.python.eager.profiler_client import monitor


def create_cnn():
    with tf.device('/device:CPU:0'):
        model=Sequential()
        model.add(Convolution2D(64,(3,3),activation='relu', input_shape=(32,32,3)))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Convolution2D(128,(3,3),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Convolution2D(256,(3,3),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Convolution2D(1024,(2,2),activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(256,activation='relu'))
        model.add(Dense(128,activation='relu'))
        keras.layers.Dropout(0.3)
        model.add(Dense(256,activation='relu'))
        keras.layers.Dropout(0.2)
        # model.add(Dense(1024,activation='relu'))
        keras.layers.Dropout(0.3)
        model.add(Dense(10, activation='softmax'))
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    return model
#-----------------------------------------------------------------------------------------------------------------------
def load_data(batch_id):
    with open('preprocess_batch_' + str(batch_id)+'.p', mode='rb') as file:
        # note the encoding type is 'latin1'
        (features, labels)=pickle.load(file,encoding='latin1')
    return features,labels

#-----------------------------------------------------------------------------------------------------------------------
def load_validation_data():
    with open('preprocess_validation.p', mode='rb') as file:
        # note the encoding type is 'latin1'
        (features, labels)=pickle.load(file,encoding='latin1')
    return features,labels

#-----------------------------------------------------------------------------------------------------------------------
def combine_data():
    validation_f, validation_l = load_validation_data()
    all_features, all_lables = load_data(1)
    for i in range(4):
        features, lables = load_data(i + 2)
        all_features = np.append(all_features, features, axis=0)
        all_lables = np.append(all_lables, lables, axis=0)
    return all_lables,all_features,validation_f,validation_l

#-----------------------------------------------------------------------------------------------------------------------
# load data from dataset
all_lables,all_features,validation_f,validation_l = combine_data()

#create cnn
m=create_cnn()

# define the checkpoint
filepath = "cifar10.h5"
lrr = ReduceLROnPlateau(monitor='val_loss',
                        patience=2,
                        verbose=1,
                        factor=0.4,
                        min_lr=0.0001
                        )
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss',patience=3),
    tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min'),
    lrr
]

history=m.fit(all_features,all_lables,epochs=20,validation_data=(validation_f,validation_l),callbacks=my_callbacks)
m.save('cifar10.model')
print("Trained Finished Successfully!")


