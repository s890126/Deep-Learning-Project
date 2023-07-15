import tensorflow as tf
import keras
import numpy as np
from keras.models import Model, load_model
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.initializers import glorot_uniform
import keras.backend as K

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def Model3 (input_shape = (48, 48, 1), classes = 2):
      # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    
    X = Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),padding ='same')(X_input)
    X = BatchNormalization(axis=bn_axis)(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(X)
    
    X = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding = 'same')(X)
    X = BatchNormalization(axis=bn_axis)(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(X)
    

    X = Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding = 'same')(X)
    X = BatchNormalization(axis=bn_axis)(X)
    X = Activation("relu")(X)



    X = Flatten()(X)

    X = Dense(128,activation='relu')(X)

    X_output = Dense(classes)(X)
    X = BatchNormalization()(X)
    model = Model(X_input,X_output)
    model.compile(loss=euclidean_distance_loss, optimizer='adam')
    
    return model

import glob
import os
import cv2
import pandas as pd
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split  
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

def create_point(name):
    names = name.split('.')
    

    fin = open("./train_pos/"+names[0]+".txt", 'r')
    x,y=fin.read().split(',')


        
        

    return (int(x),int(y))

def train_X(image_dir):
    train_image = glob.glob(image_dir+ '.jpg')
    training = []
    
    for img_path in train_image:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(48,48),interpolation=cv2.INTER_AREA)
        img = img_to_array(gray)
        img = img/255
        training.append(img)
        
    return np.array(training)

def train_Y(image_dir):
    imgs = glob.glob(image_dir+ '.jpg')
    Y = []
    for i in imgs:
      idx = os.path.basename(i)
      original = create_point(idx)
      point = original
      Y.append(point)
    
    #df = pd.DataFrame(Y, index = index)     
    return np.array(Y)



X = train_X('./training/*')
print(X.shape)
y = train_Y('./training/*')
print(y.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2) 



model = Model3((48,48,1),2)
model.summary()

history = model.fit(X, y, epochs=300, validation_data=(X, y),
           batch_size=8,verbose=1)    
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

model.save('final10_2.h5')
