from tensorflow.keras.models import load_model
import glob
import os
import cv2
import pandas as pd
from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
from keras.models import load_model

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

def create_point(name):
    names = name.split('.')
    

    fin = open("./10areaans3/"+names[0]+".txt", 'r')
    x,y=fin.read().split(',')
    
    return int(x),int(y)

def Drawone(image,index,face_x,face_y):
    size = 23

    name = index.split('.')[0] + '.txt'
    img = image.copy()

    original = create_point(index)
    point = original
    radius = 1
    thickness = 4
    cv2.circle(img, point, radius,(255,0,0), thickness)
    cv2.circle(img, (face_x,face_y), radius,(0,255,0), thickness)
    cv2.rectangle(img, (face_x-size, face_y-size), (face_x + size, face_y + size), (0, 255, 0), 2)

    f = open('./distance3/'+name ,'w')
    f.write(str(point[0])+','+str(point[1])+'\n')
    f.write(str(face_x)+','+str(face_y))
    f.close()
    return img

Model = load_model('final10_2.h5',custom_objects={'euclidean_distance_loss':euclidean_distance_loss})
test_image = glob.glob('./train66/*'+'.jpg')

for img_path in test_image:
  index = os.path.basename(img_path)
  root = './finaloutput3/' + index
  img = cv2.imread(img_path)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  gray = cv2.resize(gray,(48,48),interpolation=cv2.INTER_AREA)
  array = img_to_array(gray)
  array = array.reshape((1,48,48,1))/255.0
  predict = Model.predict(array)
  x,y = int(predict[0,0]),int(predict[0,1])
  image = Drawone(img,index,x,y)
  cv2.imwrite(root,image)
