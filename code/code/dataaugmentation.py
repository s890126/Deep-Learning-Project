# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 21:27:59 2021

@author: c0096
"""
import cv2
import os
from pathlib import Path
import numpy as np
import glob

from tqdm import tqdm 
from skimage import exposure
import random


def equalize(img):
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    
    return img_rescale


def readImg(path):
    img = cv2.imread(path)
    img = img[:,:,::-1]
    return img

def readGray(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def create_point(name):
    names = name.split('.')

    fin = open("./train11/"+names[0]+".txt", 'r')
    body = fin.read().split("\n")
          
    points= []
    for i in range(19):
        points.append((int(body[i].split(',')[0]) , int(body[i].split(',')[1])))
    return points


def coordination(index):
    data = create_point(index)
    coordination = data
    x,y = coordination[9]
    return x,y

   

def main2():
    root = './training/'
    txt_root = './train_pos/'
    Imgs = glob.glob('./train11/*'+'.jpg')  
    Path(root).mkdir(parents=True, exist_ok=True)
    tbar = tqdm(Imgs)
    radius = 1
    thickness = 2
    for img_path in tbar:
        file = os.path.basename(img_path)
        name = file.split('.')[0]
        img = readGray(img_path)
        x,y = coordination(file)
        #cv2.circle(img,(x,y), radius,(255,0,0), thickness)
        for i in range(30):
            x1 = x - random.randint(-50,50)
            y1 = y - random.randint(-50,50)
            
            crop =  img[y1-120:y1+120,x1-120:x1+120]  
            cv2.imwrite(root+name+'-'+str(i)+'.jpg',crop)
            f = open(txt_root+name+'-'+str(i)+'.txt','w')
            point = [x-(x1-120),y-(y1-120)]
            f.write(str(point[0])+','+str(point[1]))
            f.close()
 

main2()


