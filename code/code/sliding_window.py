# import the necessary packages
from helpers3 import pyramid
from helpers3 import sliding_window
import glob
import os
import cv2
from tqdm import tqdm 

from keras.preprocessing.image import img_to_array
from  tensorflow.keras.models import load_model
import numpy as np
from keras import backend as K 

z=0


w=1400
h=1350

def Drawone(image,index,face_x,face_y):
    name = index.split('.')[0] + '.txt'
    name2 = index.split('.')[0]
    img = image.copy()
    crop_img1 = img[face_y:face_y+h, face_x:face_x+w]
    cv2.imwrite('./output_face3/3/'+name2+'.jpg', crop_img1)


    cv2.rectangle(img, (face_x,face_y), (face_x+w, face_y+h), (0,0,255), 2)

    f = open('./face_ans3/'+name ,'w')
    f.write(str(face_x)+','+str(face_y))

    f.close()
    
    return img

root = './test1/*'
images = glob.glob(root)
tbar = tqdm(images)
model = load_model('cnn_model1.h5')

for path in tbar:
    image = cv2.imread(path)
    index = os.path.basename(path)
    root = './output_face3/' + index
    (winW, winH) = (1400, 1350)
    image2 = image.copy()  
    face_x = 0

    face_y = 0

    z=0

	# loop over the sliding window for each stepSize of the image
    for (x, y, window) in sliding_window(image,stepSize=60, windowSize=(winW, winH)):
        
    		# if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
    		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
    		# WINDOW
    
        img = cv2.resize(window,(120,150))
        img = img_to_array(img)
        img = img.reshape(1, 120, 150, 3)
        
        img = img/255.0
        
        result = model.predict(img)
        b = np.array(result[0])
        
    
        if (b[4]>z):
            z=b[4]
            face_x = x
            face_y = y 

        # since we do not have a classifier, we'll just draw the window
        #clone =image2.copy()
        #cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        #cv2.imshow("Window", clone)
        #cv2.waitKey(1)
        #time.sleep(0.025)
    
    after = Drawone(image2,index,face_x,face_y)
    cv2.imwrite(root,after)

    #cv2.rectangle(image2, (face_x, face_y), (face_x + winW, face_y + winH), (0, 255, 0), 2)
    #cv2.imshow('Window',after)
    #cv2.waitKey()
    
K.clear_session()
