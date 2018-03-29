
# coding: utf-8

# In[1]:


import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from config_reader import config_reader
import scipy
import math
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import matplotlib
import pylab as plt
import numpy as np
import util
import time
import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import accuracy_score
from scipy.ndimage.filters import gaussian_filter
import sys
from tkinter import *
from tkinter import messagebox


# Helper functions to create a model

# In[2]:


def relu(x): 
    return Activation('relu')(x)

def conv(x, nf, ks, name):
    x1 = Conv2D(nf, (ks, ks), padding='same', name=name)(x)
    return x1

def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x

def vgg_block(x):
     
    # Block 1
    x = conv(x, 64, 3, "conv1_1")
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2")
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")

    # Block 2
    x = conv(x, 128, 3, "conv2_1")
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2")
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1")
    
    # Block 3
    x = conv(x, 256, 3, "conv3_1")
    x = relu(x)    
    x = conv(x, 256, 3, "conv3_2")
    x = relu(x)    
    x = conv(x, 256, 3, "conv3_3")
    x = relu(x)    
    x = conv(x, 256, 3, "conv3_4")
    x = relu(x)    
    x = pooling(x, 2, 2, "pool3_1")
    
    # Block 4
    x = conv(x, 512, 3, "conv4_1")
    x = relu(x)    
    x = conv(x, 512, 3, "conv4_2")
    x = relu(x)    
    
    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM")
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM")
    x = relu(x)
    
    return x

def stage1_block(x, num_p, branch):
    
    # Block 1        
    x = conv(x, 128, 3, "conv5_1_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 128, 3, "conv5_2_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 128, 3, "conv5_3_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 512, 1, "conv5_4_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, num_p, 1, "conv5_5_CPM_L%d" % branch)
    
    return x

def stageT_block(x, num_p, stage, branch):
        
    # Block 1        
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch))
    
    return x


# Create keras model and load weights

# In[ ]:


weights_path = "model/keras/model.h5" # orginal weights converted from caffe
#weights_path = "training/weights.best.h5" # weights tarined from scratch 

input_shape = (None,None,3)

img_input = Input(shape=input_shape)

stages = 6
np_branch1 = 38
np_branch2 = 19

img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

# VGG
stage0_out = vgg_block(img_normalized)

# stage 1
stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1)
stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2)
x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

# stage t >= 2
for sn in range(2, stages + 1):
    stageT_branch1_out = stageT_block(x, np_branch1, sn, 1)
    stageT_branch2_out = stageT_block(x, np_branch2, sn, 2)
    if (sn < stages):
        x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

model = Model(img_input, [stageT_branch1_out, stageT_branch2_out])
model.load_weights(weights_path)


# Load a sample image

# In[ ]:


dataintake=[]
label=[]
start_time=time.time()
counter=0
print('start '+ str(1000 * (time.time() - start_time))+' ms')
for file in os.listdir('./data/videos'):
    cap = cv2.VideoCapture('./data/videos/'+file)
    #frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    i=0
    imname=1
    if file[0:1] == 'a':
        pose='arm curl'
    elif file[0:1] == 's':
        pose='squat'
    elif file[0:1] == 'l':
        pose='lunge'
    elif file[0:1] == 'p':
        pose='push up'
    while(True):
        ret, framereal = cap.read()
        if counter%15==0:
            try:
                extra=framereal
                oriImg1 = framereal # B,G,R order
                if pose!='push up':
                    oriImg1 = oriImg1.transpose(1,0,2)
                oriImg = oriImg1
                param, model_params = config_reader()
                scale = 0.8
                imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])        
                input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels) 
                output_blobs = model.predict(input_img)
                heatmap = np.squeeze(output_blobs[1]) # output 1 is heatmaps
                heatmap = cv2.resize(heatmap, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
                heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
                heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
                all_peaks = []
                all_peaks.append(str(imname)+'.jpg')
                all_peaks.append(pose)
                peak_counter = 0
                for part in range(19-1):
                    map_ori = heatmap[:,:,part]
                    map = gaussian_filter(map_ori, sigma=3)
                    map_left = np.zeros(map.shape)
                    map_left[1:,:] = map[:-1,:]
                    map_right = np.zeros(map.shape)
                    map_right[:-1,:] = map[1:,:]
                    map_up = np.zeros(map.shape)
                    map_up[:,1:] = map[:,:-1]
                    map_down = np.zeros(map.shape)
                    map_down[:,:-1] = map[:,1:]
                    peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
                    peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
                    if peaks==[]:
                        l=[0,0]
                        all_peaks.append(l)
                    else:
                        l=[peaks[0][0],peaks[0][1]]
                        all_peaks.append(l)
                label.append(pose)
                dataintake.append(all_peaks)
                print(file,counter,all_peaks,label[-1])
                fig.savefig('./data/newframes/'+imname+'.jpg')
                imname=imname+1
            except:
                print('error image')
        counter=counter+1
        if ret==0:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
print('stop '+ str(1000 * (time.time() - start_time))+' ms')
import pickle
pickle.dump( dataintake, open( "savedata1.p", "wb" ) )
pickle.dump( label, open( "savelabel1.p", "wb" ) )


# In[11]:


print(dataintake.shape,len(label))


# Heatmap for right knee. Note that the body part is encoded in the 3th channel so in this case right knee is 
# at index 9. All body parts are defined in config: 
# part_str = [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]

# In[5]:



x = pickle.load( open( "savedata.p", "rb" ) )
print(x)

