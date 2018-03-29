
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
import pickle
import copy


# In[2]:


clf = pickle.load(open('class_model.sav', 'rb'))
def norm_01(data):
    #standardizind mean 0 variance 1
    norm=[]
    #print(data)
    for i in range(len(data)):
        cur=data[i,:]
        #cur=np.squeeze(cur)
        #print(cur)
        s_data= (cur - cur.mean()) / cur.std()
        #normalizing [0,1] min-max scaling
        s_data = (s_data - s_data.min()) / (s_data.max() - s_data.min())
        #print(s_data)
        norm.append(s_data)
    norm=np.asarray(norm)    
    return norm

def num_to_exer(num):
    ans="none"
    if(num==1):
        ans='arm curl'
    elif(num==2):
        ans='lunge'
    elif(num==3):
        ans='squat'
    elif(num==4):
        ans='push up'
    return ans

def discretize(classlabels):
    for i in range(0,len(classlabels)):
        #print(classlabels[i])
        if(classlabels[i]=='arm curl'):
            classlabels[i]=1
        elif(classlabels[i]=='lunge'):
            classlabels[i]=2
        elif(classlabels[i]=='squat'):
            classlabels[i]=3
        elif(classlabels[i]=='push up'):
            classlabels[i]=4
        else:
            print("error")
    return classlabels

def convert_to_points(data):
    moddata=[]
    #print(data.shape)
    for i in range(0,len(data)):
        cur=[]
        for j in range(0,data.shape[1],2):
            point=[]
            point.append(data[i][j])
            point.append(data[i][j+1])
            cur.append(point)
        moddata.append(cur)
    return moddata        

def calc_dist(data):
    newdata=[]
    for i in range(0,len(data)):
        dist=[]
        for j in range(0,len(data[i])-1):
            for k in range(j+1,len(data[i])):
                calc=((data[i][j][0]-data[i][k][0])**2 + (data[i][j][1]-data[i][k][1])**2)**(1/2.0)
                calc=calc*10000
                dist.append(calc)
        newdata.append(dist)
    return newdata   
     
def dist_point_plane(p0, p1, p2): 
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    nom = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denom = ((y2 - y1)**2 + (x2 - x1) ** 2) ** 0.5
    if(denom==0):
        result=((x0-x2)**2 + (y2-y0)**2)**(1/2.0)
    else:
        result = nom / denom
    result=result*10000    
    return result

def calc_plane_dist(data,svmfeeddata):
    for i in range(0,len(data)):
        for j in range(0,len(data[i])-1):
            for k in range(j+1,len(data[i])):
                    for m in range(0,len(data[i])):
                        d = dist_point_plane(data[i][j],data[i][k],data[i][m])
                        svmfeeddata[i].append(d)
    return svmfeeddata
   
def generate_matrix(data):
    val=data[...,1:]
    lab=data[...,0]
    val=np.asmatrix(val).astype(np.float)
    #print(val.shape,lab.shape)
    lab=discretize(lab)
    val=norm_01(val)
    print(val.shape)
    val=np.squeeze(val)
    print(val.shape)
    moddata=convert_to_points(val) 
    svmfeeddata=calc_dist(moddata)
    svmfeeddata=np.hstack((val,svmfeeddata)).tolist() #comment this line if you dont want to feed the points (x,y) to the SVM.
    svmfeeddata=calc_plane_dist(moddata,svmfeeddata) 
    svmfeeddata=np.asarray(svmfeeddata,float)
    lab=lab.astype('int')
    print(svmfeeddata.shape)
    svmfeeddata = np.nan_to_num(svmfeeddata)
    #svmfeeddata.eliminate_zeros()
    print(svmfeeddata.shape)
    print(np.any(np.isnan(svmfeeddata)))
    print(np.all(np.isfinite(svmfeeddata)))
    return svmfeeddata,lab                                                       


# In[3]:


from PIL import Image
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


# In[38]:


pu = Image.open("./data/pushup.png")
bg = Image.open("./data/background.png")
sq = Image.open("./data/squat.png")
lu = Image.open("./data/lunge.png")
ac = Image.open("./data/armcurl.png")
video_capture = cv2.VideoCapture('./data/videos/ltestn.mp4')
frameWidth = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#out = cv2.VideoWriter('./data/outputl.avi', -1, 10.0, (frameWidth,frameHeight),True)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('./data/output1.avi',fourcc, 8.0, (640, 360),True)
i=0
j=0
while True:
# Capture frame-by-frame
    ret, framereal = video_capture.read()
    if i%10==0:
        print(i,end=' ')
        start_time=time.time()
        #ret, framereal = video_capture.read()
        try:
            framereal = framereal.transpose(1,0,2)
            extra=framereal
            oriImg1 = framereal # B,G,R order
            #if pose!='push up':
            oriImg1 = oriImg1.transpose(1,0,2)
            oriImg = oriImg1
            param, model_params = config_reader()
            scale = 0.2
            imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])        
            input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels) 
            output_blobs = model.predict(input_img)
            heatmap = np.squeeze(output_blobs[1]) # output 1 is heatmaps
            heatmap = cv2.resize(heatmap, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            all_peaks = []
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
            testdata=[[]]
            for q in range(len(all_peaks)):
                testdata[0].append(all_peaks[q][0])
                testdata[0].append(all_peaks[q][1])
            testdata=np.array(testdata)
            print(testdata)
            '''rawdata=convert_to_points(testdata)
            raw1=[]
            for i in range(len(rawdata)):
                a=[int(rawdata[0][i][0]),int(rawdata[0][i][1])]
                raw1.append(tuple(a))
            '''#print(testdata,rawdata)
            colors = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
            cmap = matplotlib.cm.get_cmap('hsv')

            canvas = framereal # B,G,R order
            td=norm_01(testdata)
            print(td.shape)
            md=convert_to_points(td) 
            sd=calc_dist(md)
            sd=np.hstack((td,sd)).tolist() #comment this line if you dont want to feed the points (x,y) to the SVM.
            sd=calc_plane_dist(md,sd) 
            sd=np.asarray(sd,float)
            print(sd.shape)
            sd = np.nan_to_num(sd)
            #svmfeeddata.eliminate_zeros()
            print(np.any(np.isnan(sd)))
            print(np.all(np.isfinite(sd)))
            #print(svmfeeddata.shape)
            #svmfeeddata = np.nan_to_num(svmfeeddata)
            l=clf.predict(sd)

            if l==1:
                foreground = ac
            else:
                if l==2:
                    foreground = lu
                else:
                    if l==3:
                        foreground = sq 
                    else:
                        if l==4:
                            foreground = pu
                        else:
                            if l==5:
                                foreground = bg
            #print(fr)
            framereal = framereal.transpose(1,0,2)
            framereal=framereal/255
            framereal = Image.fromarray(np.uint8((framereal)*255))      
            framereal.paste(foreground, (0, 0), foreground)
            framereal = np.asarray(framereal)  
            #out.write(framereal)
            cv2.imshow('frame',framereal)
        except:
            framereal = framereal.transpose(1,0,2)
            framereal=framereal/255
            framereal = Image.fromarray(np.uint8((framereal)*255))      
            framereal.paste(bg, (0, 0),bg)
            framereal = np.asarray(framereal)  
            out.write(framereal)
            cv2.imshow('frame',framereal)
        #print('took '+ str(1000 * (time.time() - start_time))+' ms')
        
    i=i+1
    if ret==0:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


