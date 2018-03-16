# -*- coding: utf-8 -*-
"""
@author: karthik
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import scipy
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import accuracy_score

opts = {'rdir': 'C:/Users/karthik/Desktop/mlabs_main/w2s1/frames/',
        'refSize' : [80,80,3]}

def num_to_exer(num):
    ans="none"
    if(num==1):
        ans='armcurl'
    elif(num==2):
        ans='lunge'
    elif(num==3):
        ans='squat'
    elif(num==4):
        ans='pushup'
    return ans

def train_test_split(fdata,trainSize):
    np.random.shuffle(fdata)
    splitLoc = np.floor(trainSize*len(fdata))
    splitLoc = splitLoc.astype(int)
    #print(splitLoc)
    # create the random split
    ftrain = fdata[0:splitLoc,:]
    ftest = fdata[splitLoc+1:,:]
    return ftrain, ftest 
        
def readData():
#    cwd=os.getcwd()
    getimage = None
    os.chdir("C:/Users/karthik/Desktop/mlabs_main/w2s1")
    file= 'data.csv'
    x1 = pd.read_csv(file)
    #x1=pd.ExcelFile(file)
    #df1=x1.parse('Sheet1')
    #print(df1)
    #print(x1)
    npdata=x1.as_matrix()
    #print(npdata)
    loc=npdata[:,0]
    for i in range(len(loc)):
        dirname = opts['rdir']+loc[i]
        #print(dirname)
        if(not(dirname.endswith('.jpg'))):  
            dirname = dirname+'.jpg'
        # read Image
        try:
            #print("hello")
            img = np.asarray(Image.open(dirname))
            img = scipy.misc.imresize(img,(opts['refSize']))     
            # collapse into vector
            feat = np.reshape(img,(1,np.prod(opts['refSize'])))
            #print(feat)
            # append to dataset
            if getimage is None:
                getimage = feat
                #print(getimage)
            else:
                getimage = np.vstack((getimage,feat))
        except FileNotFoundError:
            print(dirname)
            print("fetch error")
    
    return npdata,getimage

def norm_01(data):
    #standardizind mean 0 variance 1
    norm=[]
    for i in range(len(data)):
        cur=data[i,:]
        s_data= (cur - cur.mean()) / cur.std()
        #normalizing [0,1] min-max scaling
        s_data = (s_data - s_data.min()) / (s_data.max() - s_data.min())
        norm.append(s_data)
    norm=np.asarray(norm)    
    return norm
    
def discretize(classlabels):
    for i in range(0,len(classlabels)):
        #print(classlabels[i])
        if(classlabels[i]=='armcurl'):
            classlabels[i]=1
        elif(classlabels[i]=='lunge'):
            classlabels[i]=2
        elif(classlabels[i]=='squat'):
            classlabels[i]=3
        elif(classlabels[i]=='pushup'):
            classlabels[i]=4
        else:
            print("error")
    return classlabels

def convert_to_points(data):
    moddata=[]
    
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
    val=data[...,2:]
    lab=data[...,1]
    lab=discretize(lab)
    val=norm_01(val)
    moddata=convert_to_points(val) 
    svmfeeddata=calc_dist(moddata)
    #svmfeeddata=np.hstack((val,svmfeeddata)).tolist() #comment this line if you dont want to feed the points (x,y) to the SVM.
    svmfeeddata=calc_plane_dist(moddata,svmfeeddata) 
    svmfeeddata=np.asarray(svmfeeddata,float)
    lab=lab.astype('int')
    return svmfeeddata,lab                                                       




try:
    #np.set_printoptions(threshold=np.nan)
    data, imgdata=readData()
    print("reading complete")
    dataview=data.tolist()
    kf = KFold(n_splits=4,shuffle=True)
    conf_mat=np.full((4,4),0)
    scores=[]
    for train_index, test_index in kf.split(data):
        ftrain, ftest = data[train_index], data[test_index]
        itrain,itest = imgdata[train_index], imgdata[test_index]
        itrain,itest = itrain[:,2:],itest[:,2:]
        itrain,itest = norm_01(itrain),norm_01(itest)
#    ftrain,ftest = train_test_split(data,0.7)
        train_d,train_l=generate_matrix(ftrain)
        train_d=np.hstack((train_d,itrain))
        test_d,test_l=generate_matrix(ftest)
        test_d=np.hstack((test_d,itest))
        clf = svm.LinearSVC(C=3.5)
#        clf = svm.SVC(decision_function_shape='ovo',tol=0.00001,degree=3)
        clf.fit(train_d, train_l)
        
        ftrain_pred=clf.predict(train_d)
        ftest_pred=clf.predict(test_d)
        cur_score=accuracy_score(test_l,ftest_pred, normalize=True)
        scores.append(cur_score)
        train_score=accuracy_score(train_l,ftrain_pred,normalize=True)
        print(train_score,cur_score)
        
        for j in range(len(test_l)):
            if(ftest_pred[j]!=test_l[j]):
                print("misclassified - ",ftest[j,0]," where ",num_to_exer(ftest[j,1])," as ",num_to_exer(ftest_pred[j]))
                
        cur_matrix=confusion_matrix(test_l,ftest_pred)
        cur_matrix=np.asarray(cur_matrix)
        conf_mat=np.add(conf_mat, cur_matrix)
    scores=np.asarray(scores)    
    print(scores.mean())
    print(conf_mat)           
except:
    print("error")     
    
    