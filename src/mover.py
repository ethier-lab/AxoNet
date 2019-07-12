# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:31:39 2019

@author: mritch3
"""

import os, glob, math
import numpy as np, scipy.stats as st
from scipy.io import loadmat
from skimage.filters import gaussian 
import scipy.misc
from PIL import Image
from skimage.transform import resize
import matplotlib as mp
p=.6 #training p


#x=loadmat(r'C:\Users\mritch3\Desktop\L2UBD\data\split_Ims_Gts_2.mat')
x=loadmat(r'C:\Users\mritch3\Desktop\L2UBD\data\data.mat')
c=x['c'][0,:]
ims=x['images'][0,:]
gt=x['gt'][0,:]
countTotals=x['countTotals'][0,:]
annotations=x['annotations'][0,:]

randVec=np.loadtxt(r'C:\Users\mritch3\Desktop\L2UBD\data\randvec.txt').astype(int)
#annotations=annotations[randVec]
nameList=[]


check=[]
check2=[]
ciOut=np.zeros((1474,5))
diffs=np.zeros((1474,4))
#%% make confidence intervals
for i in range(len(annotations)):
    a=annotations[i]
    a=np.sum(a,(0,1))
    if a.size>1:
        (low,high)=st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
        mean=np.mean(a)
        diffs[i,:]=a-mean
        stdev=np.std(a)
        ciOut[i,:]=np.asarray([low,high,c[i],mean,stdev])
    else:
        ciOut[i,:]=np.asarray([a,a,c[i],a,0])
        diffs[i,:]=0

## %% blur annotations, retaining same count
#for i in range(len(gt)):
#    check.append(sum(sum((gt[i]))))
#    gt[i]=gaussian(gt[i],sigma=8,preserve_range=True)
#    check2.append(sum(sum((gt[i]))))
#    if check[i]!=0:
#        gt[i]=gt[i]*(check[i]/check2[i])
#    check2[i]=(sum(sum((gt[i]))))



    
sze=192
# %% resize and rescale 
#for i in range(len(c)):
#    ims[i]=resize(ims[i],(sze,sze),anti_aliasing=False)
#    
#    s1=sum(sum(gt[i]))
#    gt[i]=resize(gt[i],(sze,sze),anti_aliasing=False)
#    s2=sum(sum(gt[i]))
#    if (s2!=0):
#        gt[i]=gt[i]*(s1/s2)


check=[]
# %% blur annotations, retaining same count
for i in range(len(gt)):
    check.append(sum(sum((gt[i]))))
    

# %% get means for all
meanIm=np.mean([x for x in ims])
sdevIm=np.std([x for x in ims])
meanGT=np.mean([x for x in gt])
sdevGt=np.std([x for x in gt])
maxGT=np.max([x for x in gt])

    
# %% deposit all
print('deposit all')
count=[0,0,0]

#chdir+delete
os.chdir(r'C:\Users\mritch3\Desktop\L2UBD\data\all')
files = glob.glob('*')
for f in files:
    os.remove(f)


for i in range(len(c)):
    if c[i]==1:
        first='y-rats'
        count[0]=count[0]+1
    elif  c[i]==2:
        first='g-rats'
        count[1]=count[1]+1
    elif  c[i]==3:
        first='b-rats'
        count[2]=count[2]+1
        
    name=first+"_%03d"%count[c[i]-1]
    nameList.append(name)
    
    im_this=ims[i]
    gt_this=gt[i]
    
        
    #np.save(name+'_image' + '.npy', im_this)
    #np.save(name+'_gt'+ '.npy', gt_this)
    
    print(i/len(c))

# %% randomize along previously used vec
if True:
    
    c=c[randVec]
    ims=ims[randVec]
    gt=gt[randVec]
    nameList = [nameList[i] for i in randVec]
    ciOut=ciOut[randVec,:]
    countTotals=countTotals[randVec]
    annotations=annotations[randVec]

check=[]
# %% blur annotations, retaining same count
for i in range(len(gt)):
    check.append(sum(sum((gt[i]))))
    

# %% deposit test/train split

#set up split data
n=len(c)
trainC=c[:int(n*p)]
trainIms=ims[:int(n*p)]
trainGt=gt[:int(n*p)]
trainNameList=[nameList[i] for i in range(int(n*p))]
testC=c[int(n*p):]
testIms=ims[int(n*p):]
testGt=gt[int(n*p):]
testNameList=[nameList[i+int(n*p)] for i in range(int(1+n*(1-p)))]

# %%depost train
print('deposit train')
os.chdir('..')
os.chdir('train')
files = glob.glob('image/*')
for f in files:
    os.remove(f)
files = glob.glob('label/*')
for f in files:
    os.remove(f)
    
for i in range(len(trainC)):
            
    name=trainNameList[i]
    im_this=trainIms[i]
    gt_this=trainGt[i]
    
    np.save('image\\' + name + '.npy',im_this)
    np.save('label\\'+name + '.npy', gt_this)
    #im = Image.fromarray(A)
    #im.save("your_file.jpeg")
    print(i/len(trainC))

# %%depost test
print('deposit test')
os.chdir('..')
os.chdir('test')    
files = glob.glob('image/*')
for f in files:
    os.remove(f)
files = glob.glob('gt/*')
for f in files:
    os.remove(f)
files = glob.glob('pred/*')
for f in files:
    os.remove(f)
    
for i in range(len(testC)):
            
    name=testNameList[i]
    im_this=testIms[i]
    gt_this=testGt[i]
    name='test_' + str('%03d' % i)
    np.save('image\\' + name + '.npy',im_this)
    np.save('gt\\'+name + '.npy' , gt_this)    
    print(i/len(testC))
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    