# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:31:39 2019

@author: mritch3
"""

import os, glob, pandas
import numpy as np, scipy.stats as st
from scipy.io import loadmat
from PIL import Image

destparam=''
resample=True
np.random.seed(1)
p=.6 #proportion in training set. The rest is split 50%-50% between validate/test
sze=192 #common dataset image size



os.chdir('..') #get to root directory
root=os.getcwd()

x=loadmat(root + r'\data\data_updated_3-22-2020.mat')
c=x['c'][0,:]
ims=x['images'][0,:]
gt=x['gt'][0,:]
countTotals=x['countTotals'][0,:]
annotations=x['annotations'][0,:]

randVec=np.loadtxt(r'data/randvec.txt').astype(int)
#annotations=annotations[randVec]
nameList=[]


for i in range(len(gt)):
    gt[i][gt[i]<0]=0
    

#%% make new assignments preserving animals exclusively in one dataset or another
#
##get source info
IDs = pandas.read_excel(r'data\nerveIDs_forcode.xlsx')
names = np.array(IDs)[:,0].astype(str)
IDs = np.array(IDs)[:,1].astype(int)

number=np.zeros(26)
IDset=np.arange(1,27)
for i in np.arange(0,26):
    number[i]=np.sum(IDs==(i+1)) #contains number of training images per source nerve
    

# %% get the names for all, used to segregate animals by subset
print('deposit all')
count=[0,0,0]

#chdir to data folder + delete files already there
os.chdir(root + r'\data\all')
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
    #these next two lines print all of the images. comment to skip this, is not needed unless you are looking at the full dataset for some analysis
    #np.save(name+'_image' + '.npy', im_this)
    #np.save(name+'_gt'+ '.npy', gt_this)
    

# assign nerves with proportionate representation
testlist=[]
trainlist=[]
vallist=[]
#do Y
vec=np.asarray([23,22,21,20])
multi=[0,0,1,1]
choice=np.random.permutation([0,1])

trainlist.append(vec[multi==choice[0]][0])
trainlist.append(vec[multi==choice[0]][1])
vallist.append(vec[multi==choice[1]][0])
testlist.append(vec[multi==choice[1]][1])
#do G
vec=np.asarray([19,18,17,16,15,14,13,12])
multi=[0,0,1,1,2,2,3,3]
choice=np.random.permutation([0,1,2,3])

trainlist.append(vec[multi==choice[0]][0])
trainlist.append(vec[multi==choice[0]][1])
trainlist.append(vec[multi==choice[1]][0])
trainlist.append(vec[multi==choice[1]][1])

vallist.append(vec[multi==choice[2]][0])
vallist.append(vec[multi==choice[2]][1])

testlist.append(vec[multi==choice[3]][0])
testlist.append(vec[multi==choice[3]][1])

#do Bs
vec=np.asarray([0,1,2,3,4,5,6,7,8,9,10,11]) #these are all the options
multi=[0,1,1,2,2,3,3,4,4,0,5,5] #these are the indices as origin image (same length as vec)
choice=np.random.permutation([0,1,2,3,4,5]) #mix up the animals with both nerves
#give 4 animals to train
for i in range(4):
    trainlist.append(vec[multi==choice[i]][0])
    trainlist.append(vec[multi==choice[i]][1])
#give remaining to val + test to even out
vallist.append(vec[multi==choice[4]][0])
vallist.append(vec[multi==choice[4]][1])
testlist.append(vec[multi==choice[5]][0])
testlist.append(vec[multi==choice[5]][1])

#add b21 OS and OD (leftover cases) to validation set to even out counts. added in as zero-case.
vallist.append(24) #od labeled as 25 
vallist.append(25) #os labeled as 26 




#%% get vectors for each
labs=[]
for i in range(len(IDs)):
    if IDs[i]-1 in trainlist:
        labs.append('train')
    if IDs[i]-1 in vallist:
        labs.append('validate')
    if IDs[i]-1 in testlist:
        labs.append('test')
        
labs=np.asarray(labs)
# count the number of each images in each set
countTrain=sum(labs=='train')
countVal=sum(labs=='validate')
countTes=sum(labs=='test')


#%% make confidence intervals from manual counts
#initialize list to ensure counts are not changing
check=[]
ciOut=np.zeros((len(IDs),5))
diffs=np.zeros((len(IDs),4))
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
   
    
# %% deposit test/train/validate split
nameList=np.asarray(nameList)
#set up split data
where=labs=='train'
n=len(c)
trainC=c[where]
trainIms=ims[where]
trainGt=gt[where]
trainNameList=nameList[where] 

where=labs=='validate'
valC=c[where]
valIms=ims[where]
valGt=gt[where]
valNameList=nameList[where] 

where=labs=='test'
testC=c[where]
testIms=ims[where]
testGt=gt[where]
testNameList=nameList[where]

#%%resample training set for even distribution
if resample:
    counts=[]
    for i in range(len(trainGt)):
        counts.append(np.sum(trainGt[i]))
    counts=np.sort(np.asarray(counts))
    maxim=np.max(counts)
    minim=np.min(counts)
    numBins=10
    diff=(maxim-minim)/numBins
    indices=np.asarray([]).astype(int)
    nBins=np.histogram(counts,numBins)[0]
    locs=np.histogram(counts,numBins)[1]
    #hardcoded for my use
    reps=np.asarray([2,1,1,1,1,1,1,2,3,4])
    #reps=np.asarray([3,2,2,1,1,1,1,3,4,5])
    for i in range(numBins):
        find=(counts>=(locs[i]))&(counts<=(locs[i+1]))
        for j in range(reps[i]):
            indices=np.append(indices, np.where(find))
            
    trainC=trainC[indices]
    trainIms=trainIms[indices]
    trainGt=trainGt[indices]
    trainNameList=trainNameList[indices] 


# %%depost train
os.chdir('..')
os.chdir('train'+str(destparam))
files = glob.glob('image/*')
#remove every file in folder
for f in files:
    os.remove(f)
files = glob.glob('label/*')
#delete files there already
for f in files:
    os.remove(f)
#print
for i in range(len(trainC)):
    name=trainNameList[i]
    im_this=trainIms[i]
    gt_this=trainGt[i]
    #write image and gts in .npy format
    np.save('image\\' + name +'_' + str(i) + '.npy',im_this)
    np.save('label\\'+name +'_' + str(i) + '.npy', gt_this)
    #update user
    print(50*'\n')
    print('depositing training set')
    print(i/len(trainC))
    
# %%depost validate
os.chdir('..')
os.chdir('validation'+str(destparam))    
#delete files there already
files = glob.glob('image/*')
for f in files:
    os.remove(f)
files = glob.glob('gt/*')
for f in files:
    os.remove(f)
files = glob.glob('pred/*')
for f in files:
    os.remove(f)
files = glob.glob('fintest/*')
for f in files:
    os.remove(f)
#print
for i in range(len(valC)):
    name=valNameList[i]
    im_this=valIms[i]
    gt_this=valGt[i]
    name='validation_' + str('%03d' % i)
    #save np arrays
    #write image and gts in .npy format
    np.save('image\\' + name + '.npy',im_this)
    np.save('gt\\'+name + '.npy' , gt_this)
    
    #do image
    im = Image.fromarray(im_this)
    im.save('fintest/x'+str(i)+'y0.png',  format='png', subsampling=0, quality=100, compress_level=0)
    im.close()
    
    print(50*'\n')   
    print('depositing validation set')
    print(i/len(valC))
    
# %%depost test
os.chdir('..')
os.chdir('test'+str(destparam))    
#os.mkdir('fintest')
#delete files there already
files = glob.glob('image/*')
for f in files:
    os.remove(f)
files = glob.glob('gt/*')
for f in files:
    os.remove(f)
files = glob.glob('pred/*')
for f in files:
    os.remove(f)
#print
for i in range(len(testC)):
    name=testNameList[i]
    im_this=testIms[i]
    gt_this=testGt[i]
    name='test_' + str('%03d' % i)
    #write image and gts in .npy format
    #save np arrays
    np.save('image\\' + name + '.npy',im_this)
    np.save('gt\\'+name + '.npy' , gt_this)
    
    #do image
    im = Image.fromarray(im_this)
    im.save('fintest/x'+str(i)+'y0.png',  format='png', subsampling=0, quality=100, compress_level=0)
    im.close()
    
    print(50*'\n')   
    print('depositing testing set')
    print(i/len(testC))
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    