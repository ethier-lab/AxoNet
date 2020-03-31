# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:26:14 2019

@author: mritch3
"""

from model import *
from data import *
import data
import os, glob
from matplotlib.pyplot import *
import numpy as np
import skimage.transform as trans
import PIL as pil
from PIL import Image
from scipy import stats
import datetime
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras import backend as keras
import tensorflow as tf


redoMover=False

#get to root directory
os.chdir('..')
home=os.getcwd()

#%% Check for data.mat and download if needed
os.chdir('src')
import setup #used to download data.mat if needed 


#%% Check if data has been unloaded
os.chdir(home)
dataCheck=glob.glob('data/test/gt/*.npy')
if len(dataCheck)==0 or redoMover:
    os.chdir('src')
    import mover #used to download data if needed 

os.chdir(home)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size=1
n_reps=1






#%%Setting up dataset use
print('Setting up dataset use...')
date  = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
mName = 'saved models/final_resampled_3-22-2020.hdf5'
generator = trainGenerator(batch_size,n_reps)

#%% training setup
Model=unet(mName)
model_checkpoint = ModelCheckpoint(mName, monitor='loss',verbose=2, save_best_only=True)
earlystop=EarlyStopping(patience=40)
reduceLRplat=ReduceLROnPlateau(factor=0.1, patience=20, verbose=1, cooldown=0, min_lr=0)
#create validation generator for model evaluation at the end of each epoch
valGen=valGenerator()


#train
hist=Model.fit_generator(generator,steps_per_epoch=100,epochs=1000,callbacks=[model_checkpoint],validation_data=valGen,validation_steps=10) #validation_data=valGen
#Model.save(mName)
history=Model.history
plot(history.history['loss'])
plot(history.history['val_loss'])
title('model loss')
ylabel('loss')
xlabel('epoch')
legend(['train', 'test'], loc='upper left')
show()


# %% Test model use
(outs,names)=evaluate(Model,'val') # test on validation set


# %% Plot
figure(figsize=(8,8))
ylim(0,100)
xlim(0,100)
scatter(outs[0,:],outs[1,:],20)
gca().set_aspect('equal', adjustable='box')

r=np.corrcoef(outs[0,:],outs[1,:])[0,1]
print('R2 value is ' + str(r*r))

slope, intercept, r_value, p_value, std_err = stats.linregress(outs[0,:],outs[1,:])
line = slope*outs[0,:]+intercept
plot(outs[0,:],line)
stng='y = '+str(slope)[:5]+'x + '+str(intercept)[:5]+'\nR2 = ' + str(r*r)[:5]
text(15,70,stng)




#evaluate on final testing set if finalized model
if False:
    (outs2,names2)=evaluate(Model,'test')
    
    # %% Plot
    figure(figsize=(8,8))
    ylim(0,100)
    xlim(0,100)
    outs2[1,:]=(outs2[1,:]-intercept)/slope
    scatter(outs2[0,:],outs2[1,:],20)
    gca().set_aspect('equal', adjustable='box')
    
    r=np.corrcoef(outs2[0,:],outs2[1,:])[0,1]
    print('R2 value is ' + str(r*r))
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(outs2[0,:],outs2[1,:])
    line = slope*outs2[0,:]+intercept
    plot(outs2[0,:],line)
    stng='y = '+str(slope)[:5]+'x + '+str(intercept)[:5]+'\nR2 = ' + str(r*r)[:5]
    text(15,70,stng)
    
    outsall=np.concatenate((np.transpose(outs),np.transpose(outs2)),1)
    
    
    #find clostest fit
    diffs=outs2[1,:]-outs2[0,:] #pred minus gt
    minDiff=np.where(np.abs(diffs)==np.min(np.abs(diffs)))[0][0] #126, -31.79 off
    maxDiff=np.where(np.abs(diffs)==np.max(np.abs(diffs)))[0][0] #144, -.0047 off
    
    
    testFold='data\\test\\'
    imageFold='image\\'
    gtFold='gt\\'
    predFold='pred\\'
    
    
    imFiles=glob.glob(testFold+imageFold+'*')
    n=len(imFiles)
    orderVec=np.asarray(np.loadtxt(r'C:\Users\mritch3\Desktop\L2UBD\data\full_name_divisions.txt', dtype=str))
    imFiles=orderVec[-n:]
    i=0
    names=[]
    
    imFiles=imFiles[int(n/2):]   
    n=len(imFiles)
    outs=np.zeros((2,n))
    imFiles=imFiles[[minDiff,maxDiff]] #min first then max
    ims=[]
    for file in imFiles:
        name=os.path.split(file)[1]
        
        gt=np.load(testFold+gtFold+name+'.npy')
        #outs[0,i]=gt.sum().sum()
        gtsav=gt
        img=imChange(np.load(testFold+imageFold+file+'.npy'))
        (img,dummy)=mirrorer(img,gt)
        img=np.expand_dims(img,0)
        img=np.expand_dims(img,3)
        name=name[:len(name)-4]+'_predicted'
        
        pred=Model.predict(img)/1000
        pred=pred[0,p:-p,p:-p,0]
        img=img[0,p:-p,p:-p,0]
        #gt=255*gt/gt.max()
    
        #pred=(pred/.5)
        img=(img-img.min())
        img=255*img/img.max()
        img=np.expand_dims(img,2)
        pred=np.expand_dims(pred,2)
#        pred=pred/5
#        if i==0:
#            gt=gt*2.6
#            pred=pred/1.3
#        if i==1:
#            gt=gt*.5
        a=.003626
        
        
        
        pred=255*(pred/a)
        gt=255*(gt/a)
        
        
        
        
            
        gtim=pil.Image.fromarray(gt.astype('uint8'))
        gtim.save(str(i)+'image_gt.png')
        
        finIm=np.concatenate((img,img,img),2).astype('uint8')
        imshow(finIm)
        im=pil.Image.fromarray(finIm)
        im.save(str(i)+'image.png')
        
        predim=pred[:,:,0]
        predim=predim.astype('uint8')
        predim=pil.Image.fromarray(predim)
        predim.save(str(i)+'image_pred.png')
        i=i+1
    
    
    
#make color scalebar
h=500
w=40
array=np.zeros((h,w))
inc=255/h

val=255
for i in range(h):
    array[i,:]=val
    val=val-inc

im=pil.Image.fromarray(array.astype('uint8'))
im.save('scale bar.png')
















