# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:26:14 2019
@author: matthew

This script calls the main functions for everything in the axon counting method development, training, and analysis.
These called functions are found in data.py and model.py, both in the src folder.
This script also plots the results without any linear corrections.
"""

import os, glob, datetime, data as data, model as model
import matplotlib.pyplot as pyp
import numpy as np
import PIL as pil
from scipy import stats
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback, LearningRateScheduler
import tensorflow as tf

#%% script setup
redoMover=False #if you want to unpackage the files from their source again
train=False #if you want to resume training
new=False   #if you want to start from a random initial parameter model
sourceparam=''

batch_size=1 #batch size is number of images predicted per training step. best at 1 per literature

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
batch_size=1



#define date string. use if making many models in short period of time
date = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
#define save name for model
mNameOld=r'saved models/final_resampled_3-22-2020.hdf5'


#set up training and validation dataset generators. see data.py.
print('Setting up dataset use...')
generator = data.trainGenerator(batch_size,train_path='data\\train'+sourceparam,image_folder='image',gt_folder='label',subset='training')
mName='new_model.hdf5'
mName=r'saved models/'+mName
valGen=data.valGenerator(sourceparam)
#%% model training
if new:
    Model=model.unet() #see model.py, initialize u-net architecture
    #Model=model.unet() #see model.py, initialize u-net architecture
else:
    Model=model.unet(mNameOld)

#set up keras callbacks- used in training call
model_checkpoint = ModelCheckpoint(mName, monitor='loss',verbose=2, save_best_only=True) #saves model if performance increases
earlystop=EarlyStopping(patience=40) #if no improvement in 40 iterations, stops training
reduceLRplat=ReduceLROnPlateau(factor=0.1, patience=20, verbose=1, cooldown=0, min_lr=0) #reduce lr on plateua
def LRscheduler(epoch):
  if epoch < 0:
    return 0.0001
  else:
    return 0.0001 * np.exp(0.2 * (0 - epoch))
LRschedule = LearningRateScheduler(LRscheduler)

#train the model
if train:
    Model.fit_generator(generator,steps_per_epoch=200,epochs=1*115,callbacks=[model_checkpoint],validation_data=valGen,validation_steps=20)
Model.save(mName) #saves current version of the model using defined model name

# %% plot training history
if train:
    history=Model.history
    pyp.plot(history.history['loss'][1:])
    pyp.plot(history.history['val_loss'][1:])
    pyp.title('model loss')
    pyp.ylabel('loss')
    pyp.xlabel('epoch')
    pyp.legend(['training loss', 'validation loss'], loc='upper left')
    pyp.show()
    

p=data.p
diff=[]
preds=[]
gts=[]

# %% Apply to training set as pre-test 

(outsT,names,predpix, gtpix)=data.evaluate_train(Model,sourceparam)

predpix=np.asarray(predpix)
gtpix=np.asarray(gtpix)
slope, intercept, r_value, p_value, std_err = stats.linregress(gtpix,predpix)
# %% Plot training pixel results
pyp.figure(figsize=(8,8))
pyp.ylim(0,100)
pyp.xlim(0,100)
pyp.scatter(outsT[0,:],outsT[1,:],20)
pyp.gca().set_aspect('equal', adjustable='box')
r=np.corrcoef(outsT[0,:],outsT[1,:])[0,1]
print('R2 value is ' + str(r*r))
slope, intercept, r_value, p_value, std_err = stats.linregress(outsT[0,:],outsT[1,:])
line = slope*outsT[0,:]+intercept
pyp.plot(outsT[0,:],line)
stng='y = '+str(slope)[:5]+'x + '+str(intercept)[:5]+'\nR2 = ' + str(r*r)[:5]
pyp.text(15,70,stng)

# %% Apply to validation set as pre-test   
(outs,names)=data.evaluate(Model,'val',sourceparam)

# %% Plot validation results
pyp.figure(figsize=(8,8))
pyp.ylim(0,100)
pyp.xlim(0,100)
pyp.scatter(outs[0,:],outs[1,:],20)
pyp.gca().set_aspect('equal', adjustable='box')
r=np.corrcoef(outs[0,:],outs[1,:])[0,1]
print('R2 value is ' + str(r*r))
slope, intercept, r_value, p_value, std_err = stats.linregress(outs[0,:],outs[1,:])
line = slope*outs[0,:]+intercept
pyp.plot(outs[0,:],line)
stng='y = '+str(slope)[:5]+'x + '+str(intercept)[:5]+'\nR2 = ' + str(r*r)[:5]
pyp.text(15,70,stng)
valOuts=np.transpose(outs)

# %% evaluate on final testing set of finalized model
if True:
    (outs2,names2)=data.evaluate(Model,'test',sourceparam)
    
    # %% Plot
    pyp.figure(figsize=(8,8))
    pyp.ylim(0,100)
    pyp.xlim(0,100)
    #outs2[1,:]=(outs2[1,:]-intercept)/slope
    pyp.scatter(outs2[0,:],outs2[1,:],20)
    pyp.gca().set_aspect('equal', adjustable='box')
    r=np.corrcoef(outs2[0,:],outs2[1,:])[0,1]
    print('R2 value is ' + str(r*r))
    slope, intercept, r_value, p_value, std_err = stats.linregress(outs2[0,:],outs2[1,:])
    line = slope*outs2[0,:]+intercept
    pyp.plot(outs2[0,:],line)
    stng='y = '+str(slope)[:5]+'x + '+str(intercept)[:5]+'\nR2 = ' + str(r*r)[:5]
    pyp.text(15,70,stng)
    #make vector of validation and testing set results
    #outsall=np.concatenate((np.transpose(outs),np.transpose(outs2)),1)
    outs=np.transpose(outs)    
    outs2=np.transpose(outs2)
    












