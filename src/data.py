# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:26:52 2019

@author: mritch3
"""
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os, glob
import skimage.io as io
import skimage.transform as trans
import matplotlib as mp
from PIL import Image
from skimage import exposure
import math

mult=1000
p=16


def imChange(im):
    '''
    conducts normalization by image.
    currently results in [-1,1] range
    '''
    sat=1.00
    im=im/np.max(im)
    im=np.clip(im*sat,0,1)
    im=im/np.max(im)
    im=2*(im-np.mean(im))
    return im
        
def trainGenerator(batch_size,n_reps,train_path='data\\train',image_folder='image',gt_folder='label',subset='training'):
    '''
    generates input images and masks to be fed to model trainer function
    
    #todo: remove nreps, re-randomize after all images have been used
    '''
    
    #get filenames
    imFiles=glob.glob(train_path+'\\'+image_folder+'\\*')
    gtFiles=glob.glob(train_path+'\\'+gt_folder+'\\*')
    n=len(imFiles)
    rVec=np.random.permutation(n)
    ##randomize and make list longer
    imFiles = [imFiles[i] for i in rVec] 
    gtFiles = [gtFiles[i] for i in rVec] 
    if n_reps>1:
        for i in range(n_reps):
            rVec=np.random.permutation(n)
            imFiles = imFiles+ [imFiles[i] for j in rVec]
            gtFiles = gtFiles+ [gtFiles[i] for j in rVec]
            
    nOutputs=math.floor(n*n_reps/batch_size)
    i=0
    while (True) :
       
       #load first
       img=np.load(imFiles[i*batch_size])
       gt=np.load(gtFiles[i*batch_size])
       (img,gt)=mirrorer(img,gt)
       img=np.expand_dims(img,0)
       img=np.expand_dims(img,3)
       gt=np.expand_dims(gt,0)
       gt=np.expand_dims(gt,3)
       #load others
       if batch_size>1:
           for j in range(batch_size-1):
               imgNew=np.load(imFiles[i*batch_size+j+1])
               gtNew=np.load(gtFiles[i*batch_size+j+1])
               (imgNew,gtNew)=mirrorer(imgNew,gtNew)
               
               imgNew=np.expand_dims(imgNew,0)
               imgNew=np.expand_dims(imgNew,3)
               
               gtNew=np.expand_dims(gtNew,0)
               gtNew=np.expand_dims(gtNew,3)
               img=np.concatenate((img,imgNew),axis=0)
               gt=np.concatenate((gt,gtNew),axis=0)
       #augment
       (img,gt)=randAug(img,gt)
       
       if i==nOutputs-1:
           i=0
           imFiles = [imFiles[i] for i in rVec] 
           gtFiles = [gtFiles[i] for i in rVec] 
       else:
           i=i+1
               
       yield (imChange(img),mult*gt)
        
        
def randAug(img,gt):
    '''
    augments image and mask at the same time
    currently: 
        mirrors with P=0.5
        rotates by 90 degrees with a P=.25 for each orientation
        multiplies image intensities by a random factor in range [-0.15, 0.15]
    '''
    flip=np.random.rand(1)>.5
    rot=math.floor(4.0*np.random.rand(1))
    
    if flip:
        img=np.flip(img, 1)
        gt =np.flip(gt,  1)
        
    img=np.rot90(img,rot,axes=(1, 2))    
    gt =np.rot90(gt, rot,axes=(1, 2))
    
    imshift=1+(.3*np.random.rand(1)-.15)
    img=img*imshift
    
    return img,gt

def mirrorer(image,mask,p=p):
    '''
    pads image sides by mirroring p pixels from the edges
    '''
    #do for image
    ax=image.shape
    top=image[:p,:]
    bot=image[(ax[0]-p):,:]
    image=np.concatenate((top[::-1,:], image, bot[::-1,:]), axis=0, out=None)
    left=image[:,:p]
    right=image[:,(ax[1]-p):]
    image=np.concatenate((left[:,::-1], image, right[:,::-1]), axis=1, out=None)
    mirroredIm=image
    #now do for gt
    image=mask
    ax=image.shape
    top=image[:p,:]
    bot=image[(ax[0]-p):,:]
    image=np.concatenate((top[::-1,:], image, bot[::-1,:]), axis=0, out=None)
    left=image[:,:p]
    right=image[:,(ax[1]-p):]
    image=np.concatenate((left[:,::-1], image, right[:,::-1]), axis=1, out=None)
    mirroredMask=image
    return mirroredIm,mirroredMask
    


def valGenerator(testFold='data\\test\\', imageFold='image\\', gtFold='gt\\'):
    '''
    generates input images and masks to be fed to model validation checkpoints
    '''   
    imFiles=glob.glob(testFold+imageFold+'*')
    n=len(imFiles)    
    #get only the val set
    imFiles=imFiles[:int(n/2)]
    n=len(imFiles)   
    ##randomize
    i=0
    rVec=np.random.permutation(n)
    imFiles = [imFiles[j] for j in rVec]  
    while True:
        file=imFiles[i]
        name=os.path.split(file)[1]
        gt=np.load(testFold+gtFold+name)*mult
        img=imChange(np.load(file))
        (img,gt)=mirrorer(img,gt)
        img = np.array(img)[np.newaxis, : , :, np.newaxis]     
        gt = np.array(gt)[np.newaxis, : , :, np.newaxis]  
        
        if i==n-1: #reset generator
            i=0
            rVec=np.random.permutation(n)
            imFiles = [imFiles[j] for j in rVec]  
            
        yield img, gt


def evaluate(Model,Set,testFold='data\\test\\', imageFold='image\\', gtFold='gt\\', predFold='pred\\',p=p):
    imFiles=glob.glob(testFold+imageFold+'*')
    
    n=len(imFiles)
    
    i=0
    names=[]
    if Set=='val':
        imFiles=imFiles[:int(n/2)]
    elif Set=='test':
        imFiles=imFiles[int(n/2):]   
    n=len(imFiles)
    outs=np.zeros((2,n))
    
    for file in imFiles:
        name=os.path.split(file)[1]
        
        gt=np.load(testFold+gtFold+name)#*.000646+.0005493
        outs[0,i]=gt.sum().sum()
        
        img=imChange(np.load(testFold+imageFold+name))
        (img,dummy)=mirrorer(img,gt)
        img=np.expand_dims(img,0)
        img=np.expand_dims(img,3)
        name=name[:len(name)-4]+'_predicted'
        
        pred=Model.predict(img)
        pred=pred[0,p:-p,p:-p,0]
        pred=pred/mult
        outs[1,i]=pred.sum().sum()
        np.save(testFold+predFold+name,pred)
        if (i%10==0):
            print(i/n)
        i=i+1
        names.append(name)
        
    
    #mp.pyplot.scatter(outs[0,:],outs[1,:])    
    return outs, names

def evaluate2(Model,Set,testFold='data\\test\\', imageFold='image\\', gtFold='gt\\', predFold='pred\\',p=p):
    imFiles=glob.glob(testFold+imageFold+'*')
    
    n=len(imFiles)
    
    i=0
    names=[]
    if Set=='val':
        imFiles=imFiles[:int(n/2)]
    elif Set=='test':
        imFiles=imFiles[int(n/2):]   
    n=len(imFiles)
    outs=np.zeros((2,n))
    
    for file in imFiles:
        name=os.path.split(file)[1]
        
        gt=np.load(testFold+gtFold+name)#*.000646+.0005493
        outs[0,i]=gt.sum().sum()
        
        img=imChange(np.load(testFold+imageFold+name))
        (img,dummy)=mirrorer(img,gt)
        img=np.expand_dims(img,0)
        img=np.expand_dims(img,3)
        name=name[:len(name)-4]+'_predicted'
        
        pred1=Model.predict(img)
        pred2=Model.predict(np.rot90(img,1,axes=(1, 2)))
        pred3=Model.predict(np.rot90(img,2,axes=(1, 2)))
        pred4=Model.predict(np.rot90(img,3,axes=(1, 2)))
        pred=(pred1+pred2+pred3+pred4)/4
        
        pred=pred[0,p:-p,p:-p,0]
        pred=pred/mult
        outs[1,i]=pred.sum().sum()
        np.save(testFold+predFold+name,pred)
        if (i%10==0):
            print(i/n)
        i=i+1
        names.append(name)
        
    
    #mp.pyplot.scatter(outs[0,:],outs[1,:])    
    return outs, names