# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:26:52 2019
@author: matthew

These functions do all data management for the axon counting model training and evaluating pipeline.
"""


from __future__ import print_function
import numpy as np 
import os, glob, math
from skimage.filters import gaussian

#scaling factor for density maps
mult=1000
#edge mirroring factor (number of pixels extended in each direction during mirroring)
p=16


def imChange(im):
    '''
    conducts normalization by image.
    currently results in [-1,1] range.
    
    z normalization, (image-mean(image))/(2*SD(image)). clipped to [-1,1].
    '''
    ## TODO: try other image enhancement strategies
    
    im=im/np.max(im)
    if np.all(im>=.95):
        im=im
    else:
        im=im/np.max(im) #(redundant ?)
        m=np.mean(im[im<.95]) #taking mean of pixels less than .95
        sd=np.std(im[im<.95]) #taking std of pixels less than .95
        im=np.clip((im-m)/(2*sd),-1,1)
    return im
## TODO evaluate the density map transformations
def densityMapChange(dmap):
    '''
    conducts normalization by density map.
    currently does a log transform after scaling by a multiplicative factor.
    '''
    #dmap=np.log(dmap+1)
    dmap=mult*dmap
    return dmap

def densityMapChangeBack(dmap):
    '''
    conducts normalization by density map.
    currently does a log transform after scaling by a multiplicative factor.
    '''
    #dmap=(np.exp((1)*dmap))-1
    #print(np.sum(dmap))
    dmap=dmap/mult
    #print(np.sum(dmap))
    return dmap
        
def trainGenerator(batch_size,train_path='data\\train',image_folder='image',gt_folder='label',subset='training'):
    '''
    generates input images and masks to be fed to model trainer function
    #todo: fix batch size stuff to do in for loop or similar
    
    https://wiki.python.org/moin/Generators
    '''
    #get filenames from data folders
    imFiles=glob.glob(train_path+'\\'+image_folder+'\\*')
    gtFiles=glob.glob(train_path+'\\'+gt_folder+'\\*')
    n=len(imFiles)
    #randomize dataset image list
    rVec=np.random.permutation(n)
    imFiles = [imFiles[i] for i in rVec] 
    gtFiles = [gtFiles[i] for i in rVec] 
    nOutputs=math.floor(n/batch_size)
    i=0
    while (True) :
       #load first
       try:
           img=np.load(imFiles[i*batch_size])
           gt=np.load(gtFiles[i*batch_size])
       except:
           continue
       #normalize
       img=imChange(img)
       gt=densityMapChange(gt)
       #augment
       (img,gt)=randAug(img,gt)
       #mirror
       (img,gt)=mirrorer(img,gt)
       #change dimensions to make tensorflow happy (result is 4D, shape [batch size, image rows, image cols, image channels (1 in this case)]) 
       img=np.expand_dims(img,0)
       img=np.expand_dims(img,3)
       gt=np.expand_dims(gt,0)
       gt=np.expand_dims(gt,3)
       #load others in the same batch. follows same procedure as loading the first image. consider unifying these 
       if batch_size>1:
           for j in range(batch_size-1):
               imgNew=np.load(imFiles[i*batch_size+j+1])
               gtNew=np.load(gtFiles[i*batch_size+j+1])
               #normalize
               imgNew=imChange(imgNew)
               gtNew=densityMapChange(gtNew)
               #augment
               (imgNew,gtNew)=randAug(imgNew,gtNew)
               #mirror
               (imgNew,gtNew)=mirrorer(imgNew,gtNew)
               #make tensorflow happy with dimensions. see note above
               imgNew=np.expand_dims(imgNew,0)
               imgNew=np.expand_dims(imgNew,3)
               gtNew=np.expand_dims(gtNew,0)
               gtNew=np.expand_dims(gtNew,3)
               #concatenate to minibatch arrays
               img=np.concatenate((img,imgNew),axis=0)
               gt=np.concatenate((gt,gtNew),axis=0)
       #reset order if have used all training inputs
       if i==nOutputs-1:
           i=0
           imFiles = [imFiles[i] for i in rVec] 
           gtFiles = [gtFiles[i] for i in rVec] 
       else:
           i=i+1
       yield (img,gt)
        
        
def randAug(img,gt):
    '''
    augments image and mask at the same time
    currently: 
        mirrors with P=0.5
        rotates by 90 degrees with a P=.25 for each orientation
        multiplies image intensities by a random factor in range [0.85, 1.15] (multiplicative speckle noise)
        blurs image by random sigma in range [0,3]
    '''
    #decide if image will be flipped (mirrored)
    flip=np.random.rand(1)>.5
    #decide which multiple of 90 degrees to use during rotation
    rot=math.floor(4.0*np.random.rand(1))
    #create array of speckle factors
    speckle=1+(.3*np.random.rand(img.shape[0],img.shape[1])-.15)
    #decide if image will be blurred
    blur=np.random.rand(1)>.8
    #create random sigma for blurring
    sigma=round((3*np.random.rand(1))[0])
    
    #flip
    if flip:
        img=np.flip(img, 1)
        gt =np.flip(gt,  1)
    #rotate
    img=np.rot90(img,rot,axes=(0, 1))    
    gt =np.rot90(gt, rot,axes=(0, 1))
    #speckle noise
    img=img*speckle #TODO investigate
    #gaussian blur
    if blur:
        img=gaussian(img, sigma)
    return img,gt

def mirrorer(image,mask,p=p):
    '''
    pads image sides by mirroring p pixels from the edges.
    works by concatenating original arrays with reversed border tiles
    '''
    #mirror image
    ax=image.shape
    top=image[:p,:]
    bot=image[(ax[0]-p):,:]
    image=np.concatenate((top[::-1,:], image, bot[::-1,:]), axis=0, out=None)
    left=image[:,:p]
    right=image[:,(ax[1]-p):]
    image=np.concatenate((left[:,::-1], image, right[:,::-1]), axis=1, out=None)
    mirroredIm=image
    #now mirror gt array
    image=mask
    ax=image.shape
    top=image[:p,:]
    bot=image[(ax[0]-p):,:]
    image=np.concatenate((top[::-1,:], image, bot[::-1,:]), axis=0, out=None)
    left=image[:,:p]
    right=image[:,(ax[1]-p):]
    image=np.concatenate((left[:,::-1], image, right[:,::-1]), axis=1, out=None)
    mirroredDensity=image
    return mirroredIm,mirroredDensity
    


def valGenerator(sourceparam,valFold='data\\validation\\', imageFold='image\\', gtFold='gt\\'):
    '''
    generates input images and masks to be fed to model validation checkpoints.
    does not augment.
    '''   
    valFold='data\\validation'+sourceparam+'\\'
    imFiles=glob.glob(valFold+imageFold+'*')
    n=len(imFiles)    
    #randomize validation set order
    rVec=np.random.permutation(n)
    imFiles = [imFiles[j] for j in rVec]  
    #make generator
    i=0
    while True:
        #get file name
        file=imFiles[i]
        name=os.path.split(file)[1]
        #load files
        try:
            img=np.load(valFold+imageFold+name)
            gt=np.load(valFold+gtFold+name)
        except:
            continue
        
        #normalize 
        img=imChange(img)
        gt=densityMapChange(gt)
        #mirror
        (img,gt)=mirrorer(img,gt)
        #add dimensions for tensorflow model compatibility
        img = np.array(img)[np.newaxis, : , :, np.newaxis]     
        gt = np.array(gt)[np.newaxis, : , :, np.newaxis]  
        #restart generator if we have stepped through all of the validation dataset files
        i=i+1
        if i==n-1:
            i=0
            rVec=np.random.permutation(n)
            imFiles = [imFiles[j] for j in rVec]  
        yield img, gt


def evaluate(Model,Set,sourceparam,evalFold='data\\test\\', imageFold='image\\', gtFold='gt\\', predFold='pred\\',p=p):
    
    #change list to only validation or testing subset
    #TODO edit me for elegance
    if Set=='val':
        evalFold='data\\validation'+sourceparam+'\\'
    elif Set=='test':
        evalFold='data\\test'+sourceparam+'\\' 
    #read in files from testing folder
    imFiles=glob.glob(evalFold+imageFold+'*')
    n=len(imFiles)
    i=0
    names=[]
    n=len(imFiles)
    outs=np.zeros((2,n))
    
    #step through files and apply model
    for file in imFiles:
        name=os.path.split(file)[1]
        try:
            img=np.load(evalFold+imageFold+name)
            gt=np.load(evalFold+gtFold+name)
        except:
            continue
        outs[0,i]=gt.sum().sum()
        #get to standard input form
        (img,dummy)=mirrorer(img,gt)
        img=imChange(img)
        img=np.expand_dims(img,0)
        img=np.expand_dims(img,3)
        name=name[:len(name)-4]+'_predicted'
        #apply model
        pred=Model.predict(img)
        pred=pred[0,p:-p,p:-p,0]
        pred=densityMapChangeBack(pred)
        #write count to outs vector
        outs[1,i]=pred.sum().sum()
        np.save(evalFold+predFold+name,pred)
        if (i%10==0):
            print(i/n)
        i=i+1
        names.append(name)
        
    return outs, names

def evaluate_train(Model,sourceparam,evalFold='data\\train\\', imageFold='image\\', gtFold='label\\', predFold='pred\\',p=p):
    
    #change list to only validation or testing subset
    #TODO edit me for elegance
    evalFold='data\\train'+sourceparam+'\\'
    #read in files from testing folder
    imFiles=glob.glob(evalFold+imageFold+'*')
    n=len(imFiles)
    i=0
    names=[]
    n=len(imFiles)
    outs=np.zeros((2,n))
    predpix=np.asarray([])
    gtpix=np.asarray([])
    #step through files and apply model
    for file in imFiles:
        name=os.path.split(file)[1]
        try:
            img=np.load(evalFold+imageFold+name)
            gt=np.load(evalFold+gtFold+name)
        except:
            continue
        outs[0,i]=gt.sum().sum()
        #get to standard input form
        (img,dummy)=mirrorer(img,gt)
        img=imChange(img)
        img=np.expand_dims(img,0)
        img=np.expand_dims(img,3)
        name=name[:len(name)-4]+'_predicted'
        #apply model
        pred=Model.predict(img)
        pred=pred[0,p:-p,p:-p,0]
        pred=densityMapChangeBack(pred)
        #write count to outs vector
        outs[1,i]=pred.sum().sum()
        #np.save(evalFold+predFold+name,pred)
        if (i%10==0):
            print(i/n)
        i=i+1
        predpix=np.concatenate((predpix,pred.flatten()))
        gtpix=np.concatenate((gtpix,gt.flatten()))
        names.append(name)
        
    return outs, names, predpix, gtpix

#evaluate2 uses all 4 90 degree rotations of the testing image and takes the average of the produced density maps for increased accuracy.
#otherwise it should be the same as evaluate.
    #use if can increase fiji speed
def evaluate2(Model,Set,evalFold='data\\test\\', imageFold='image\\', gtFold='gt\\', predFold='pred\\',p=p):
    imFiles=glob.glob(evalFold+imageFold+'*')
    n=len(imFiles)
    i=0
    names=[]
    #TODO edit me for elegance
    if Set=='val':
        evalFold='data\\validation\\'
    elif Set=='test':
        evalFold=evalFold 
    n=len(imFiles)
    outs=np.zeros((2,n))
    #step through files and apply model
    for file in imFiles:
        name=os.path.split(file)[1]
        img=np.load(evalFold+imageFold+name)
        gt=np.load(evalFold+gtFold+name)
        outs[0,i]=gt.sum().sum()
        
        (img,dummy)=mirrorer(img,gt)
        img=imChange(img)
        
        img=np.expand_dims(img,0)
        img=np.expand_dims(img,3)
        name=name[:len(name)-4]+'_predicted'
        
        pred1=Model.predict(img)
        pred2=Model.predict(np.rot90(img,1,axes=(1, 2)))
        pred3=Model.predict(np.rot90(img,2,axes=(1, 2)))
        pred4=Model.predict(np.rot90(img,3,axes=(1, 2)))
        pred=(pred1+pred2+pred3+pred4)/4
        
        pred=pred[0,p:-p,p:-p,0]
        pred=densityMapChangeBack(pred)
        outs[1,i]=pred.sum().sum()
        np.save(evalFold+predFold+name,pred)
        if (i%10==0):
            print(i/n)
        i=i+1
        names.append(name)
        
    
    #mp.pyplot.scatter(outs[0,:],outs[1,:])    
    return outs, names

def trainGenerator_special(nuse,batch_size,train_path='data\\train',image_folder='image',gt_folder='label',subset='training'):
    '''
    generates input images and masks to be fed to model trainer function
    #todo: fix batch size stuff to do in for loop or similar
    
    makes the dataset only n images
    '''
    #get filenames from data folders
    imFiles=glob.glob(train_path+'\\'+image_folder+'\\*')
    gtFiles=glob.glob(train_path+'\\'+gt_folder+'\\*')
    n=len(imFiles)
    #randomize dataset image list
    rVec=np.random.permutation(n)
    imFiles = [imFiles[i] for i in rVec]
    gtFiles = [gtFiles[i] for i in rVec]
    nOutputs=math.floor(n/batch_size)
    i=0
    while (True) :
       #i=15
       #load first
       try:
           img=np.load(imFiles[i*batch_size])
           gt=np.load(gtFiles[i*batch_size])
       except:
           continue
       #normalize
       img=imChange(img)
       gt=densityMapChange(gt)
       #augment #tuned off for this use
       #(img,gt)=randAug(img,gt)
       #mirror
       (img,gt)=mirrorer(img,gt)
       #change dimensions to make tensorflow happy (result is 4D, shape [batch size, image rows, image cols, image channels (1 in this case)]) 
       img=np.expand_dims(img,0)
       img=np.expand_dims(img,3)
       gt=np.expand_dims(gt,0)
       gt=np.expand_dims(gt,3)
       #load others in the same batch. follows same procedure as loading the first image. consider unifying these 
       if batch_size>1:
           for j in range(batch_size-1):
               imgNew=np.load(imFiles[i*batch_size+j+1])
               gtNew=np.load(gtFiles[i*batch_size+j+1])
               #normalize
               imgNew=imChange(imgNew)
               gtNew=densityMapChange(gtNew)
               #augment
               (imgNew,gtNew)=randAug(imgNew,gtNew)
               #mirror
               (imgNew,gtNew)=mirrorer(imgNew,gtNew)
               #make tensorflow happy with dimensions. see note above
               imgNew=np.expand_dims(imgNew,0)
               imgNew=np.expand_dims(imgNew,3)
               gtNew=np.expand_dims(gtNew,0)
               gtNew=np.expand_dims(gtNew,3)
               #concatenate to minibatch arrays
               img=np.concatenate((img,imgNew),axis=0)
               gt=np.concatenate((gt,gtNew),axis=0)
       #reset order if have used all training inputs
       if i==nuse-1:
           i=0
           #imFiles = [imFiles[i] for i in rVec] 
           #gtFiles = [gtFiles[i] for i in rVec] 
       else:
           i=i+1
       yield (img,gt)