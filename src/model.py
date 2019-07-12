# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:50:27 2019

@author: mritch3
"""

import data_915_plateau as dat
import max_subarray_tf
import numpy as np 
import matplotlib as mp 
import os, glob
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras import backend as keras
import tensorflow as tf
import keras.layers.advanced_activations
import tensorflow_probability as tfp

base_n=5
p=16
def f(x):
    return 2**x

def unet(pretrained_weights = None,input_shape = (224,224,1)):
    inputs = Input(input_shape,name='input_image')
    conv1 = Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #pool1= BatchNormalization(axis=3)(pool1)
    
    conv2 = Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    #conv2= BatchNormalization(axis=3)(conv2)
    conv2 = Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #pool2= BatchNormalization(axis=3)(pool2)
    
    conv3 = Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    #conv3= BatchNormalization(axis=3)(conv3)
    conv3 = Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #pool3= BatchNormalization(axis=3)(pool3)
    
    conv4 = Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    #conv4= BatchNormalization(axis=3)(conv4)
    conv4 = Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #pool4= BatchNormalization(axis=3)(pool4)
    
    conv5 = Conv2D(f(base_n+4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(f(base_n+4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(f(base_n+3), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(f(base_n+2), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(f(base_n+1), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(f(base_n), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9= BatchNormalization(axis=3)(conv9)
    conv10 = Conv2D(1, 1 ,activation = 'relu',name='output_map')(conv9)

    model = Model(inputs = inputs, outputs = conv10) 
    model.compile(optimizer = Adam(lr = 1E-4), loss = mean_squared_error_weighted, metrics = ['mean_absolute_error','mean_squared_error',countErr]) 
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def ssim(target,prediction):
    out=tf.image.ssim(target,prediction,2)
    out=-1*out
    return out

def corr2(target,prediction):
    out=tfp.stats.correlation(tf.reshape(target[0,p:-p,p:-p,0],[224**2,1]),tf.reshape(prediction[0,p:-p,p:-p,0],[224**2,1]))
    return out
    
    
def mesa_dist(target, prediction):    
    target=target[0,:,:,0]
    prediction=prediction[0,:,:,0]
    diff1=target-prediction
    diff2=-target+prediction
    dMesa= tf.math.reduce_max(max_subarray_tf.maxSubArray_2D(diff1)[0],max_subarray_tf.maxSubArray_2D(diff2)[0])
    return dMesa
    
def mean_squared_error_weighted(y_true, y_pred):
    #y_true=1000*y_true
    #y_pred=1000*y_pred
    dens = tf.not_equal(y_true, 0)
    sqdiff=K.square(y_pred - y_true)
    sqdiff=tf.where(dens, 1*sqdiff, sqdiff) #condition, iftrue, iffalse
    #return K.mean(sqdiff[0,int(p/2):-int(p/2),int(p/2):-int(p/2),0])
    return K.mean(sqdiff[0,p:-p,p:-p,0])

def max_squared_error_weighted(y_true, y_pred):
    dens = tf.not_equal(y_true, 0)
    sqdiff=K.square(y_pred - y_true)
    sqdiff=tf.where(dens, sqdiff, sqdiff) #condition, iftrue, iffalse
    return K.max(sqdiff)


def tot_err(y_true, y_pred):
    return K.square(K.sum(y_true- y_pred))

def conv_relu_bn(nFilt,layer):
    conv = Conv2D(nFilt, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer)
    conv= BatchNormalization(axis=3)(conv)
    conv = Conv2D(nFilt, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
    conv= BatchNormalization(axis=3)(conv)
    return conv

def IOU(target, prediction):
    intersection = (target==1) & (prediction ==1)
    union = (target==1) | (prediction ==1)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def countErr(target, prediction):
    a=K.sum(target)
    b=K.sum(prediction)
    error = (b-a)
    return tf.math.abs(error/dat.mult)

def targSum(target, prediction):
    a=K.sum(target)
    return a
def predSum(target, prediction):
    a=K.sum(prediction)
    return a

def unet_burg(pretrained_weights = None,input_shape = (668,888,1)):
    inputs = Input(input_shape)
    conv1 = Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #conv1= BatchNormalization(axis=3)(conv1)
    conv1 = Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #pool1= BatchNormalization(axis=3)(pool1)
    
    conv2 = Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    #conv2= BatchNormalization(axis=3)(conv2)
    conv2 = Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #pool2= BatchNormalization(axis=3)(pool2)
    
    conv3 = Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    #conv3= BatchNormalization(axis=3)(conv3)
    conv3 = Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #pool3= BatchNormalization(axis=3)(pool3)
    
    conv4 = Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    #conv4= BatchNormalization(axis=3)(conv4)
    conv4 = Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #pool4= BatchNormalization(axis=3)(pool4)
    
    conv5 = Conv2D(f(base_n+4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(f(base_n+4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(f(base_n+3), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(f(base_n+2), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(f(base_n+1), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(f(base_n), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
   # conv9 = Conv2D(f(base_n-1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'relu',)(conv9)

    model = Model(inputs = inputs, outputs = conv10) #
    model.compile(optimizer = Adam(lr = 1E-4), loss = mean_squared_error_weighted, metrics = ['mean_absolute_error','mean_squared_error',countErr]) ## add iou later
    

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def unet_batchnorm(pretrained_weights = None,input_shape = (288,288,1)):
    inputs = Input(input_shape)
    conv1 = Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1= BatchNormalization(axis=3)(conv1)
    conv1 = Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1= BatchNormalization(axis=3)(pool1)
    
    conv2 = Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2= BatchNormalization(axis=3)(conv2)
    conv2 = Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2= BatchNormalization(axis=3)(pool2)
    
    conv3 = Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3= BatchNormalization(axis=3)(conv3)
    conv3 = Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3= BatchNormalization(axis=3)(pool3)
    
    conv4 = Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4= BatchNormalization(axis=3)(conv4)
    conv4 = Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    pool4= BatchNormalization(axis=3)(pool4)
    
    conv5 = Conv2D(f(base_n+4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5= BatchNormalization(axis=3)(conv5)
    conv5 = Conv2D(f(base_n+4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5= BatchNormalization(axis=3)(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(f(base_n+3), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6= BatchNormalization(axis=3)(conv6)    
    conv6 = Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6= BatchNormalization(axis=3)(conv6)

    up7 = Conv2D(f(base_n+2), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7= BatchNormalization(axis=3)(conv7) 
    conv7 = Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7= BatchNormalization(axis=3)(conv7) 
    
    up8 = Conv2D(f(base_n+1), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8= BatchNormalization(axis=3)(conv8)
    conv8 = Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8= BatchNormalization(axis=3)(conv8)

    up9 = Conv2D(f(base_n), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9= BatchNormalization(axis=3)(conv9)
    conv9 = Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9= BatchNormalization(axis=3)(conv9)
    conv9 = Conv2D(f(base_n-1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1,activation = 'relu',)(conv9)

    model = Model(inputs = inputs, outputs = conv10) #
    model.compile(optimizer = Adam(lr = 2E-3), loss = mean_squared_error_weighted, metrics = []) 

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def unet_applied(pretrained_weights = None,input_shape = (2080,2080,1)):
    inputs = Input(input_shape)
    conv1 = Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #conv1= BatchNormalization(axis=3)(conv1)
    conv1 = Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #pool1= BatchNormalization(axis=3)(pool1)
    
    conv2 = Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    #conv2= BatchNormalization(axis=3)(conv2)
    conv2 = Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #pool2= BatchNormalization(axis=3)(pool2)
    
    conv3 = Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    #conv3= BatchNormalization(axis=3)(conv3)
    conv3 = Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #pool3= BatchNormalization(axis=3)(pool3)
    
    conv4 = Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    #conv4= BatchNormalization(axis=3)(conv4)
    conv4 = Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #pool4= BatchNormalization(axis=3)(pool4)
    
    conv5 = Conv2D(f(base_n+4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(f(base_n+4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(f(base_n+3), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(f(base_n+2), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(f(base_n+1), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(f(base_n), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = Conv2D(f(base_n-1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1 ,activation = 'relu',)(conv9)

    model = Model(inputs = inputs, outputs = conv10) 
    model.compile(optimizer = Adam(lr = 1E-4, decay=.000001), loss = mean_squared_error_weighted, metrics = ['mean_absolute_error','mean_squared_error',countErr]) 
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model