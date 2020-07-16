# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:50:27 2019
@author: matthew

These functions create the model and all of the custom model metrics we use.
Heavy use of tensorflow and keras.
"""

import data
#import max_subarray_tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as kOpt
from tensorflow.keras import backend as keras
import tensorflow as tf
#import tensorflow_probability as tfp

#define a few parameters
base_n=5
p=16
#this function is just shorthand for the base-2 exponential function
def f(x):
    return 2**x

def unet(pretrained_weights = None,input_shape = (224,224,1)):
    #see ronnenberger for architecture description
    
    inputs = layers.Input(input_shape,name='image_input')
    conv1 = layers.Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = layers.Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    #pool1= BatchNormalization(axis=3)(pool1)
    
    conv2 = layers.Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    #conv2= BatchNormalization(axis=3)(conv2)
    conv2 = layers.Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    #pool2= BatchNormalization(axis=3)(pool2)
    
    conv3 = layers.Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    #conv3= BatchNormalization(axis=3)(conv3)
    conv3 = layers.Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    #pool3= BatchNormalization(axis=3)(pool3)
    
    conv4 = layers.Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    #conv4= BatchNormalization(axis=3)(conv4)
    conv4 = layers.Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)
    #pool4= BatchNormalization(axis=3)(pool4)
    
    conv5 = layers.Conv2D(f(base_n+4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = layers.Conv2D(f(base_n+4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)
    
    up6 = layers.Conv2D(f(base_n+3), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(drop5))
    merge6 = layers.concatenate([drop4,up6], axis = 3)
    conv6 = layers.Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = layers.Conv2D(f(base_n+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    up7 = layers.Conv2D(f(base_n+2), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(conv6))
    merge7 = layers.concatenate([conv3,up7], axis = 3)
    conv7 = layers.Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = layers.Conv2D(f(base_n+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    up8 = layers.Conv2D(f(base_n+1), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(conv7))
    merge8 = layers.concatenate([conv2,up8], axis = 3)
    conv8 = layers.Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = layers.Conv2D(f(base_n+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    up9 = layers.Conv2D(f(base_n), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(conv8))
    merge9 = layers.concatenate([conv1,up9], axis = 3)
    conv9 = layers.Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    #conv9=Dropout(rate=.2)(conv9)
    conv9 = layers.Conv2D(f(base_n), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = layers.Conv2D(1, 1 ,activation = 'relu',)(conv9)
    
    model = models.Model(inputs = inputs, outputs = conv10) 
    #model.compile(optimizer = kOpt.Adam(lr = 1E-4), loss = mean_squared_error_weighted, metrics = ['mean_absolute_error','mean_squared_error',countErr,countErr_signed,countErr_relative]) 
    model.compile(optimizer = kOpt.Adam(lr = 1E-4), loss = mean_squared_error_weighted) 
    #decay = 1E-4/100
    #load existing model if provided
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    return model

#unused shorthand for batchnorm/conv combo
def conv_relu_bn(nFilt,layer):
    conv = layers.Conv2D(nFilt, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer)
    conv= layers.BatchNormalization(axis=3)(conv)
    conv = layers.Conv2D(nFilt, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
    conv= layers.BatchNormalization(axis=3)(conv)
    return conv

#%% define custom loss functions and metrics for tensorflow. 
    
#structural similarity index. can be useful to look at overall density map quality.
#see https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html for details    
def ssim(target,prediction):
    out=tf.image.ssim(target,prediction,2)
    out=-1*out
    return out
#correlation of gt and predicted count density values, pixelwise
#def corr2(target,prediction):
#    out=tfp.stats.correlation(tf.reshape(target[0,p:-p,p:-p,0],[192**2,1]),tf.reshape(prediction[0,p:-p,p:-p,0],[192**2,1]))
#    return out
#not used. implements Lempitsky et al.'s loss function
def mesa_dist(target, prediction):    
    target=target[0,:,:,0]
    prediction=prediction[0,:,:,0]
    diff1=target-prediction
    diff2=-target+prediction
    dMesa= tf.math.reduce_max(max_subarray_tf.maxSubArray_2D(diff1)[0],max_subarray_tf.maxSubArray_2D(diff2)[0])
    return dMesa
#weighted mse. not weighted if weight=1
def mean_squared_error_weighted(y_true, y_pred):
    #essentially works as MSE with the cropping to remove mirrored regions
    weight=1
    dens = tf.not_equal(y_true, 0)
    sqdiff=keras.square(y_pred - y_true)
    sqdiff=tf.where(dens, weight*sqdiff, sqdiff) #condition, iftrue, iffalse
    return keras.mean(sqdiff[:,p:-p,p:-p,:]) 

def mean_squared_error_bias(y_true, y_pred):
    #combo loss function of MSE and whole image error metric
    weight=1
    dens = tf.not_equal(y_true, 0)
    sqdiff=keras.square(y_pred - y_true)
    sqdiff=tf.where(dens, weight*sqdiff, sqdiff) #condition, iftrue, iffalse
    err=countErr(y_true, y_pred)
    return keras.mean(sqdiff[0,p:-p,p:-p,0])+(.1*tf.square(err))


def mean_squared_error_worst(y_true, y_pred):
    n=.25 #evaluate worst 1/4 of pixels- attempted substitute for max subarray
    
    sqdiff=keras.square(y_pred - y_true)[0,p:-p,p:-p,0]
    sqdiff=tf.reshape(sqdiff,[-1])
    sqdiff=tf.sort(sqdiff) #sorts in ascending order
    sqdiff=sqdiff[int(n*192*192):]
    
    return keras.mean(sqdiff)


#evaluate loss only at worst pixel
def max_squared_error_weighted(y_true, y_pred):
    dens = tf.not_equal(y_true, 0)
    sqdiff=keras.square(y_pred - y_true)
    sqdiff=tf.where(dens, sqdiff, sqdiff) #condition, iftrue, iffalse
    return keras.max(sqdiff)
#square of summed error
def tot_err(y_true, y_pred):
    #returns total image error squared
    return keras.square(keras.sum(y_true- y_pred)/data.mult)
#absolute value of total error in image
def countErr(target, prediction):
    #target=tf.math.exp(target)-1
    #prediction=tf.math.exp(prediction)-1
    
    a=keras.sum(target)
    b=keras.sum(prediction)
    error = (b-a)
    return tf.math.abs(error/data.mult)
#relative error over full image
def countErr_relative(target, prediction):
    #target=tf.math.exp(target)-1
    #prediction=tf.math.exp(prediction)-1
    #calculates percent error
    
    a=keras.sum(target)+1
    b=keras.sum(prediction)+1
    error = (b-a)/a
    return tf.math.abs(error)
#signed total error in image
def countErr_signed(target, prediction):
    #target=tf.math.exp(target)-1
    #prediction=tf.math.exp(prediction)-1
    
    
    a=keras.sum(target)
    b=keras.sum(prediction)
    error = (b-a)
    return (error/data.mult)
#get averages for prediciton and target values
def targSum(target, prediction):
    a=keras.sum(target)
    return a
def predSum(target, prediction):
    a=keras.sum(prediction)
    return a

