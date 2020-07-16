# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:55:06 2019

@author: mritch3
"""

from model import *
from data import *
import os, glob, time
import matplotlib as mp
from matplotlib.pyplot import *
import numpy as np
import skimage.transform as trans
import PIL as pil
from PIL import Image
from scipy import stats

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.utils import build_tensor_info
from tensorflow.keras import backend as K





os.chdir('..')
export_path=r'export_model'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size=1
n_reps=1

mName='saved models/final_resampled_3-22-2020.hdf5'

#%% load model


#load model from saved file
#model = tf.keras.models.load_model(mName,custom_objects={'mean_squared_error_weighted': mean_squared_error_weighted,'countErr': countErr,'countErr_relative': countErr_relative,'countErr_signed': countErr_signed})

#%% RUN THIS IF TF VERSION 2
#model.save("AxoNet-model")

#%% RUN THIS IF TF VERSION 1
#get initializer
model = tf.keras.models.load_model(mName,custom_objects={'mean_squared_error_weighted': mean_squared_error_weighted,'countErr': countErr,'countErr_relative': countErr_relative,'countErr_signed': countErr_signed})
#model = unet()
#model.load_weights(mName)
#get initializer
init=tf.global_variables_initializer()
#get current keras session. stop other code from running while this is being done
sess= tf.keras.backend.get_session()
#initialize model
sess.run(init)
#change inputs to match whatever custom loss functs you used
tf.saved_model.simple_save(sess, export_path, tags=[tag_constants.SERVING], inputs={'input_image': model.input},outputs={'output_map': model.output})
