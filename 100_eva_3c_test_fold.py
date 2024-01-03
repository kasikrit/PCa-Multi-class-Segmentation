# import os
# os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
from tensorflow import keras

# import segmentation_models as sm #1.0.1

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
# import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adadelta, Nadam, Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os, platform, sys
from pathlib import Path, PureWindowsPath, PurePosixPath
# from tqdm import tqdm
from random import sample, choice
# from PIL import Image

from datetime import datetime 
import random
import glob #glob2

# import tensorflow_addons as tfa
# tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)

# from livelossplot import PlotLossesKeras

import os
from matplotlib import pyplot as plt

# from PIL import Image
# from tensorflow.keras import backend, optimizers

#FOCAL LOSS AND DICE METRIC
#Focal loss helps focus more on tough to segment classes.
# from focal_loss import BinaryFocalLoss, SparseCategoricalFocalLoss

import utility
import models

from utility import ( 
    dice_coef, dice_coef_loss, jacard_coef,
    jacard_coef_loss, iou_coef1, dice_coef1,
    iou, jaccard_distance, dice_coef2, dice_coef3, precision, recall, accuracy)

import multi_class_IoU

print(tf.test.gpu_device_name())
# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # tf.config.experimental.set_memory_growth(gpus[0], enable=True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
              
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

print('\ntf version: ', tf.__version__)
print('keras version: ', keras.__version__)
# print('segmentation model version: ', sm.__version__)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
print('tf.test.is_built_with_cuda(): ', tf.test.is_built_with_cuda())

#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 20:46:01 2022

@author: kasikritdamkliang
"""

#%
default_n_classes = 5
default_classes = [0, 1, 2, 3, 4]

# n_classes = 4
# classes = [0, 1, 2, 3]
# labels = ['BG+FG', 'Normal','GP3', 'GP4']

n_classes = 3
classes = [0, 1, 2]
# labels = ['BG+FG+Normal', 'GP3', 'GP4']
labels = ['Normal', 'GP3', 'GP4']

# n_classes = 2
# classes = [0, 1]
# labels = ['BG+FG+Normal','GP3']

seed = 1337
epochs = 120
# epochs = 100
batch_size = 8
# batch_size = 32 #for SM 
check_ratio = 4 # amont to plot = batch_size/check_ratio
img_size = patch_size = 256
target_size = patch_size
dropout_rate = 0.25
# dropout_rate = 0.4
# dropout_rate = 0.35
# split_train_val = 0.35
init_lr = 1e-4
activation = 'softmax'

LOGSCREEN = False
# LOGSCREEN = True
CW = False
# PROB = True
# PROB = False
# SAMPLES = True
SAMPLES = False
random_ratio = 0.8

#1
# BACKBONE = 'resnet34' 
# BACKBONE = 'seresnet50'
# BACKBONE = 'inceptionresnetv2'
# BACKBONE = 'mobilenet'
# BACKBONE = 'seresnext50'
# BACKBONE = 'seresnext101'
BACKBONE = 'dilated-att-res-unet'
PREP_BACKBONE = 'seresnet50'

fold='fold_3' #2
job = '100slides'
# preprocess_input = sm.get_preprocessing(PREP_BACKBONE)

data_note = f'NT-{job}-eva-3c-test-fold-{BACKBONE}'

# train_pair_exp_file = f'train_pair_exp_{fold}_N_GP3_GP4_relative_path.dat'
# val_pair_exp_file = f'val_pair_exp_{fold}_N_GP3_GP4_relative_path.dat'
test_pair_exp_file = 'test_pair_exp_relative_path.dat'

if fold=='fold_1':
    class_weights =  [0.6689810602050595,
                      1.4180794131137744,
                      1.2499839925533194]
    
elif fold=='fold_2':
    class_weights = [0.6395503885274618,
                     1.4008227869987053,
                     1.3840152857246992]

elif fold=='fold_3':
   # boot up N, GP3, GP4 ratio 0.85
    class_weights = [0.6666002869103805, 
                     1.4151877180904922, 
                     1.2606674654315284]
    
elif fold=='fold_4':
    # boot up N, GP3, GP4 ratio 0.85
    class_weights = [0.671382400720509,
                     1.3393257399755412,
                     1.309086395875927]
else:
    class_weights = [0.6567904125037317,
                     1.3347561416532616,
                     1.3731665973899736]

#%%
if platform.system() == 'Darwin':
    # base_directory = '/Users/kasikritdamkliang/Datasets/PCa'
    base_directory = '/Volumes/MyPassportB'
    base_patch = 'slides'
    dataset_directory = os.path.join(base_directory, base_patch)
    model_best_path = '/Users/kasikritdamkliang/Library/CloudStorage/OneDrive-PrinceofSongklaUniversity/PCa/models/NT/5folds/'
    save_path = 'evaluate/'
    slide_no_pos = 2
    verbose=1
    # train_pair_exp_file = '10slides_train_pair_exp_N_GP3_GP4.dat'
    # val_pair_exp_file = '10slides_val_pair_exp_N_GP3_GP4.dat'

elif platform.system() == 'Linux':    
    base_directory = '.'
    base_patch = job
    dataset_directory = os.path.join(base_directory, base_patch)
    model_best_path = 'evaluate/'
    save_path = model_best_path
    slide_no_pos = 2
    verbose=2

else:
    model_best_path = Path(os.path.abspath(r"C:\Users\kasikrit\OneDrive - Prince of Songkla University\PCa\models"))
    # dataset_directory = Path(os.path.abspath('C:\PCa-Kasikrit\segmentation\slides'))
    dataset_directory = Path(os.path.abspath(r"E:\slides\svs"))
    slide_no_pos = 3
    verbose=1

file_name_pos = slide_no_pos + 2

#%%
# print(model_best_path)
# print('\nLoading models...')
# models = list(glob.glob(os.path.join(model_best_path, '*.hdf5'), 
#                         recursive=True ))
# models.sort()

# model_1 = load_model(models[3], compile=False)
# model_2 = load_model(models[4], compile=False)
# model_3 = load_model(models[0], compile=False)
# model_4 = load_model(models[1], compile=False)
# model_5 = load_model(models[2], compile=False)

# print('Models loaded')

# model_list = [model_1, model_2, model_3, model_4, model_5]
# for model in model_list:
#     print(model)

# from pathlib import PureWindowsPath, PurePosixPath
# path = PureWindowsPath(r"E:\slides\svs")
# PurePosixPath('.//Slides', *path.parts[1:])

#%%
### Pipe standard outout to file
timestr = datetime.now().strftime("%Y%m%d-%H%M")
print("Running date: ", timestr)
log_datetime = datetime.now()
log_file_name = data_note + "-" + timestr + ".txt"
model_name = data_note + "-" + timestr

#%%
if LOGSCREEN:
    log_file = open(log_file_name, "w")

    old_stdout = sys.stdout
    sys.stdout = log_file

#%
start_exe1 = datetime.now()
print("\nRunning date: ", timestr)
print('Start exe time: ', start_exe1)
print('\ntf version: ', tf.__version__)
print('keras version: ', keras.__version__)
# print('segmentation model version: ', sm.__version__)

print("\ndata_note: ", data_note)
print("n_classes: ", n_classes)
print("labels: ", labels)
print('LOGSCREEN: ', LOGSCREEN)
print('dataset_directory: ', dataset_directory)
print('model_best_path: ', model_best_path)
print('model_name: ', model_name)
# print('test_dataset_path: ', test_dataset_path)
print('epochs: ', epochs)
print('batch_size: ', batch_size)
print('activation: ', activation)
print('dropout_rate: ', dropout_rate)
# print('split_train_val: ', split_train_val)
print('init_lr: ', init_lr)
print('PREP_BACKBONE: ', PREP_BACKBONE)
print('dataset_directory: ', dataset_directory)
        
#%%
print('\n\nEnsemble evaluation for test')

fold_list = [1, 2, 3, 4, 5]

for fold in [3]: 
    print('\nfold: ', fold)
    y_test_true_file = os.path.join(save_path, f"y_test_true.npy")
    y_test_true = np.load(y_test_true_file)
    print('Loaded y_test_true: ', y_test_true.shape)
    
    y_test_pred_file = os.path.join(save_path, f"y_test_pred_3.npy")          
    y_test_pred = np.load(y_test_pred_file)
    print('Loaded y_test_pred: ', y_test_pred.shape) 

    #%
    y_test_pred_argmax = np.argmax(y_test_pred, axis=3)
    print('y_test_preds_avg_argmax.shape: ', y_test_pred_argmax.shape)
           
    #%
    # 'class_labels' specifies the classes of interest (excluding 'normal')
    class_labels = [0, 1, 2]  # Gleason patterns 3 and 4  

    if SAMPLES:
        random_size =int(len(y_test_true) * random_ratio)
        batch_idx = random.sample(range(1, len(y_test_true)), 
                                  random_size)
        print('len(batch_idx): ', len(batch_idx))
    
        
        y_true_masks = y_test_true[batch_idx].flatten()
        y_pred_masks = y_test_pred_argmax[batch_idx].flatten()
    else:
        y_true_masks = y_test_true.flatten()
        y_pred_masks = y_test_pred_argmax.flatten()
    
    print(f"{y_true_masks.shape=}, {y_pred_masks.shape=}")    
    
    jac_scores = multi_class_IoU.iou_for_classes(
        y_true_masks,
        y_pred_masks,
        class_labels)  
    print(f"{np.mean(jac_scores)=}\n")
    
    iou_scores = multi_class_IoU.calculate_mean_iou_for_classes(
        y_true_masks,
        y_pred_masks,
        class_labels)  
    print(f"{np.mean(iou_scores)=}\n")
        
    dice_scores = multi_class_IoU.calculate_dice_for_classes(
        y_true_masks,
        y_pred_masks,
        class_labels)
    print(f"{np.mean(dice_scores)=}\n")

    precision_scores = multi_class_IoU.precision(
        y_true_masks,
        y_pred_masks,
        class_labels)
    # print(f"Precision all classes: {dice_scores}")
    mean_precision = tf.reduce_mean(precision_scores)
    print(f"{mean_precision.numpy()=}\n")

    recall_scores = multi_class_IoU.recall(
        y_true_masks,
        y_pred_masks,
        class_labels)
    print(f"Recall all classes: {recall_scores}")
    mean_recall = tf.reduce_mean(recall_scores)
    print(f"{mean_recall.numpy()=}\n")

    accuracy_scores = multi_class_IoU.accuracy(
        y_true_masks,
        y_pred_masks,
        class_labels)
    # print(f"Accuracy all classes: {accuracy_scores}")
    mean_accuracy = tf.reduce_mean(accuracy_scores)
    print(f"{mean_accuracy.numpy()=}\n")

  
#%%
start_exe2 = datetime.now() - start_exe1
print('Execution times: ', data_note, start_exe2, '\n')


#%%
if LOGSCREEN:
    sys.stdout = old_stdout
    log_file.close()   

#%%


