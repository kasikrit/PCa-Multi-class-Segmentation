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

data_note = f'NT-{job}-eva-3c-val-fold-{BACKBONE}'

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
    save_path = './evaluate'
    slide_no_pos = 2
    verbose=1
    # train_pair_exp_file = '10slides_train_pair_exp_N_GP3_GP4.dat'
    # val_pair_exp_file = '10slides_val_pair_exp_N_GP3_GP4.dat'

elif platform.system() == 'Linux':    
    base_directory = '.'
    base_patch = job
    dataset_directory = os.path.join(base_directory, base_patch)
    model_best_path = './evaluate'
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
def sanity_check_with_patch_id(X, y, note, pair_idx, pairs):   
    for i, pair_id in enumerate(pair_idx):
        image_file = pairs[pair_id][0].split(os.path.sep)[file_name_pos]
        mask_file = pairs[pair_id][1].split(os.path.sep)[file_name_pos]
        slide_id = pairs[pair_id][1].split(os.path.sep)[slide_no_pos]
        image_id = str(slide_id + '_' + image_file)
        mask_id = str(slide_id + '_' + mask_file)
        
        plt.figure(figsize=(12, 6), dpi=600)
        plt.subplot(121)
        plt.imshow(X[i])
        plt.title(note + ' Image: ' + image_id )
        plt.subplot(122)
        plt.imshow(y[i], cmap='gray')
        (unique, counts) = np.unique(y[i], return_counts=True)
        xlabel = str(unique) + "\n" + str(counts)
        plt.xlabel(xlabel)
        plt.title('Mask: ' + mask_id )
        plt.show()
        print(np.unique(y[i], return_counts=True))
        
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
# import pickle
# # train_pair_exp_file = '10slides_train_pair_inc_N_GP3_GP4.dat'
# with open (train_pair_exp_file, 'rb') as fp:
#     train_pair_RBF = pickle.load(fp)
# print('\nloaded train_pair_RBF: ', len(train_pair_RBF)) 
# print(train_pair_RBF[:5])

# path = train_pair_RBF[0]
# path3 = PureWindowsPath(dataset_directory, path[0])
# print(path3)
# path4 = PurePosixPath(path3)
# print(path4)

# path = '/Volumes/MyPassportB/slides/1862/mask20x/59_57.png'
# path1 =  PurePosixPath(path)

#%%
# import pickle
# # val_pair_exp_file = '10slides_val_pair_inc_N_GP3_GP4.dat'
# with open (val_pair_exp_file, 'rb') as fp:
#     val_pair_exp = pickle.load(fp)
# print('\nloaded val_pair_exp: ', len(val_pair_exp)) 
    
# #%%
# import pickle
# with open (test_pair_exp_file, 'rb') as fp:
#     test_pair_exp = pickle.load(fp)
# print('loaded test_pair_exp: ', len(test_pair_exp)) 

#%%
# print('\nclass_weights = ', class_weights)

# dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
# focal_loss = sm.losses.CategoricalFocalLoss()
# # focal_loss = sm.losses.BinaryFocalLoss()
# total_loss = dice_loss + (1 * focal_loss)

# metrics_list = [
#             #'accuracy', 
#             sm.metrics.IOUScore(
#                         # threshold=0.5, 
#                         class_weights=class_weights
#                         ), 
#             sm.metrics.FScore(
#                         # threshold=0.5, 
#                         class_weights=class_weights
#                          ), 
#             #iou_coef1, dice_coef1,
#             iou, jaccard_distance, total_loss,
#             dice_coef2, 
#             #dice_coef3, #error line 140 y_true_flatten = np.asarray(y_true).astype(np.bool)
#             precision, recall, accuracy,
#             ] 
#%%
#stop

#%%
# t1 = datetime.now()
# import pickle
# fold_list = [1, 2, 3, 4, 5]
# for fold in fold_list:
#     print(f"\n\n{fold=}")
#     val_pair_exp_file = f'val_pair_exp_fold_{fold}_N_GP3_GP4_relative_path.dat'
#     print(val_pair_exp_file)
#     with open (val_pair_exp_file, 'rb') as fp:
#         val_pair_exp = pickle.load(fp)
#     print('\nLoaded val_pair_exp: ', len(val_pair_exp))

#     print('\nCreate val_generator_inf')
#     val_generator_inf = utility.DataGenerator(
#         dataset_directory,
#         val_pair_exp,
#         # val_pair_exp_samples,
#         num_classes=n_classes,
#         batch_size=batch_size, 
#         dim=(patch_size, patch_size, 3),
#         # dim=(384, 384, 3), #PSPNet
#         shuffle=False,
#         inference=True,
#         augmentation=utility.get_training_augmentation(),
#         preprocessing=utility.get_preprocessing(preprocess_input),
#         rescale=None,
#     )
#     val_steps_inf = val_generator_inf.__len__()
#     print('val_steps_inf: ', val_steps_inf)

#     #%
#     image_number = random.randint(0, val_steps_inf-1)
#     print('random image number: ', image_number)
#     X_val, y_val, pair_idx_val = val_generator_inf.__getitem__(image_number)
#     print(f"sanicty check {X_val.shape=}, {y_val.shape=}")
#     y_val_argmax = np.argmax(y_val, axis=3).astype('uint8')
    
#     cnt_model = 1
#     for best_model in model_list:
#         y_val_true_file = os.path.join(save_path, f"y_val_true_fold_{fold}_{cnt_model}.npy")
#         y_val_pred_file = os.path.join(save_path, f"y_val_pred_fold_{fold}_{cnt_model}.npy")
       
#         print('\nVal_true: ', y_val_true_file)
#         print('Val_pred: ', y_val_pred_file)
        
#         print("\nEvaluate the whole val set")
#         y_val_list = []
#         y_val_pred_list = []
#         pair_idx_val_list = []
           
#         # for batch in range(1, val_steps_RBF+1):
#         print('\nPredicting...')
#         for batch in tqdm(range(val_generator_inf.__len__())):
#             X_val, y_val, pair_idx_val = val_generator_inf.__getitem__(batch)
#             # print('\n', X_val.shape, y_val.shape, len(pair_idx_val))
            
#             y_val_argmax = np.argmax(y_val, axis=3).astype('uint8')
#             # print('y_val_argmax.shape: ', y_val_argmax.shape)
#             verbose=2   
#             y_val_pred = best_model.predict(
#                             X_val,   
#                             batch_size=batch_size, 
#                             verbose=verbose)
        
#             # y_pred_argmax = np.argmax(y_pred, axis=3).astype('uint8')
#             # print('\ny_pred_argmax.shape: ', y_pred_argmax.shape)
               
#             # X_val_list.append(X_val)
#             y_val_list.append(y_val_argmax)
#             y_val_pred_list.append(y_val_pred)
#             pair_idx_val_list.append(pair_idx_val)
        
#         # #%
#         # X_val = np.concatenate(X_val_list, axis=0)
#         y_val_true = np.concatenate(y_val_list, axis=0)
#         y_val_pred = np.concatenate(y_val_pred_list, axis=0)
#         pair_idx_val = np.concatenate(pair_idx_val_list, axis=0)
#         print('\ny_val_true.shape, y_val_pred.shape, pair_idx_val.shape')
#         print(y_val_true.shape, y_val_pred.shape, pair_idx_val.shape)
#         # print(y_val_pred.shape)
#         del y_val_list
#         del y_val_pred_list
        
#         # y_val_pred_argmax = np.argmax(y_val_pred, axis=3) #int64
#         # print(y_val_true.shape, y_val_pred_argmax.shape)
        
#         print("\nSaving prediction of validation")
#         np.save(y_val_true_file, y_val_true)
#         np.save(y_val_pred_file, y_val_pred)
#         print('Saved: ', y_val_true_file)
#         print('Saved: ', y_val_pred_file)
#         # del y_val_true
#         # del y_val_pred
        
#         cnt_model=cnt_model + 1 #last cmd
        
#%%
print('\n\nEnsemble evaluation for val')

fold_list = [1, 2, 3, 4, 5]

for fold in [3]: 
    print('\nfold: ', fold)
    y_val_true_file = os.path.join(save_path, f"y_val_true_fold_{fold}_1.npy")
    y_val_true = np.load(y_val_true_file)
    print('Loaded y_val_true: ', y_val_true.shape)
    
    y_val_pred_file = os.path.join(save_path, f"y_val_pred_fold_{fold}_{fold}.npy")          
    y_val_pred = np.load(y_val_pred_file)
    print('Loaded y_val_pred: ', y_val_pred.shape) 

    #%
    y_val_pred_argmax = np.argmax(y_val_pred, axis=3)
    print('y_val_preds_avg_argmax.shape: ', y_val_pred_argmax.shape)
           
    #%
    # 'class_labels' specifies the classes of interest (excluding 'normal')
    class_labels = [0, 1, 2]  # Gleason patterns 3 and 4  

    if SAMPLES:
        random_size =int(len(y_val_true) * random_ratio)
        batch_idx = random.sample(range(1, len(y_val_true)), 
                                  random_size)
        print('len(batch_idx): ', len(batch_idx))
    
        
        y_true_masks = y_val_true[batch_idx].flatten()
        y_pred_masks = y_val_pred_argmax[batch_idx].flatten()
    else:
        y_true_masks = y_val_true.flatten()
        y_pred_masks = y_val_pred_argmax.flatten()
    
    print(f"{y_true_masks.shape=}, {y_pred_masks.shape=}")    
    
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


