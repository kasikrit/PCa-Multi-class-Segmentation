import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
from tensorflow import keras

import segmentation_models as sm #1.0.1

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import cv2
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
from tqdm import tqdm
from random import sample, choice
# from PIL import Image

from datetime import datetime 
import random
import glob #glob2

import tensorflow_addons as tfa
tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)

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
print('segmentation model version: ', sm.__version__)

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

#1
# BACKBONE = 'resnet34' 
# BACKBONE = 'seresnet50'
# BACKBONE = 'inceptionresnetv2'
# BACKBONE = 'mobilenet'
# BACKBONE = 'seresnext50'
# BACKBONE = 'seresnext101'
BACKBONE = 'dilated-att-res-unet'
PREP_BACKBONE = 'seresnet50'

fold='fold_1' #2
job = '100slides'
preprocess_input = sm.get_preprocessing(PREP_BACKBONE)

data_note = f'NT-9-{job}-{fold}-train-3c-val-loss-{BACKBONE}-{epochs}ep'

train_pair_exp_file = f'train_pair_exp_{fold}_N_GP3_GP4_relative_path.dat'
val_pair_exp_file = f'val_pair_exp_{fold}_N_GP3_GP4_relative_path.dat'
test_pair_exp_file = 'test_pair_exp_relative_path.dat'

#classweihts ratio 0.8
if fold=='fold_1':
    class_weights =  [0.6689810602050595,
                      1.4180794131137744,
                      1.2499839925533194]

elif fold=='fold_2':
    class_weights = [0.6395503885274618,
                     1.4008227869987053,
                     1.3840152857246992]
elif fold=='fold_3':
    # class_weights = [0.4596716336304325, 2.730686392729672, 2.1818543768747634]
    # boot up N, GP3, GP4 ratio 0.85
    class_weights = [0.6666002869103805, 
                     1.4151877180904922, 
                     1.2606674654315284]
elif fold=='fold_4':
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
    model_best_path = '.'
    slide_no_pos = 2
    verbose=1
    # train_pair_exp_file = '10slides_train_pair_exp_N_GP3_GP4.dat'
    # val_pair_RBF_file = '10slides_val_pair_exp_N_GP3_GP4.dat'

elif platform.system() == 'Linux':    
    base_directory = '.'
    base_patch = job
    dataset_directory = os.path.join(base_directory, base_patch)
    model_best_path = Path(os.path.abspath('.'))  
    slide_no_pos = 2
    verbose=2

else:
    model_best_path = Path(os.path.abspath(r"C:\Users\kasikrit\OneDrive - Prince of Songkla University\PCa\models"))
    # dataset_directory = Path(os.path.abspath('C:\PCa-Kasikrit\segmentation\slides'))
    dataset_directory = Path(os.path.abspath(r"E:\slides\svs"))
    slide_no_pos = 3
    verbose=1

file_name_pos = slide_no_pos + 2


# from pathlib import PureWindowsPath, PurePosixPath
# path = PureWindowsPath(r"E:\slides\svs")
# PurePosixPath('.//Slides', *path.parts[1:])

#%%
def make_pair1(img,label,dataset):
    pairs = []
    for im in img:
        pairs.append( (im , dataset/ label / (im.stem + ".png")) )
    
    return pairs

def make_pair(imgs,labels):
    pairs = []
    for img, mask in zip(imgs, labels):
        pairs.append( (img, mask) )
    
    return pairs

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
print('segmentation model version: ', sm.__version__)

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
import pickle
# train_pair_exp_file = '10slides_train_pair_inc_N_GP3_GP4.dat'
with open (train_pair_exp_file, 'rb') as fp:
    train_pair_RBF = pickle.load(fp)
print('\nloaded train_pair_RBF: ', len(train_pair_RBF)) 
print(train_pair_RBF[:5])

# path = train_pair_RBF[0]
# path3 = PureWindowsPath(dataset_directory, path[0])
# print(path3)
# path4 = PurePosixPath(path3)
# print(path4)

# path = '/Volumes/MyPassportB/slides/1862/mask20x/59_57.png'
# path1 =  PurePosixPath(path)

#%%
import pickle
# val_pair_RBF_file = '10slides_val_pair_inc_N_GP3_GP4.dat'
with open (val_pair_exp_file, 'rb') as fp:
    val_pair_RBF = pickle.load(fp)
print('\nloaded val_pair_RBF: ', len(val_pair_RBF)) 
    
#%%
import pickle
with open (test_pair_exp_file, 'rb') as fp:
    test_pair_RBF = pickle.load(fp)
print('loaded test_pair_RBF: ', len(test_pair_RBF)) 

#%% #5
import utility
print('\nCreate train_generator_RBF')
train_generator_RBF = utility.DataGenerator(
    dataset_directory,
    train_pair_RBF,
    num_classes=n_classes,
    batch_size=batch_size, 
    dim=(patch_size, patch_size, 3),
    # dim=(384, 384, 3), #PSPNet
    # dim=(target_size, target_size, 3),
    shuffle=True,
    augmentation=utility.get_training_augmentation(),
    inference=False,
    preprocessing=utility.get_preprocessing(preprocess_input),
    rescale=None,
    )
train_steps_RBF = train_generator_RBF.__len__()
print('train_steps_RBF: ', train_steps_RBF)

#%%
image_number = random.randint(0, train_steps_RBF)
print('random image number: ', image_number)
X_train, y_train = train_generator_RBF.__getitem__(image_number)
print(X_train.shape, y_train.shape)
y_train_argmax = np.argmax(y_train, axis=3).astype('uint8')

#
utility.sanity_check(X_train, y_train_argmax, 
    note='Train exp ', 
    batch_size=batch_size//check_ratio)
    

#%% #6
print('\nCreate val_generator_RBF')
val_generator_RBF = utility.DataGenerator(
    dataset_directory,
    val_pair_RBF,
    num_classes=n_classes,
    batch_size=batch_size, 
    dim=(patch_size, patch_size, 3),
    # dim=(384, 384, 3), #PSPNet
    shuffle=True,
    augmentation=utility.get_training_augmentation(),
    preprocessing=utility.get_preprocessing(preprocess_input),
    rescale=None,
    )
val_steps_RBF = val_generator_RBF.__len__()
print('val_steps_RBF: ', val_steps_RBF)

#%
image_number = random.randint(0, val_steps_RBF-1)
print('random image number: ', image_number)
X_val, y_val = val_generator_RBF.__getitem__(image_number)
print(X_val.shape, y_val.shape)
y_val_argmax = np.argmax(y_val, axis=3).astype('uint8')

utility.sanity_check(X_val, y_val_argmax,
    note='Val exp ', batch_size=batch_size//check_ratio)

#%%
test_generator_RBF = utility.DataGenerator(
        dataset_directory,
        test_pair_RBF,
        num_classes=n_classes,
        batch_size=batch_size, 
        dim=(patch_size, patch_size, 3),
        shuffle=True,
        augmentation=utility.get_training_augmentation(),
        preprocessing=utility.get_preprocessing(preprocess_input),
        )
test_steps_RBF = test_generator_RBF.__len__()
print('test_steps_RBF: ', test_steps_RBF)

#%
image_number = random.randint(0, test_steps_RBF)
print('random image number: ', image_number)
X_test, y_test = test_generator_RBF.__getitem__(image_number)
print(X_test.shape, y_test.shape)
y_test_argmax = np.argmax(y_test, axis=3).astype('uint8')

# utility.sanity_check(X_test, y_test_argmax, note='Test exp ', batch_size=batch_size//check_ratio)

#%%
if CW == True:
    print("\nGet all labels preparing for computing class weights")
    train_labels_RBF = utility.get_all_labels(train_generator_RBF)
    print('train_labels_RBF.shape: ', train_labels_RBF.shape)
    train_labels_RBF_flatten = train_labels_RBF.flatten()
    print('train_labels_RBF_flatten.shape: ', train_labels_RBF_flatten.shape)
    print('Calculating class weights...')
    class_weights = utility.cal_class_weight(train_labels_RBF_flatten, classes)
    print('class_weights: ', class_weights)
    

#%%
print('\nclass_weights = ', class_weights)

dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
focal_loss = sm.losses.CategoricalFocalLoss()
# focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, 
# above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss
# sm_total_loss = sm.losses.categorical_focal_dice_loss  

# sm_metrics = [
#             sm.metrics.IOUScore(
#             # threshold=0.5, 
#             class_weights=class_weights
#             ), 
#             sm.metrics.FScore(
#             # threshold=0.5, 
#             class_weights=class_weights
#              )
#            ]


metrics_list = [
            #'accuracy', 
            sm.metrics.IOUScore(
                        # threshold=0.5, 
                        class_weights=class_weights
                        ), 
            sm.metrics.FScore(
                        # threshold=0.5, 
                        class_weights=class_weights
                         ), 
            #iou_coef1, dice_coef1,
            iou, jaccard_distance, total_loss,
            dice_coef2, 
            #dice_coef3, #error line 140 y_true_flatten = np.asarray(y_true).astype(np.bool)
            precision, recall, accuracy,
            ] 

#%% #4
import models
print('\nDefine and config model') 
FILTER_NUM=64
dilation_rates = [1, 3, 5, 7, 11, 13]
model = models.Dilated_Attention_ResUNet(input_shape=(patch_size, patch_size, 3),
                      NUM_CLASSES=n_classes, 
                      FILTER_NUM=FILTER_NUM,
                      # FILTER_SIZE=3,
                      dropout_rate=dropout_rate, 
                      # batch_norm=True,
                      activation=activation,
                      dilation_rates=dilation_rates,
                      )
print('Model: ', model.name)
print('FILTER_NUM: ', FILTER_NUM)
print('dilation_rates: ', dilation_rates)

# print(model.summary())

#%%
# print("\nTransfer trained weights from: ", backup_model_best_tr)
# unet_model = load_model(backup_model_best_tr, compile=False)
# print('\nTransfer completed.')

#%%
model.compile(optimizer=Adam(learning_rate = init_lr), 
        loss=total_loss, # run properly
        # loss = utility.jaccard_distance, # run properly on NT and 100 slides
        metrics = metrics_list,
        )
print('\nloss = total_loss')
print('\nmetrics = ', metrics_list)

#%%
print(model.summary())
                                        
#%%
backup_model_best = os.path.join(model_best_path, f'{model_name}.hdf5')

print('\nbackup_model_best: ', backup_model_best)
mcp2 = ModelCheckpoint(backup_model_best, save_best_only=True) 

reLR = ReduceLROnPlateau(
                  # monitor='val_iou_coef1',
                  # monitor='val_dice_coef2',
                  # monitor='val_jaccard_distance',
                  moitor='val_total_loss',
                  factor=0.8,
                  patience=5,
                  verbose=1,
                  mode='auto',
                  #min_lr = 0.00001,#1e-5
                  min_lr = init_lr/100,
                  # min_lr = init_lr/epochs,
                )

early_stop = EarlyStopping(
            patience=10,
            # patience=epochs//patience_ratio,  # Patience should be larger than the one in ReduceLROnPlateau
            min_delta=init_lr/100,
            # min_delta = init_lr/epochs,
            )


print("\nreLR monitor: val_total_loss 0.8")
# print("\nearly_stop: patience=9, min_delta=init_lr/100")


#%%
print("\n\nPerform training...");
print(data_note)
t3 = datetime.now()
with tf.device('/device:GPU:0'):
    model_history = model.fit(
            train_generator_RBF, 
            # steps_per_epoch=train_steps_RBF,
            validation_data=val_generator_RBF,   
            # validation_steps=val_steps_RBF,
            epochs=epochs,
            verbose=verbose,
            callbacks=[
                reLR,
                mcp2,
                early_stop, 
                tqdm_callback, 
                # PlotLossesKeras(),
                ],
            )
t4 = datetime.now() - t3
print("\nTraining time: ", t4)

del model

os.mkdir(model_name)
print("Create dir: ", model_name)

#%%
# convert the history.history dict to a pandas DataFrame and save as csv for
# future plotting

model_history_df = pd.DataFrame(model_history.history) 

history_file = f'{model_name}_history_df.csv'
history_file_path = os.path.join(model_name, history_file)
with open(history_file_path, mode='w') as f:
    model_history_df.to_csv(f)  
print("\nSaved: ", history_file_path)

#%%
# if LOGSCREEN==True:

history1 = model_history

#%plot the training and validation accuracy and loss at each epoch
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(loss) + 1)
fig = plt.figure(figsize=(8, 6), dpi=600)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'orange', label='Validation loss')
plt.title('Training and validation loss ' + data_note)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
loss_file = f'loss-{model_name}.png'
loss_file_path = os.path.join(model_name, loss_file)
fig.savefig(loss_file_path)

#%%
acc = history1.history['dice_coef2']
val_acc = history1.history['val_dice_coef2']
fig = plt.figure(figsize=(8, 6), dpi=600)
plt.plot(epochs, acc, 'b', label='Training Dice_coef2')
plt.plot(epochs, val_acc, 'orange', label='Validation Dice_coef2')
plt.title('Training and validation Dice_coef ' + data_note)
plt.xlabel('Epochs')
plt.ylabel('Dice_coef2')
plt.legend()
plt.show()
dice_file = f'dice_coef2-{model_name}.png'
dice_file_path = os.path.join(model_name, dice_file)
fig.savefig(dice_file_path)

#%%
acc = history1.history['iou']
val_acc = history1.history['val_iou']
fig = plt.figure(figsize=(8, 6), dpi=600)
plt.plot(epochs, acc, 'b', label='Training IoU_coef')
plt.plot(epochs, val_acc, 'orange', label='Validation IoU_coef')
plt.title('Training and validation IoU_coef ' + data_note)
plt.xlabel('Epochs')
plt.ylabel('IoU_coef')
plt.legend()
plt.show()
iou_file = f'iou-{model_name}.png'
iou_file_path = os.path.join(model_name, iou_file)
fig.savefig(iou_file_path)

#%%
acc = history1.history['f1-score']
val_acc = history1.history['val_f1-score']
fig = plt.figure(figsize=(8, 6), dpi=600)
plt.plot(epochs, acc, 'b', label='Training F1-score')
plt.plot(epochs, val_acc, 'orange', label='Validation F1-score')
plt.title('Training and validation F1-Score ' + data_note)
plt.xlabel('Epochs')
plt.ylabel('F1-score')
plt.legend()
plt.show()
f1_file = f'f1_score-{model_name}.png'
f1_file_path = os.path.join(model_name, f1_file)
fig.savefig(f1_file_path)

#%%
lr =  history1.history['lr']
epochs = range(1, len(loss) + 1)
fig = plt.figure(figsize=(8, 6), dpi=600)
plt.plot(epochs, lr, 'b', label='Training LR')
plt.title('Trainining learning rate ' + data_note)
plt.xlabel('Epochs')
plt.ylabel('LR')
plt.legend()
plt.show()
lr_file = f'LR-{model_name}.png'
lr_file_path = os.path.join(model_name, lr_file)
fig.savefig(lr_file_path)

#%%
acc = history1.history['jaccard_distance']
val_acc = history1.history['val_jaccard_distance']
fig = plt.figure(figsize=(8, 6), dpi=600)
plt.plot(epochs, acc, 'b', label='Training jaccard_distance')
plt.plot(epochs, val_acc, 'orange', label='Validation jaccard_distance')
plt.title('Training and validation jaccard_distance ' + data_note)
plt.xlabel('Epochs')
plt.ylabel('jaccard_distance')
plt.legend()
plt.show()
f1_file = f'jaccard_distance-{model_name}.png'
f1_file_path = os.path.join(model_name, f1_file)
fig.savefig(f1_file_path)
  
#%%#####################################################
# backup_model_best = 'C:/Users/kasikrit/OneDrive/models/backup_model_fold_4-train-3c-SL-val-dilated-att-res-unet-120ep-20230514-1127.hdf5'
print('\nbackup_model_best: ', backup_model_best)
best_model = load_model(backup_model_best, compile=False)
# print(best_model.summary())
print('\nLoaded: ' , backup_model_best)


best_model.compile(optimizer=Adam(learning_rate = init_lr), 
        loss=total_loss, # run properly
        metrics = metrics_list
        )
print('\nloss = (total_loss =  dice_loss + (1 * focal_loss)')
print('\nmetrics = ', metrics_list)


#%%
print("\nEvaluate model for the val set")

with tf.device('/device:GPU:0'):
    scores = best_model.evaluate(val_generator_RBF, 
                             batch_size=batch_size, 
                             verbose=verbose)

print()     
for metric, value in zip(best_model.metrics_names, scores):
    print("mean {}: {:.4}".format(metric, value))
    
print()    
# for metric, value in zip(best_model.metrics_names, scores):
#     print("mean {}: {:.2}".format(metric, value))

#%%


#%% #7
# import utility
# print('\nCreate val_generator_RBF_inf')
# val_generator_RBF_inf = utility.DataGenerator(
#     val_pair_RBF,
#     num_classes=n_classes,
#     batch_size=batch_size, 
#     dim=(patch_size, patch_size, 3),
#     # dim=(target_size,target_size,3),
#     # dim=(384, 384, 3), #PSPNet
#     shuffle=True,
#     inference=True,
#     augmentation=utility.get_training_augmentation(),
#     preprocessing=utility.get_preprocessing(preprocess_input),
#     )
# val_steps_RBF = val_generator_RBF_inf.__len__()
# print('val_steps_RBF: ', val_steps_RBF)

#%%
# image_number = random.randint(0, val_steps_RBF)
# print('random image number: ', image_number)
# X_val, y_val, pair_idx_val = val_generator_RBF_inf.__getitem__(image_number)
# print(X_val.shape, y_val.shape)
# y_val_argmax = np.argmax(y_val, axis=3).astype('uint8')

# for i in range(0, len(y_val_argmax)):
#     (unique, counts) = np.unique(y_val_argmax[i], return_counts=True)
#     print(unique, counts)

# sanity_check_with_patch_id(X_val, y_val_argmax, note='Val ', 
#             pair_idx=pair_idx_val, 
#             pairs=val_pair_RBF)

#%%
# print("\nPredict for the val set")
# y_pred = best_model.predict(
#                     X_val,   
#                     batch_size=batch_size, 
#                     verbose=verbose)

# y_pred_argmax = np.argmax(y_pred, axis=3).astype('uint8')
# print('y_pred_argmax.shape: ', y_pred_argmax.shape)
# print('y_val_argmax.shape: ', y_val_argmax.shape)

# (unique, counts) = np.unique(y_pred_argmax, return_counts=True)
# print(unique, counts)

#%%
# print("\nInference Plot\n")
# utility.inference_plot_v2(X_val, y_val_argmax, y_pred_argmax, 
#                note='Val ',
#                pair_idx=pair_idx_val,
#                pairs=val_pair_RBF,
#                slide_no_pos=slide_no_pos
#                )

#%%
# print("Evaluate for only random one batch")
# y_true = np.array(y_val_argmax, dtype='float32')
# y_pred = np.array(y_pred_argmax, dtype='float32')

# print('dice_coef:', dice_coef(y_true, y_pred))
# print('jacard_coef:', jacard_coef(y_true, y_pred))

# print('jacard_coef_loss:', jacard_coef_loss(y_true, y_pred))
# print('dice_coef_loss:', dice_coef_loss(y_true, y_pred))

#%%
# print("\nEvaluate the whole val set")
# # exe1 = datetime.now()
# # sample_size = 10
# # X_val_list= []
# y_val_list = []
# y_val_pred_list = []
# pair_idx_val_list = []
   
# # for batch in range(1, val_steps_RBF+1):
# print('\nPredicting...')
# for batch in tqdm(range(val_generator_RBF_inf.__len__())): 
#     X_val, y_val, pair_idx_val = val_generator_RBF_inf.__getitem__(batch)
#     # print('\n', X_val.shape, y_val.shape, len(pair_idx_val))
    
#     y_val_argmax = np.argmax(y_val, axis=3).astype('uint8')
#     # print('y_val_argmax.shape: ', y_val_argmax.shape)
#     # verbose=2   
#     y_val_pred = best_model.predict(
#                     X_val,   
#                     batch_size=batch_size, 
#                     verbose=2)

#     # y_pred_argmax = np.argmax(y_pred, axis=3).astype('uint8')
#     # print('\ny_pred_argmax.shape: ', y_pred_argmax.shape)
    

#     # X_val_list.append(X_val)
#     y_val_list.append(y_val_argmax)
#     y_val_pred_list.append(y_val_pred)
#     pair_idx_val_list.append(pair_idx_val)

# #%
# # X_val = np.concatenate(X_val_list, axis=0)
# y_val_true = np.concatenate(y_val_list, axis=0)
# y_val_pred = np.concatenate(y_val_pred_list, axis=0)
# pair_idx_val = np.concatenate(pair_idx_val_list, axis=0)
# print('y_val_true.shape, y_val_pred.shape, pair_idx_val.shape')
# print(y_val_true.shape, y_val_pred.shape, pair_idx_val.shape)
# del y_val_list
# del y_val_pred_list

# y_val_pred_argmax = np.argmax(y_val_pred, axis=3) #int64
# print(y_val_true.shape, y_val_pred_argmax.shape)


# # np.save("y_val_true.npy", y_val_true)
# #np.save("y_val_pred.npy", y_val_pred)
# del y_val_pred

# #%%
# print("\nVal: Calculate iou_coef1 and dice_coef1 manually")
# with tf.device('/device:GPU:0'):
#     val_iou_coef1 = iou_coef1(np.array(y_val_true, dtype='float32'), 
#                                   np.array(y_val_pred_argmax, dtype='float32')
#                                 )
                                  
#     val_dice_coef1  = dice_coef1(np.array(y_val_true, dtype='float32'), 
#                                 np.array(y_val_pred_argmax, dtype='float32')
#                                 )
# # print('val_iou_coef1:', val_iou_coef1)
# # print('val_dice_coef1:', val_dice_coef1)

# print('val_iou_coef1: ', np.mean(val_iou_coef1))
# print('val_dice_coef1: ', np.mean(val_dice_coef1))

#% Resource exhausted
# values2c = utility.compute_MeanIoU_2c(y_val_true, y_val_pred_argmax)
# print(values2c)

#%%
# print("\nCompute MeanIoU for Val ratio")
# for i in [1,2,3]:
#     random_ratio = 0.10
#     random_size = int(len(y_val_true) * random_ratio)
#     batch_idx = random.sample(range(1, len(y_val_true)),
#                               random_size)
#     print(i, 'len(batch_idx): ', len(batch_idx), random_ratio)
    
#     iou_val_3c = utility.compute_MeanIoU_3c(y_val_true[batch_idx],
#                           y_val_pred_argmax[batch_idx])
#     # print('len(batch_idx): ', len(batch_idx), random_ratio)
#     print(iou_val_3c, '\n')
   
  
#%%
print('\n\nTEST')
exe3 = datetime.now() 
print("\nEvaluate model for the test set by mode.evaluate()")

with tf.device('/device:GPU:0'):
    verbose=2
    scores = best_model.evaluate(test_generator_RBF, 
                              batch_size=batch_size, 
                              verbose=verbose)

    
for metric, value in zip(best_model.metrics_names, scores):
    print("mean {}: {:.4}".format(metric, value))

print()
    
exe4 = datetime.now() - exe3
print('Execution times: ', data_note, exe4, '\n')

#%%
# print("\nCreate test generator for inference")
# test_generator_RBF_inf = utility.DataGenerator(
#     test_pair_RBF,
#     num_classes=n_classes,
#     batch_size=batch_size, 
#     dim=(patch_size, patch_size, 3),
#     # dim=(target_size,target_size,3),
#     shuffle=False,
#     inference=True,
#     augmentation=utility.get_training_augmentation(),
#     preprocessing=utility.get_preprocessing(preprocess_input),
#     )

# test_steps_RBF = test_generator_RBF_inf.__len__()
# print('test_steps_RBF: ', test_steps_RBF)

# image_number = random.randint(0, test_steps_RBF)
# print('random image number: ', image_number)
# X_inf, y_inf, pair_idx_inf = test_generator_RBF_inf.__getitem__(image_number)
# print(X_inf.shape, y_inf.shape)
# y_inf_argmax = np.argmax(y_inf, axis=3).astype('uint8')

# for i in range(0, len(y_inf_argmax)):
#     (unique, counts) = np.unique(y_inf_argmax[i], return_counts=True)
#     print(unique, counts)
    
# sanity_check_with_patch_id(X_inf, y_inf_argmax, note='Test ', 
#             pair_idx=pair_idx_inf, 
#             pairs=test_pair_RBF)

#%%
# print("\n\nEvaluate the whole TEST set for manual evaluation")
# t3 = datetime.now()
# # sample_size = 10
# X_test_list= []
# y_test_list = []
# y_test_pred_list = []
# pair_idx_test_list = []
   
# for batch in tqdm(range(test_generator_RBF_inf.__len__())):
#     # print('\nPredicting batch: ', batch)
#     X_test, y_test, pair_idx_test = test_generator_RBF_inf.__getitem__(batch)
      
#     # print(X_test.shape, y_test.shape, len(pair_idx_test))
#     y_test_argmax = np.argmax(y_test, axis=3).astype('uint8')
#     # verbose=2
#     with tf.device('/device:GPU:0'):
#         y_test_pred = best_model.predict(
#                         # test_generator_RBF.__getitem__(image_number),
#                         X_test,   
#                         batch_size=batch_size, 
#                         verbose=verbose)

#     # y_pred_argmax = np.argmax(y_pred, axis=3).astype('uint8')
#     # print('\ny_pred_argmax.shape: ', y_pred_argmax.shape)
#     # print('y_test_argmax.shape: ', y_test_argmax.shape)
   
#     X_test_list.append(X_test)
#     y_test_list.append(y_test_argmax)
#     y_test_pred_list.append(y_test_pred)
#     pair_idx_test_list.append(pair_idx_test)

# X_test = np.concatenate(X_test_list, axis=0)     
# y_test_true = np.concatenate(y_test_list, axis=0)
# y_test_pred = np.concatenate(y_test_pred_list, axis=0)
# pair_idx_test = np.concatenate(pair_idx_test_list, axis=0)

# print('\ny_test_true.shape, y_test_pred.shape, pair_idx_test.shape')
# print(y_test_true.shape, y_test_pred.shape, pair_idx_test.shape)
# print(X_test.shape)
# #del y_test_pred_list

# y_test_pred_argmax = np.argmax(y_test_pred, axis=3)
# print(y_test_pred_argmax.shape)

# t4 = datetime.now() - t3
# print('Execution times: ', t4, '\n')

# #%
# np.save(os.path.join(model_best_path, 'X_test_20sldes.npy'),
#                       X_test)

# y_test_true_file = os.path.join(model_best_path,
#     'y-test-true-' + model_name + '.npy')
# np.save(y_test_true_file, y_test_true)

# y_test_pred_file = os.path.join(model_best_path,
#     'y-test-pred-' + model_name + '.npy')
# np.save(y_test_pred_file, y_test_pred)
# # #del y_test_pred

# print('Saved complete')

#%%
# print("\nTest: Calculate iou_coef1 and dice_coef1 manually")
# with tf.device('/device:GPU:0'):
#     test_iou_coef1 = iou_coef1(np.array(y_test_true, dtype='float32'), 
#                                   np.array(y_test_pred_argmax, dtype='float32')
#                                 )
                                  
#     test_dice_coef1  = dice_coef1(np.array(y_test_true, dtype='float32'), 
#                                 np.array(y_test_pred_argmax, dtype='float32')
#                                 )

# # print('test_iou_coef1:', test_iou_coef1)
# # print('test_dice_coef1:', test_dice_coef1)

# print('Mean test_iou_coef1:', np.mean(test_iou_coef1))
# print('Mean test_dice_coef1:', np.mean(test_dice_coef1))

# #%%
# import numpy as np
# from sklearn.model_selection import train_test_split

# X_train1, X_test_sam, y_train1, y_test_sam = train_test_split(
#     X_test, y_test_true, test_size=0.3, random_state=42)

# print(X_test_sam.shape, y_test_sam.shape)

#%%
# verbose=1
# with tf.device('/device:GPU:0'):
#     y_test_sam_pred = best_model.predict(
#                     # test_generator_RBF.__getitem__(image_number),
#                     X_test_sam,   
#                     batch_size=batch_size, 
#                     verbose=verbose)

# y_test_sam_pred_argmax = np.argmax(y_test_sam_pred, axis=3)
# print(y_test_sam_pred_argmax.shape)

# #%%
# iou_test_3c = utility.compute_MeanIoU_3c(y_test_sam_pred_argmax,
#                                       y_test_sam)
# # print('len(batch_idx): ', len(batch_idx), random_ratio)
# print(iou_test_3c, '\n')

#%%
# print("\nCompute MeanIoU for Test ratio")
# for i in [1,2,3]:
#     random_ratio = 0.2
#     random_size = int(len(y_test_true) * random_ratio)
#     batch_idx = random.sample(range(1, len(y_test_true)), random_size)
#     print(i, 'len(batch_idx): ', len(batch_idx), random_ratio)
    
#     iou_test_3c = utility.compute_MeanIoU_3c(y_test_true[batch_idx],
#                                           y_test_pred_argmax[batch_idx])
#     # print('len(batch_idx): ', len(batch_idx), random_ratio)
#     print(iou_test_3c, '\n')


#%%
# print("\nPredict for the test set")
# image_number = random.randint(0, test_steps_RBF-1)
# print('random image number: ', image_number)
# X_inf, y_inf, pair_idx_inf = test_generator_RBF_inf.__getitem__(image_number)
# print(X_inf.shape, y_inf.shape)
# y_inf_argmax = np.argmax(y_inf, axis=3).astype('uint8')


# y_pred = best_model.predict(
#                     # test_generator_RBF.__getitem__(image_number),
#                     X_inf,   
#                     batch_size=batch_size, 
#                     verbose=verbose)

# y_pred_argmax = np.argmax(y_pred, axis=3).astype('uint8')
# print('y_pred_argmax.shape: ', y_pred_argmax.shape)

# (unique, counts) = np.unique(y_pred_argmax, return_counts=True)
# print(unique, counts)


# print("\nInference Plot\n")
# import utility
# utility.inference_plot_v2(X_inf, y_inf_argmax, y_pred_argmax, 
#                 note='Test ',
#                 pair_idx=pair_idx_inf,
#                 pairs=test_pair_RBF,
#                 slide_no_pos=slide_no_pos,
#                 )


#%%
start_exe2 = datetime.now() - start_exe1
print('Execution times: ', data_note, start_exe2, '\n')

#%%
if LOGSCREEN:
    sys.stdout = old_stdout
    log_file.close()   

#%%


