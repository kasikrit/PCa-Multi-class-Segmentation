import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm

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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical ,Sequence
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adadelta, Nadam ,Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os, platform, sys
from pathlib import Path
from tqdm import tqdm
from random import sample, choice
from PIL import Image

from datetime import datetime 
import random
import glob
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

import utility

#%
default_n_classes = 5
default_classes = [0, 1, 2, 3, 4]

n_classes = 3
classes = [0, 1, 2]
labels = ['Normal', 'GP3', 'GP4']

seed = 1337
epochs = 100
# batch_size = 8
batch_size = 64
check_ratio = 16 # amont to plot = batch_size/check_ratio
img_size = patch_size = 256
target_size = patch_size
# fold = 'fold-4'
target_size = patch_size
dropout_rate = 0.25
# dropout_rate = 0.4
# split_train_val = 0.35
lr = 1e-4
activation = 'softmax'

LOGSCREEN = False
# LOGSCREEN = True
CW = True
# CW = False
# PROB = True
# PROB = False
# BACKBONE = 'resnet34' #1
BACKBONE = 'seresnet50'
# BACKBONE = 'inceptionresnetv2'
# BACKBONE = 'mobilenet'
# BACKBONE = 'seresnext50'
# BACKBONE = 'seresnext101'
# BACKBONE = 'att-res-unet'

fold='fold_4' #2
TRAIN=True
VAL=True
TEST=False

preprocess_input = sm.get_preprocessing(BACKBONE)

data_note = f'NT-{fold}-train-3c-{BACKBONE}-{epochs}ep'

train_pair_exp_file = f'train_pair_exp_{fold}_N_GP3_GP4_relative_path.dat'
val_pair_exp_file = f'val_pair_exp_{fold}_N_GP3_GP4_relative_path.dat'
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
    # dataset_directory = Path(os.path.abspath('/Volumes/MyPassportG/slides/svs/'))
    # base_directory = '/Users/kasikritdamkliang/Datasets/PCa'
    base_directory = '/Volumes/MyPassportB/'
    base_patch = 'slides'   
    dataset_directory = os.path.join(base_directory, base_patch)
    model_best_path = '.'
    slide_no_pos = 4
    verbose=1

elif platform.system() == 'Linux':    
    base_directory = '.'
    dataset_directory = os.path.join(base_directory, '10slides')
    model_best_path = Path(os.path.abspath('.'))  
    slide_no_pos = 2
    verbose=1

else:
    model_best_path = Path(os.path.abspath(r'C:\Users\kasikrit\OneDrive - Prince of Songkla University\PCa\models'))
    # dataset_directory = Path(os.path.abspath('C:\PCa-Kasikrit\segmentation\slides'))
    dataset_directory = Path(os.path.abspath(r'E:\slides\svs'))
    slide_no_pos = 3
    file_name_pos = slide_no_pos + 2
    verbose=1

file_name_pos = slide_no_pos + 2

# model_file = 'backup_model_train-4c-att-res-unet-model-100ep-20220926-1904.hdf5'
# backup_model_best_tr = os.path.join(model_best_path, model_file)

#%
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

def read_patches(slide_list, patch_type):
    patches = []
    for slide in slide_list:
        image_directory = os.path.join(
        dataset_directory,
        slide,
        patch_type,
        '*.png'
        )
        images = list(glob.glob(image_directory, recursive=True ))
        images.sort()
        # print(images[:5])
        print('Reading: ', images[0].split(os.path.sep)[slide_no_pos])
        for patch in images:
            patch_path = patch.split(os.path.sep)
            patch_save = os.path.join(
                patch_path[slide_no_pos],
                patch_path[slide_no_pos+1],
                patch_path[slide_no_pos+2],
                )
            patches.append(patch_save)
        
    return patches

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
print('lr: ', lr)

#%%
train_slides = ['1360', '1323', '1924', '1876', '1985', '1913', '1894', '1952',
                '1878', '1940', '1937', '1844', '1977', '1848', '1346', '1682', 
                '1941', '1960', '1986', '1334', '1837', '1679', '1964', '1845', 
                '1967', '1984', '1332', '1920', '1928', '1898', '1822', '1355', 
                '1852', '1895', '1911', '1976', '1673', '1329', '1912', '1909', 
                '1994', '1849', '1678', '1975', '1343', '1942', '1997', '1917', 
                '1935', '1919', '1974', '1925', '1901', '1677', '1347', '1879', 
                '1972', '1817', '1970', '1990', '1916', '1869', '1961', '1923', 
                '1945', '1832', '1934', '1978', '1862', '1340', '1930', '1936', 
                '1828', '1676', '1948', '1915', '1910', '1788', '1338', '1684']

# Fold 1: Training set size = 64,           Validation set size = 16,           Ratio = 25.0
# Fold 2: Training set size = 64,           Validation set size = 16,           Ratio = 25.0
# Fold 3: Training set size = 64,           Validation set size = 16,           Ratio = 25.0
# Fold 4: Training set size = 64,           Validation set size = 16,           Ratio = 25.0
# Fold 5: Training set size = 64,           Validation set size = 16,           Ratio = 25.0

if fold=='fold_1':
            # Fold 1:
            # Validation set: 
    slide_list_val = [
            '1360',
            '1323',
            '1924',
            '1876',
            '1985',
            '1913',
            '1894',
            '1952',
            '1878',
            '1940',
            '1937',
            '1844',
            '1977',
            '1848',
            '1346',
            '1682',
            ]
    slide_list_train = [
            # Train set: 
            '1941',
            '1960',
            '1986',
            '1334',
            '1837',
            '1679',
            '1964',
            '1845',
            '1967',
            '1984',
            '1332',
            '1920',
            '1928',
            '1898',
            '1822',
            '1355',
            '1852',
            '1895',
            '1911',
            '1976',
            '1673',
            '1329',
            '1912',
            '1909',
            '1994',
            '1849',
            '1678',
            '1975',
            '1343',
            '1942',
            '1997',
            '1917',
            '1935',
            '1919',
            '1974',
            '1925',
            '1901',
            '1677',
            '1347',
            '1879',
            '1972',
            '1817',
            '1970',
            '1990',
            '1916',
            '1869',
            '1961',
            '1923',
            '1945',
            '1832',
            '1934',
            '1978',
            '1862',
            '1340',
            '1930',
            '1936',
            '1828',
            '1676',
            '1948',
            '1915',
            '1910',
            '1788',
            '1338',
            '1684',
            ]
elif fold == 'fold_2':            
            # Fold 2:
            # Validation set: 
    slide_list_val = [
            '1941',
            '1960',
            '1986',
            '1334',
            '1837',
            '1679',
            '1964',
            '1845',
            '1967',
            '1984',
            '1332',
            '1920',
            '1928',
            '1898',
            '1822',
            '1355',
            ]
    slide_list_train = [
            # Train set: 
            '1360',
            '1323',
            '1924',
            '1876',
            '1985',
            '1913',
            '1894',
            '1952',
            '1878',
            '1940',
            '1937',
            '1844',
            '1977',
            '1848',
            '1346',
            '1682',
            '1852',
            '1895',
            '1911',
            '1976',
            '1673',
            '1329',
            '1912',
            '1909',
            '1994',
            '1849',
            '1678',
            '1975',
            '1343',
            '1942',
            '1997',
            '1917',
            '1935',
            '1919',
            '1974',
            '1925',
            '1901',
            '1677',
            '1347',
            '1879',
            '1972',
            '1817',
            '1970',
            '1990',
            '1916',
            '1869',
            '1961',
            '1923',
            '1945',
            '1832',
            '1934',
            '1978',
            '1862',
            '1340',
            '1930',
            '1936',
            '1828',
            '1676',
            '1948',
            '1915',
            '1910',
            '1788',
            '1338',
            '1684',
            ]
elif fold=='fold_3':        
            # Fold 3:
            # Validation set: 
     slide_list_val = [
            '1852',
            '1895',
            '1911',
            '1976',
            '1673',
            '1329',
            '1912',
            '1909',
            '1994',
            '1849',
            '1678',
            '1975',
            '1343',
            '1942',
            '1997',
            '1917',
            ]
     slide_list_train = [
            # Train set: 
            '1360',
            '1323',
            '1924',
            '1876',
            '1985',
            '1913',
            '1894',
            '1952',
            '1878',
            '1940',
            '1937',
            '1844',
            '1977',
            '1848',
            '1346',
            '1682',
            '1941',
            '1960',
            '1986',
            '1334',
            '1837',
            '1679',
            '1964',
            '1845',
            '1967',
            '1984',
            '1332',
            '1920',
            '1928',
            '1898',
            '1822',
            '1355',
            '1935',
            '1919',
            '1974',
            '1925',
            '1901',
            '1677',
            '1347',
            '1879',
            '1972',
            '1817',
            '1970',
            '1990',
            '1916',
            '1869',
            '1961',
            '1923',
            '1945',
            '1832',
            '1934',
            '1978',
            '1862',
            '1340',
            '1930',
            '1936',
            '1828',
            '1676',
            '1948',
            '1915',
            '1910',
            '1788',
            '1338',
            '1684',
            ]
elif fold=='fold_4':           
            # Fold 4:
            # Validation set:
     slide_list_val = [
            '1935', #select to infer
            '1919',
            '1974',
            '1925',
            '1901',
            '1677',
            '1347',
            '1879',
            '1972',
            '1817',
            '1970',
            '1990',
            '1916',
            '1869',
            '1961',
            '1923', # select to infer
            ]
     slide_list_train = [
            # Train set: 
            '1360',
            '1323',
            '1924',
            '1876',
            '1985',
            '1913',
            '1894',
            '1952',
            '1878',
            '1940',
            '1937',
            '1844',
            '1977',
            '1848',
            '1346',
            '1682',
            '1941',
            '1960',
            '1986',
            '1334',
            '1837',
            '1679',
            '1964',
            '1845',
            '1967',
            '1984',
            '1332',
            '1920',
            '1928',
            '1898',
            '1822',
            '1355',
            '1852',
            '1895',
            '1911',
            '1976',
            '1673',
            '1329',
            '1912',
            '1909',
            '1994',
            '1849',
            '1678',
            '1975',
            '1343',
            '1942',
            '1997',
            '1917',
            '1945',
            '1832',
            '1934',
            '1978',
            '1862',
            '1340',
            '1930',
            '1936',
            '1828',
            '1676',
            '1948',
            '1915',
            '1910',
            '1788',
            '1338',
            '1684',
            ]
else:                     
            # Fold 5:
            # Validation set: 
     slide_list_val = [
            '1945',
            '1832',
            '1934',
            '1978',
            '1862',
            '1340',
            '1930',
            '1936',
            '1828',
            '1676',
            '1948',
            '1915',
            '1910',
            '1788',
            '1338',
            '1684',
            ]
     slide_list_train = [
            #Train set: 
            '1360',
            '1323',
            '1924',
            '1876',
            '1985',
            '1913',
            '1894',
            '1952',
            '1878',
            '1940',
            '1937',
            '1844',
            '1977',
            '1848',
            '1346',
            '1682',
            '1941',
            '1960',
            '1986',
            '1334',
            '1837',
            '1679',
            '1964',
            '1845',
            '1967',
            '1984',
            '1332',
            '1920',
            '1928',
            '1898',
            '1822',
            '1355',
            '1852',
            '1895',
            '1911',
            '1976',
            '1673',
            '1329',
            '1912',
            '1909',
            '1994',
            '1849',
            '1678',
            '1975',
            '1343',
            '1942',
            '1997',
            '1917',
            '1935',
            '1919',
            '1974',
            '1925',
            '1901',
            '1677',
            '1347',
            '1879',
            '1972',
            '1817',
            '1970',
            '1990',
            '1916',
            '1869',
            '1961',
            '1923',
            ]
#%    
slide_list_test = [
            '1887',
            '1968',
            '1969',
            '1890',
            '1827',
            '1989',
            '1929',
            '1833',
            '1954',
            '1939',
            '1921', 
            '1950',
            '1359',
            '1830',
            '1971',
            '1955',
            '1342',
            '1840',
            '1896',
            '1683'
            ]

#%%
# sorted(slide_list_train)
# print("\nTrain Slides:\n", slide_list_train)
print("Train slide len: ", len(slide_list_train))

##%%
image_patches_train = read_patches(slide_list_train, 'image20x')
print()
mask_patches_train = read_patches(slide_list_train, 'mask20x')

print(image_patches_train[199])
print(mask_patches_train[199])
##%%
train_pairs = make_pair(image_patches_train, mask_patches_train)
print("\nSanity check for train_pairs")
for i in range(0,10):
    x = random.randint(0, len(train_pairs)-1)
    print(x)
    print(train_pairs[x][0])
    print(train_pairs[x][1])
    print()

print('Before RBF:', len(image_patches_train), len(image_patches_train))


#%%
# sorted(slide_list_val)
# print("\nVal Slides:\n", slide_list_val)
print("Val slide len: ", len(slide_list_val))

#%
image_patches = read_patches(slide_list_val, 'image20x')
print()
mask_patches = read_patches(slide_list_val, 'mask20x')

print(image_patches[199])
print(mask_patches[199])

#%
val_pairs = make_pair(image_patches, mask_patches)
print("\nSanity check for val_pairs")
for i in range(0,10):
    x = random.randint(0, len(val_pairs)-1)
    print(x)
    print(val_pairs[x][0])
    print(val_pairs[x][1])
    print()

print('Before RBF:', len(image_patches), len(mask_patches))


#%%
# sorted(slide_list_val)
# print("\nVal Slides:\n", slide_list_val)
print("Test slide len: ", len(slide_list_test))

#%
image_patches = read_patches(slide_list_test, 'image20x')
print()
mask_patches = read_patches(slide_list_test, 'mask20x')

print(image_patches[199])
print(mask_patches[199])

#%
test_pairs = make_pair(image_patches, mask_patches)
print("\nSanity check for val_pairs")
for i in range(0,10):
    x = random.randint(0, len(test_pairs)-1)
    print(x)
    print(test_pairs[x][0])
    print(test_pairs[x][1])
    print()

print('Before RBF:', len(image_patches), len(mask_patches))


#%% 
def sanitycheck_pre(pair):
    print('Sanity check before preprocessing')
    for i in range(0,20):
        temp = choice(pair)
        img = img_to_array(load_img(temp[0], 
                    target_size=(img_size,img_size)),
                    dtype='uint8')
        # mask = cv2.imread(temp[1].as_posix(), 0)
        mask = cv2.imread(temp[1], 0)
        plt.figure(figsize=(10,10))
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(mask, cmap='gray')
        (unique, counts) = np.unique(mask, return_counts=True)
        xlabel = str(unique) + "\n" + str(counts)
        plt.xlabel(xlabel)
        plt.title('Sanity check for in train_pair')
        plt.show()
        
#%%
# sanitycheck_pre(Fg_N_50_pair)

#%%
def filter_pairs(pairs):
    print("\n Filter pairs...")
    pair_RBF = []
    N_pair = []
    GP3_pair = []
    GP4_pair = []
    for i, pair in enumerate(pairs):
        try:
            # print(i, pair)
            #mask = cv2.imread(pair[1].as_posix(), 0)
            patch_path = os.path.join(dataset_directory, pair[1])
            mask = cv2.imread(patch_path, 0)
            count, unique = np.unique(mask, return_counts=True)
            
            # soly BG
            if(count[0] == 0 and len(unique)==1): 
                # print("BG:", i, count, unique)
                pass
            # sole FG
            elif(count[0] == 1 and len(unique)==1): 
                # print("FG:", i, count, unique)
                pass
            #BG and FG
            elif( (count[0] == 0 and count[1] == 1) and len(unique)==2 ):
                # print("BG and FG:", i, count, unique)
                pass
            #solely benign
            elif(count[0] == 2 and len(unique)==1): 
                N_pair.append(pair)
                # print(count, unique)
                if(i%100==0):
                    print("Keep benign: ", i, count, unique)            
            # elif( (count[0] == 2 and count[1] == 3) and len(unique)==2 ): 
            #     N_GP3_pair.append(pair)
            #     # print(count, unique)
            #     if(i%100==0):
                    # print("Keep N+GP3: ", i, count, unique)
            # elif( (count[0] == 2 and count[1] == 4) and len(unique)==2 ): 
            #     N_GP4_pair.append(pair)
            #     # print(count, unique)
            #     if(i%100==0):
            #         print("Keep N+GP4: ", i, count, unique)
            #solely GP3
            elif(count[0] == 3 and len(unique)==1): 
                GP3_pair.append(pair)
                # print(count, unique)
                if(i%100==0):
                    print("Keep GP3: ", i, count, unique)
            #solely GP4
            elif(count[0] == 4 and len(unique)==1): 
                GP4_pair.append(pair)
                # print(count, unique)
                if(i%100==0):
                    print("Keep GP4: ", i, count, unique)        
            # # GP3+GP4
            # elif( (count[0] == 3 and count[1] == 4) and len(unique)==2 ):
            #     GP3_GP4_pair.append(pair)
            #     print("Keep GP3+GP4: ", i, count, unique)
            
            # N_GP3_GP4
            # elif( (count[0] == 2 and count[1] == 3 and count[2] == 4) 
            #     and len(unique)==3 ):
            #     N_GP3_GP4_pair.append(pair)
            #     # print(count, unique)
            #     print("Keep N+GP3+GP4: ", i, count, unique)
            
            #FG+N=> FG>50% pass
            elif( (count[0] == 1 and count[1] == 2) and len(unique)==2 ): 
                if unique[0] >= (256*256)/2 :
                    pass
            
            elif(5 in count):
                print('found 5: ', count)
                pass
            
            else:
                pair_RBF.append(pair)
        
        except IndexError:
            pass
        
        if(i%1000==0):
            print("Note: ", i, count, unique)
                
    return pair_RBF, N_pair, GP3_pair, GP4_pair
    
#%%   
        # Fg+GP3
        # try:
        #     if( (count[0] == 1 and count[1] == 3) and len(unique)==2 ):
        #         pairs.append(pair)
        #         # print(count, unique)
        #         if(i%1000==0):
        #             print("Keep: ", i, count, unique)
        # except IndexError:
        #     pass
        
        # try:
        #     if( (count[0] == 1 and count[1] == 4) and len(unique)==2 ): #FG+N
        #         pairs.append(pair)
        #         # print(count, unique)
        #         if(i%1000==0):
        #             print("Keep: ", i, count, unique)
        # except IndexError:
        #     pass
        
        # try:
        #     if( (count[0] == 2 and count[1] == 3) and len(unique)==2 ): 
        #         N_GP3_pair.append(pair)
        #         # print(count, unique)
        #         if(i%1000==0):
        #             print("Keep 2 3: ", i, count, unique)
        # except IndexError:
        #     pass
        
        # try:
        #     if( (count[0] == 2 and count[1] == 4) and len(unique)==2 ): 
        #         N_GP4_pair.append(pair)
        #         # print(count, unique)
        #         if(i%1000==0):
        #             print("Keep 2 4: ", i, count, unique)
        # except IndexError:
        #     pass
        
    
        # try:
        #     if( (count[0] == 3 and count[1] == 4) and len(unique)==2 ):
        #         GP3_GP4_pair.append(pair)
        #         # print(count, unique)
        #         if(i%1000==0):
        #             print("Keep 3 4: ", i, count, unique)
        # except IndexError:
        #     pass
        
        # Fg_N_GP3
        # try:
        #     if( (count[0] == 1 and count[1] == 2 and count[2] == 3) 
        #        and len(unique)==3 ):
        #         Fg_N_GP3_pair.append(pair)
        #         # print(count, unique)
        #         if(i%1000==0):
        #             print("Keep 1+2+3: ", i, count, unique)
        # except IndexError:
        #     pass
    
        # Fg_N_GP4
        # try:
        #     if( (count[0] == 1 and count[1] == 2 and count[2] == 4) 
        #         and len(unique)==3 ):
        #         Fg_N_GP4_pair.append(pair)
        #         # print(count, unique)
        #         if(i%1000==0):
        #             print("Keep 1+2+4: ", i, count, unique)
        # except IndexError:
        #     pass
    
        # N_GP3_GP4
        # try:
        #     if( (count[0] == 2 and count[1] == 3 and count[2] == 4) 
        #         and len(unique)==3 ):
        #         N_GP3_GP4_pair.append(pair)
        #         # print(count, unique)
        #         if(i%1000==0):
        #             print("Keep 2+3+4: ", i, count, unique)
        # except IndexError:
        #     pass
        
        # Fg_N_GP3_GP4
        # try:
        #     if( (count[0] == 1 and count[1] == 2 and 
        #          count[2] == 3 and count[3] == 4) 
        #        and len(unique)==4 ):
        #         Fg_N_GP3_GP43_pair.append(pair)
        #         # print(count, unique)
        #         if(i%1000==0):
        #             print("Keep 1+2+3: ", i, count, unique)
        # except IndexError:
        #     pass
        
        # if(count[0] == 3 and len(unique)==1): #GP3
        #     GP3_pair.append(pair)
        #     # print(count, unique)
        #     if(i%1000==0):
        #         print("Keep GP3: ", i, count, unique)
        
        # if(count[0] == 4 and len(unique)==1): #GP4
        #     GP4_pair.append(pair)  
        #     if(i%1000==0):
        #         print("Keep GP4: ", i, count, unique)
                
                
#%%
# print('\nGP3_pair: ', len(GP3_pair)) 
# print('\nGP4_pair: ', len(GP4_pair)) 
# print('\nN_pair: ', len(N_pair)) 
# print('\nFg_N_pair: ', len(Fg_N_pair))
# print('\nFg_GP3_pair: ', len(Fg_GP3_pair))
# print('\nFg_GP4_pair: ', len(Fg_GP4_pair))
# print('\n N_GP3_pair: ', len(N_GP3_pair))
# print('\n N_GP4_pair: ', len(N_GP4_pair))
# print('\n Fg_N_GP3_GP4_pair: ', len(Fg_N_GP3_GP43_pair))
# print('\n Fg_N_GP4_pair: ', len(Fg_N_GP4_pair))
# %
# pair = val_pair[14000][1]
# mask = cv2.imread(pair.as_posix(), 0)
# count, unique = np.unique(mask, return_counts=True)

#%%


#%% Train
if TRAIN:
    train_pair_RBF, N_pair, GP3_pair, GP4_pair = filter_pairs(train_pairs)
    
    #%
    pair_total = len(train_pair_RBF) + len(N_pair) + len(GP3_pair) + len(GP4_pair)
    print('pair_total: ', pair_total)
    
    #%
    in_samples = len(train_pair_RBF)*.30
    
    #%
    y = [1, 2, 3, 4]
    x = [len(train_pair_RBF),
        len(N_pair) * (in_samples//len(N_pair)),
        len(GP3_pair) * (in_samples//len(GP3_pair)),
        len(GP4_pair) * (in_samples//len(GP4_pair)),
        ]
     
    # Create a pandas dataframe
    df = pd.DataFrame({
        "labels": y,
        "counts": x})
    
    fig = plt.figure(figsize=(8, 6), dpi = 600) 
    
    ax = sns.barplot(x='labels', y="counts", data=df)
    
    #%
    N_new = int(in_samples//len(N_pair)-1)
    GP3_new = int(in_samples//len(GP3_pair)-1)
    GP4_new = int(in_samples//len(GP4_pair)-1)
        
    N_pair1 = N_pair + (N_pair * N_new)
    GP3_pair1= GP3_pair + (GP3_pair * GP3_new)
    GP4_pair1= GP4_pair + (GP4_pair * GP4_new)
    
    train_pair_exp = train_pair_RBF + \
                N_pair1 + \
                GP3_pair1 + \
                GP4_pair1
                
    print('train_pair_exp: ', len(train_pair_exp)) 
        
    utility.write_pairs(train_pair_exp_file, train_pair_exp)
     
    # train_pair_RBF_file = 'train_pair_RBF1_fold_3_N_GP3_GP4.dat'
    
    train_pair_exp = utility.load_pairs(train_pair_exp_file)

#%% Validation
if VAL:
    val_pair_RBF, N_pair, GP3_pair, GP4_pair = filter_pairs(val_pairs)
     
    #%
    val_pair_total = len(val_pair_RBF) + len(N_pair) + len(GP3_pair) + len(GP4_pair)
    print('val_pair_total: ', val_pair_total)
    in_samples = len(val_pair_RBF)*.30
    
    y = [1, 2, 3, 4]
    x = [len(val_pair_RBF),
        len(N_pair) * (in_samples//len(N_pair)),
        len(GP3_pair) * (in_samples//len(GP3_pair)),
        len(GP4_pair) * (in_samples//len(GP4_pair)),
        ]
     
    # Create a pandas dataframe
    df = pd.DataFrame({
        "labels": y,
        "counts": x})
    fig = plt.figure(figsize=(8, 6), dpi = 600) 
    
    ax = sns.barplot(x='labels', y="counts", data=df)
    
    N_new = int(in_samples//len(N_pair)-1)
    GP3_new = int(in_samples//len(GP3_pair)-1)
    GP4_new = int(in_samples//len(GP4_pair)-1)
        
    N_pair1 = N_pair + (N_pair * N_new)
    GP3_pair1= GP3_pair + (GP3_pair * GP3_new)
    GP4_pair1= GP4_pair + (GP4_pair * GP4_new)
    
    val_pair_exp = val_pair_RBF + \
                N_pair1 + \
                GP3_pair1 + \
                GP4_pair1
                
    print('val_pair_exp ', len(val_pair_exp)) 
      
    #%
    # val_pair_RBF_file = 'val_pair_RBF1_fold_3_N_GP3_GP4_MPB.dat'  
    utility.write_pairs(val_pair_exp_file, val_pair_exp)
     
    #%
    val_pair_exp = utility.load_pairs(val_pair_exp_file)

#%% Test 
if TEST:
    test_pair_RBF, FN_pair, GP3_pair, GP4_pair = filter_pairs(test_pairs)
     
    #%
    samples_ratio = 0.2
    
    test_pair_total = len(test_pair_RBF) + len(N_pair) + len(GP3_pair) + len(GP4_pair)
    print('test_pair_total: ', test_pair_total)
    in_samples = len(test_pair_RBF)*samples_ratio
    
    y = [1, 2, 3, 4]
    x = [len(test_pair_RBF),
        len(N_pair) * (in_samples//len(N_pair)),
        len(GP3_pair) * (in_samples//len(GP3_pair)),
        len(GP4_pair) * (in_samples//len(GP4_pair)),
        ]
     
    # Create a pandas dataframe
    df = pd.DataFrame({
        "labels": y,
        "counts": x})
    fig = plt.figure(figsize=(8, 6), dpi = 600) 
    
    ax = sns.barplot(x='labels', y="counts", data=df)
    
    N_new = int(in_samples//len(N_pair)-1)
    GP3_new = int(in_samples//len(GP3_pair)-1)
    GP4_new = int(in_samples//len(GP4_pair)-1)
        
    N_pair1 = N_pair + (N_pair * N_new)
    GP3_pair1= GP3_pair + (GP3_pair * GP3_new)
    GP4_pair1= GP4_pair + (GP4_pair * GP4_new)
    
    test_pair_exp = test_pair_RBF + \
                N_pair1 + \
                GP3_pair1 + \
                GP4_pair1
    
    print('test_pair_RBF: ', len(test_pair_RBF))           
    print('test_pair_exp: ', len(test_pair_exp)) 
        
    utility.write_pairs(test_pair_exp_file, test_pair_exp) 
    
    # test_pair_RBF_file = 'test_pair_RBF1.dat'
    test_pair_exp = utility.load_pairs(test_pair_exp_file) 
    

#%%
import utility
print('\nCreate train_generator')
train_generator = utility.DataGenerator(
    dataset_directory,
    train_pair_exp,
    # Fg_N_GP3_pair,
    # GP3_GP4_pair,
    # Fg_N_50_pair,
    # GP4_pair,
    # N_pair,
    num_classes=n_classes,
    batch_size=batch_size, 
    dim=(patch_size, patch_size, 3),
    # dim=(384, 384, 3), #PSPNet
    # dim=(target_size, target_size, 3),
    shuffle=True,
    augmentation=utility.get_training_augmentation(),
    inference=False,
    # preprocessing=utility.get_preprocessing(preprocess_input),
    )
train_steps = train_generator.__len__()
print('train_steps: ', train_steps)

#%%
image_number = random.randint(0, train_steps-1)
print('random image number: ', image_number)
X_train, y_train = train_generator.__getitem__(image_number)
print(X_train.shape, y_train.shape)
y_train_argmax = np.argmax(y_train, axis=3).astype('int8')

#
utility.sanity_check(X_train, y_train_argmax, 
                     note='Train ', 
                       batch_size=batch_size//check_ratio
                      # batch_size=batch_size
                     )
  
#%%
print('\nCreate val_generator')
val_generator = utility.DataGenerator(
    dataset_directory,
    val_pair_exp,
    # GP3_GP4_pair,
    #N_pair,
    # GP3_pair,
    # GP4_pair,
    num_classes=n_classes,
    batch_size=batch_size, 
    dim=(patch_size, patch_size, 3),
    # dim=(384, 384, 3), #PSPNet
    shuffle=True,
    augmentation=utility.get_training_augmentation(),
    # preprocessing=utility.get_preprocessing(preprocess_input),
    )
val_steps = val_generator.__len__()
print('val_steps: ', val_steps)

#%
image_number = random.randint(0, val_steps-1)
print('random image number: ', image_number)
X_val, y_val = val_generator.__getitem__(image_number)
print(X_val.shape, y_val.shape)
y_val_argmax = np.argmax(y_val, axis=3).astype('uint8')

utility.sanity_check(X_val, y_val_argmax, 
                     note='Val ', 
                     batch_size=batch_size//check_ratio)
  
#%%
# print('\nCreate test_generator')
# test_generator = utility.DataGenerator(
#     dataset_directory,
#     test_pair_exp,
#     num_classes=n_classes,
#     batch_size=batch_size, 
#     dim=(patch_size, patch_size, 3),
#     # dim=(384, 384, 3), #PSPNet
#     shuffle=True,
#     augmentation=utility.get_training_augmentation(),
#     # preprocessing=utility.get_preprocessing(preprocess_input),
#     )
# test_steps = test_generator.__len__()
# print('val_steps: ', test_steps)

# #%
# image_number = random.randint(0, test_steps-1)
# print('random image number: ', image_number)
# X_test, y_test = test_generator.__getitem__(image_number)
# print(X_test.shape, y_test.shape)
# y_test_argmax = np.argmax(y_test, axis=3).astype('uint8')

# utility.sanity_check(X_test, y_test_argmax, 
#                      note='Test ', 
#                      batch_size=batch_size//check_ratio)

#%%
# random_size = 0.99
# random_factor = int(val_steps_RBF * random_size)
# random_steps = random.sample(range(1, val_steps_RBF), random_factor)
# print(len(random_steps))
# #print(random_idx)

# val_labels_RBF = utility.get_all_labels(val_generator_RBF, random_steps)
# print('val_labels_RBF.shape: ', val_labels_RBF.shape)

# (unique, counts) = np.unique(val_labels_RBF, return_counts=True)
# print(unique, counts)
        
#%%
def get_labels(data_generator, steps):
    y_test_GP3 = []
    y_test_GP4 = []
    
    # for i in tqdm(range(data_generator.__len__())):
    for i in tqdm(steps):    
        X_test, y_test = data_generator.__getitem__(i)
        # print(i, y_test.shape)
        count, unique = np.unique(y_test, return_counts=True)

        if(count[0] == 1 and len(unique)==1): #GP3
            print("GP:", i, count, unique)
            y_test_GP3.append(y_test)
    
        if(count[0] == 2 and len(unique)==1): #GP4
            print("GP:", i, count, unique)
            y_test_GP4.append(y_test)
        
         
    # X_test1 = np.concatenate(X_test_list, axis=0)
    y_GP3 = np.concatenate(y_test_GP3, axis=0).astype('int8')
    y_GP4 = np.concatenate(y_test_GP4, axis=0).astype('int8')
    print(y_GP3.shape, y_GP4.shape)
    y_GP3_argmax = np.argmax(y_GP3, axis=3).astype('int8')
    y_GP4_argmax = np.argmax(y_GP4, axis=3).astype('int8')
    print(y_GP3_argmax.shape, y_GP4_argmax.shape)
   
    return y_GP3_argmax, y_GP4_argmax

#%%
# GP3_batch, GP4_batch = get_labels(train_generator_RBF, random_steps)

#%%
random_size = 0.85
random_factor = int(train_steps * random_size)
random_steps = random.sample(range(1, train_steps), 
                             random_factor)
print(len(random_steps))
#print(random_idx)

#%%
if CW == True:
    print("\nGet all labels preparing for computing class weights")
    train_labels = utility.get_all_labels(train_generator, random_steps)
    print('train_labels_RBF.shape: ', train_labels.shape)
    
    train_labels_flatten = train_labels.flatten()
    print('train_labels_RBF_flatten.shape: ', train_labels_flatten.shape)
    
    print('Calculating class weights...')
    class_weights = utility.cal_class_weight39(
        classes = classes, 
        labels = train_labels_flatten)
    print(fold, 'class_weights: ', class_weights)

#%%
start_exe2 = datetime.now() - start_exe1
print('Execution times: ', data_note, start_exe2, '\n')

#%%
if LOGSCREEN:
    sys.stdout = old_stdout
    log_file.close()   

#%%


