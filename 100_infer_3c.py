print("Val_total_loss batch_size 8\n")
# import mymodels

import utility
import models

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
from tensorflow import keras

import segmentation_models as sm #1.0.1

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
# import cv2
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.utils import to_categorical, Sequence
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout
# from tensorflow.keras.optimizers import Adadelta, Nadam, Adam
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os, platform
from pathlib import Path, PureWindowsPath, PurePosixPath
from tqdm import tqdm
from random import sample, choice
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from datetime import datetime 
import random
import glob #glob2

import tensorflow_addons as tfa
tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)

# from livelossplot import PlotLossesKeras

import os
import matplotlib

#FOCAL LOSS AND DICE METRIC
#Focal loss helps focus more on tough to segment classes.
# from focal_loss import BinaryFocalLoss, SparseCategoricalFocalLoss

from utility import ( 
    dice_coef, dice_coef_loss, jacard_coef,
    jacard_coef_loss, iou_coef1, dice_coef1,
    iou, jaccard_distance, dice_coef2, dice_coef3, precision, recall, accuracy)

import multi_class_IoU
import openslide
from patchify import patchify, unpatchify


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

start = datetime.now()

print('\ntf version: ', tf.__version__)
print('keras version: ', keras.__version__, '\n\n')

#%
def unique_counts(mask_data):
    (unique, counts) = np.unique(mask_data, return_counts=True)
    print((unique, counts) )

def example_function(thumb_gray):
    assert len(thumb_gray.shape) == 2
    assert isinstance(thumb_gray, np.ndarray)
    thresh = 100
    thumb_gray[thumb_gray > thresh] = 1
    thumb_gray[thumb_gray <= thresh] = 0
    assert np.sum((thumb_gray == 0) | (thumb_gray == 1)) == len(thumb_gray)
    return thumb_gray


def extract_annotation_region(slide_id, anotation_file, rule_file, output_dir):
    slide = wp.slide(slide_id)
    annotation = wp.annotation(annotation_file)
    rule = wp.rule(rule_file)
    annotation.make_masks(slide, rule,
                         foreground='otsu',
                         size=2000,
                        )

    annotation.export_thumb_masks(output_dir)

#% Extract large image and mask
#% patch slide and mask
def find_targetDims(TILE_SIZE, width, height, lower, upper):
    df0 = []
    df1 = []
    for i in np.arange(lower, upper):
        dif0 = np.abs(width - i*TILE_SIZE)
        df0.append([dif0, i*TILE_SIZE])

    for i in np.arange(lower, upper):
        dif1 = np.abs(height - i*TILE_SIZE)
        df1.append([dif1, i*TILE_SIZE])

    df0.sort()
    df1.sort()
    s0 = df0[0][1]
    s1 = df1[0][1]

    return s0, s1

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

def scale_img(single_patch_img):
        single_patch_img_scaled = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        return single_patch_img_scaled
    
def normalize_img(single_patch_img):
    return single_patch_img * (1/255.)


def plot_predicted_4c(slide_id, mask_data, dpi=1200, filename=None):    
    
    fig, ax = plt.subplots(dpi=dpi)     
    
    cmap1 = matplotlib.colors.ListedColormap([
                        'white', #0
                         '#00b300', #1 Benign, Green
                         'yellow', #2 GP3
                         'orange', #3 GP4
                         ])

    cax = ax.imshow(mask_data, cmap1,
              interpolation='nearest',
              vmin=0, 
              vmax=3
              )
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([
                            'Forground',
                            'Normal',
                            'GP3', 
                            'GP4',
                             ],
                            fontsize=4,
                            )  # vertically oriented colorbar
    fig.tight_layout()

    # fig.colorbar(plot)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('Prediction of slide: ' + slide_id, fontsize=8)
    _ = plt.show()
    
    if filename is not None:
        fig.savefig(filename)
        print("Saved the plot file")
        
def plot_predicted_Ca(slide_id, mask_data, 
                      dpi=1200, 
                      filename=None,
                      xlabel=None):    
    
    fig, ax = plt.subplots(dpi=dpi)     
    
    cmap1 = matplotlib.colors.ListedColormap([
                        # 'white', #0
                        # '#00b300', #1 Benign, Green
                         
                        # 'yellow', #2 GP3
                        #  'orange', #3 GP4
                         
                          # '#00b300', # green 
                          # '#ffa500', #4 orange
                          
                          # '#FFD700', # Golden Yellow
                          # '#4169E1', # Royal Blue
                          
                          '#FF9500', # Neon Orange
                          '#4169E1', # Royal Blue
                         ])

    cax = ax.imshow(mask_data, cmap1,
              interpolation='nearest',
              vmin=0, 
              vmax=1
              )
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, 1])
    cbar.ax.set_yticklabels([
                            # 'Forground',
                            # 'Normal',
                            'GP3', 
                            'GP4',
                             ],
                            fontsize=4,
                            )  # vertically oriented colorbar
    fig.tight_layout()

    # fig.colorbar(plot)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel(xlabel)
    plt.title('Prediction of slide: ' + slide_id, fontsize=8)   
    _ = plt.show()
    
    if filename is not None:
        fig.savefig(filename)
        print("Saved the plot file")
        
#%%
dataset_path = ""
if platform.system() == 'Darwin':
    # dataset_path = Path('/Volumes/MyPassport/PCa-Annotated/segmentation/Fold-4')
    test_dataset_path = Path('/Volumes/MyPassportB/test_slides/')   
    backup_model_path = Path('/Users/kasikritdamkliang/Library/CloudStorage/OneDrive-PrinceofSongklaUniversity/PCa/models/NT/5folds')
    # slide_to_predict_path = Path('/Users/kasikritdamkliang/Library/CloudStorage/OneDrive-PrinceofSongklaUniversity/PCa/WSI/')
    # slide_to_predict_path = Path('/Volumes/MyPassport/PCa-Kasikrit/svs/')
    slide_to_predict_path = Path('/Users/kasikritdamkliang/PCa/svs/')
    slide_to_predict_path = Path('/Users/kasikritdamkliang/Datasets/PKG - Biobank_CMB-PCA_v2/CMB-PCA/')
    predicted_output_path = 'infer'

elif platform.system() == 'Linux':    
    base_directory = '.'
    base_patch = '100slides'
    dataset_directory = os.path.join(base_directory, base_patch)
    model_best_path = './evaluate'
    backup_model_path = './5folds'
    save_path = model_best_path
    slide_no_pos = 2
    verbose=2
else:
    # dataset_path = Path(os.path.abspath('C:/Patches40png/segmentation/Fold-4'))
    test_dataset_path = Path(os.path.abspath('C:/50slides-data/40-segmentation/test_input'))
    backup_model_path = Path(os.path.abspath('G:/My Drive/PCa-Kasikrit/segmentation/')) 
    slide_to_predict_path = Path(os.path.abspath('C:/50slides-data/40slides-segmentation/slides'))


print("Build and config model...")
PREP_BACKBONE = 'seresnet50'
preprocess_input = sm.get_preprocessing(PREP_BACKBONE)
n_classes = 3
patch_size = 256
target_size = patch_size
ME = True
# ME = False

# preprocess_input1 = sm.get_preprocessing(BACKBONE1)
# model = sm.Unet(BACKBONE1, 
#                     input_shape=(target_size, target_size, 3),
#                    # input_shape=(target_size, target_size, 3) #PSPNet
#                    )
#%

#%%
print(f"{backup_model_path=}")
print('\nLoading models...')
list_models = []
list_models = list(glob.glob(os.path.join(backup_model_path, '*.hdf5'), 
                        recursive=True ))
list_models.sort()

for model in list_models:
    print(model, end='\n\n')

model_1 = tf.keras.models.load_model(list_models[4], compile=False)
model_2 = tf.keras.models.load_model(list_models[0], compile=False)
model_3 = tf.keras.models.load_model(list_models[1], compile=False)
model_4 = tf.keras.models.load_model(list_models[2], compile=False)
model_5 = tf.keras.models.load_model(list_models[3], compile=False)

print('Models loaded')
model_list = [model_1, model_2, model_3, model_4, model_5]
best_model = model_4


aaa
#%%

# #%%
# # model_name = model_name12
# # backup_model_best_1 = os.path.join(backup_model_path, model_name)

# # print('\nbackup_model_best_1: ', backup_model_best_1)
# # model_1 = load_model(backup_model_best_1, compile=False)
# # # model_1.compile(optim, total_loss, metrics=metrics)

# # print(model_1.summary())
# # print('\nLoaded: ', backup_model_best_1)

# # model_dir_name = model_name.split('.')[0]

# # predicted_output_path = os.path.join('.', model_dir_name)
# # if not os.path.exists(predicted_output_path):
# #     print("[INFO] 'creating {}' directory".format(predicted_output_path)) 
# #     os.makedirs(predicted_output_path)
# # else:
# #     print('\n', predicted_output_path, ' is existed\n')
    
#%%
# slide_list_test = [
             # '1342',
             # '1359',
              # '1683',
              # '1827',
             # '1830',
              # '1833',
              # '1840',
             # '1887',
             # '1890',
             # '1896',
             # '1921',
             # '1929',
             # '1939',
             # '1950',
             # '1954',
             # '1955',
             # '1968',
             # '1969',
             # '1971',
             # '1989',
       
            #just test
            # '1922',
            #'1329',
            
            # '1346',
            # '1929',
            
            # '1923',
            # '1935',
            # '1942',
            # '1347',
            # ]

# sorted(slide_list_test)
# print("\nSlides:\n", slide_list_test)
# print("slide len: ", len(slide_list_test))
# slide_list = slide_list_test


#%%
slide_list = [
#     'MSB-02472-03-01',
# 	'MSB-06184-04-02',
#     'MSB-02917-01-02',
# 	'MSB-06184-04-05',
# 	'MSB-08178-01-02',
#     'MSB-03973-01-02',
# 	'MSB-06184-07-02',
#     'MSB-05563-01-02',
    
# 	'MSB-07483-01-02',
 	'MSB-07436-03-02',
    ]

sorted(slide_list)
print("\nSlides:\n", slide_list)
print("slide len: ", len(slide_list))

#%%
for slide_id in tqdm(slide_list):
    start1 = datetime.now()
    per_slide_exe1 = datetime.now()
    print("\n\nPre-processing slide id : ", slide_id)
    
    slide_dir = os.path.join(slide_to_predict_path, slide_id)
    slide_predicted_path = os.path.join(predicted_output_path, slide_id)
    print(slide_dir)
    print(slide_predicted_path)
    
    if not os.path.exists(slide_predicted_path):
        print("[INFO] 'creating {}' directory".format(slide_predicted_path))
        os.makedirs(slide_predicted_path)
        
    slide_file = os.path.join(slide_dir + '.svs')
    slide = openslide.OpenSlide(slide_file)
    # annotation_file = slide_id + '-FG.xml'
    # rule_file = slide_id + '-FG.json'
    # output_path = os.path.join(slide_id)
    
    slide = openslide.OpenSlide(slide_file)
    print(f"File id: {slide}")
    print(f"Dimensions: {slide.dimensions}")
    #print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
    print(f"Number of levels in the image: {slide.level_count}")
    print(f"Downsample factor per level: {slide.level_downsamples}")
    print(f"Dimensions of levels: {slide.level_dimensions}")
    # print(f"Properties: {slide.properties}", sep=',')
    
    #best_level = slide.get_best_level_for_downsample(16)
    best_level = 1
    print('best_level: ', best_level)
    slide_data = slide.read_region((0,0), best_level, 
                                slide.level_dimensions[best_level])

    
    stop1 = datetime.now() - start1
    print("\tDone, read slide time: ", stop1)
    print("Slide size: ", slide_data.size)         
   
    print('\nFind suitable size of X, Y')
    
    SIZE_X = (slide_data.size[0]//target_size)*target_size #Nearest size divisible by our patch size
    SIZE_Y = (slide_data.size[1]//target_size)*target_size
    print(SIZE_X, SIZE_Y)
   
    #% compatibel with python 3.8
    print('\nCrop slide to the suitable size')
    slide_data_cr = slide_data.crop((0,0, SIZE_X, SIZE_Y))
    large_image = np.array(slide_data_cr)
    # plt.imshow(large_image)
    print(slide_data_cr.size)
    print(large_image.shape)
    # print('\nSave slide to PNG file')
    # Image.fromarray(large_image_crop).save('1332/1332-20x-based256px-cropped.png')
    
    
    #%%
    # print("Read 20X of a slide from drive")
    # slide_file = '1332/1332-20x-(19712, 35584).png'
    # slide_image = Image.open(slide_file)
    # large_image = np.array(slide_image)
    # plt.imshow(large_image)
    
    #%%
    print("\nExtracting patches of image and mask")
    
    patches_to_predict = patchify(
        large_image, 
        (target_size, target_size, 3), 
        step=target_size)  #Step=256 for 256 patches means no overlap
    print(patches_to_predict.shape)
    patches_to_predict = np.squeeze(patches_to_predict)
    print('patches_to_predict.shape: ', patches_to_predict.shape)
    print('patches_to_predict amount: ', patches_to_predict.shape[0]*patches_to_predict.shape[1])

    #%
    print('\nFilter patches')
    cnt=0
    # patch_dir = os.path.join(slide_id, 'patches_to_predict')
    # if not os.path.exists(patch_dir):
    #     print("[INFO] 'creating {}' directory".format(patch_dir))
    #     os.makedirs(patch_dir)
    
    patches_pass = list()
    for i in tqdm(range(patches_to_predict.shape[0])):
        for j in range(patches_to_predict.shape[1]):
            # print(i,j)
            single_patch = patches_to_predict[i,j]
            
            pixel_ratio = utility.detect_bg(single_patch, prob=False)
            if pixel_ratio > 20:
                # print(i, j, pixel_ratio)
                patches_pass.append((i,j))
                            
                # filename = str(i) + '_' + str(j) + '.png'
                # filepath = os.path.join(patch_dir, filename)
                # Image.fromarray(single_patch).save(filepath)
                
    print('\npatches passed pixel ratios: ', len(patches_pass) )   

    #%%
    # print("\nPredicting each patch")
    # exe0 = datetime.now()
    # segm_images = []
    # predicted_patches = []
    # for patch in patches_pass: 
    #     i, j = patch[0], patch[1]
    #     print(i, j)
    #     single_patch = patches[i, j]         
    #     single_patch = np.expand_dims(single_patch, 0)
    #     single_patch_input = preprocess_input1(single_patch)
    
    #     single_patch_prediction = model_1.predict(single_patch_input)
    #     single_patch_predicted_img = np.argmax(single_patch_prediction, axis=3)[0,:,:]
    
    #     predicted_patches.append(single_patch_predicted_img)
    
    # predicted_patches = np.array(predicted_patches)
    # print(predicted_patches.shape)
    # exe1 = datetime.now() - exe0
    # print("Prediction time: ", exe1)
    
    
    #%% 
    print('\nReconstruct predicted slide')
    
    reconstructed_image = np.zeros(shape=(large_image.shape[0], 
                                          large_image.shape[1]), 
                                    dtype='uint8')
                                    # .astype('uint8')
    
    reconstructed_image_patches = patchify(
        reconstructed_image, 
        (target_size, target_size), 
         step=target_size) 
    
    #%
    print('\nPredict passed patches')
    patch_preds_prob_list = []
    for patch in tqdm(patches_pass):
        i, j = patch[0], patch[1]
        # print(i, j)
        single_patch = patches_to_predict[i, j]         
        single_patch = np.expand_dims(single_patch, 0)
        single_patch_input = preprocess_input(single_patch)
        # single_patch_input = scale_img(single_patch)
        # single_patch_input = normalize_img(single_patch)
        if ME:
            single_patch_preds = [model.predict(single_patch_input, verbose=2)
                              for model in model_list]
            single_patch_preds_arr = np.array(single_patch_preds)
            single_patch_preds_sum = np.sum(single_patch_preds_arr, axis=0)          
            single_patch_preds_sum_avg = single_patch_preds_sum/len(model_list)          
            single_patch_preds_argmax = np.argmax(single_patch_preds_sum_avg, axis=3)[0]
            patch_preds_prob_list.append(single_patch_preds_sum_avg)
        else:            
            single_patch_preds = best_model.predict(single_patch_input) 
            single_patch_preds_argmax = np.argmax(single_patch_preds, axis=3)[0]
            patch_preds_prob_list.append(single_patch_preds_argmax)
        
        reconstructed_image_patches[i][j] = np.copy(single_patch_preds_argmax).astype('uint8')
    
    #%
    reconstructed_image1 = unpatchify(reconstructed_image_patches, 
                                      reconstructed_image.shape)
    # plt.imshow(reconstructed_image1)
    
    #%%
    # plt.imshow(np.squeeze(single_patch_input))
    # img = np.squeeze(single_patch)
    # img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # plt.imshow(img_1)
    # # Mean values from ImageNet for each channel
    # imagenet_means = np.array([103.939, 116.779, 123.68]) # Compute mean along height and width
    # # Subtract channel-wise mean from the image
    # zero_centered_img = img_1 - imagenet_means
    # plt.imshow(zero_centered_img)
    
    # img_2 = img_1/np.max(img_1)
    # plt.imshow(img_2)
    
    #%%
    # reconstructed_predicted_image_file = os.path.join(slide_id, slide_id + '-predicted-3c.png')
    plt.figure(dpi=300)
    plt.imshow(reconstructed_image1, cmap='gray')
    
    #%%
    plt.figure(dpi=300)
    plt.imshow(reconstructed_image1, cmap='RdYlBu_r')
    print(np.max(reconstructed_image1))
    
    # reconstructed_predicted_image_file = os.path.join(
    #                     slide_predicted_path,
    #                     slide_id + f'-predicted-{n_classes}.tiff')
    # Image.fromarray(reconstructed_image1).save(reconstructed_predicted_image_file)
    # print('Saved: ', reconstructed_predicted_image_file)

    #%% 
    print("\n Count values of 1 and values of 2 for GP3 and GP4")
    (unique, counts) = np.unique(reconstructed_image1, return_counts=True)
    print(unique, counts)
    #[0 1 2] [76981680  4304909   109123]
    # unique = [0, 1, 2]
    # counts = [76981680,  4304909,   109123]
    if len(unique)==1 and unique[0]==0:
        print("NO GP3+GP4 detected.")
        xlabel = "Benign=100%"
    if len(unique)==2 and unique[1]==1: #[0, 1]
        print("Only GP3 detected.")
        xlabel = "GP3=100%"
    elif len(unique)==2 and unique[1]==2: #[0, 2]
        print("Only GP4 detected.")
        xlabel = "GP4=100%"
    else: # [0, 1, 2]
        sumGP = counts[1]+counts[2]
        GP3_portion = counts[1]*100/sumGP
        GP4_portion = counts[2]*100/sumGP
        xlabel = f"GP3={GP3_portion:.2f}%, GP4={GP4_portion:.2f}%"
        print(xlabel)
        
    #%%
    print("\nPrepare predicted mask to plot only Ca");  
    #suppress background
    if n_classes >= 2:
        reconstructed_image1 = np.where(
            reconstructed_image1==0, np.nan,
            reconstructed_image1);    
        plt.imshow(reconstructed_image1)
    
    #%% save predicted slide as a single file
    # fig = plt.figure(dpi=3000)
    # plt.title('Predicted Slide Image: 1332')
    # cmap2 = matplotlib.colors.ListedColormap([
    #                                '#ffffff', # 0 white = BG + FG
    #                                '#00b300', # 1 Benign
    #                                '#ffff00', # 2 Malignant = P3+P4
    #                                ])
    # _ = plt.imshow(reconstructed_image1, cmap=cmap2,
    #               interpolation='nearest',
    #               vmin=0, 
    #               vmax=2
    #               )
    # plot_file = os.path.join(slide_id, slide_id + '-slide-level-1-predicted.png')
    # fig.savefig(plot_file)
    # plt.close()

        
    #%%
    from PIL import Image as pimg
    pimg.MAX_IMAGE_PIXELS = None
    
    print('\nPlot predicted mask over slide image')
   
    # mask_file = reconstructed_predicted_image_file
    # mask_read = pimg.open(mask_file) #mode L
    # print(unique_counts(mask_read))
    # mask_data = np.asarray(mask_read, dtype='uint8') # predicted masks   
    
    # mask_data = reconstructed_image1
    
    mask_data = reconstructed_image1
      
    print(mask_data.dtype, mask_data.shape)
    mask_data_L = pimg.fromarray(mask_data).convert('L')
    print(mask_data_L.size)
    
    alpha = 0.8
    dpi = 1200
    alpha_int = int(round(255*alpha))
    alpha_content = np.array(mask_data_L).astype('uint8') * alpha_int + (255 - alpha_int)                       
    
    alpha_content = pimg.fromarray(alpha_content)
    preview_palette = np.zeros(shape=768, dtype=int)
    
    
    preview_palette[0:9] = (np.array([ 
                        255, 255, 255, # 0 white
                        # 128, 128, 128, # dark gray
                        # 0, 179, 0, # 1 green
                        # 255, 255, 0, # 2 yellow                       
                        # 0, 179, 0, # 3 green
                        # 255, 165, 0,  #4 orange  
                        # 255,0,0 # red
                        
                        # 255, 215, 0, # Golden Yellow
                        # 65, 105, 225 # Royal Blue
                        
                        255, 149, 0, # Neon Orange
                        65, 105, 225 # Royal Blue
                    ])).astype('uint8')

    #%
    mask_data_L.putpalette(data=preview_palette.tolist())
    mask_rgb = mask_data_L.convert(mode='RGB')
    
    overlayed_image = pimg.composite(image1=mask_rgb, 
                                      image2=slide_data_cr, 
                                      mask=alpha_content)
    
    
    overlayed_image.thumbnail(size=slide_data.size, resample=0)
    
    plt.figure(dpi=300)
    plt.imshow(overlayed_image)
    
    #%%
    if ME:
        plot_outfile = os.path.join(slide_predicted_path,  \
            slide_id + '-slide-level-1-predicted-over-original-slide-alpha' + \
                str(alpha) + \
                '-dpi' + str(dpi) + \
                '-ME' + \
                '.png')  
    else:
        plot_outfile = os.path.join(slide_predicted_path,  \
                slide_id + '-slide-level-1-predicted-over-original-slide-alpha' + \
                    str(alpha) + \
                    '-dpi' + str(dpi) + \
                    '.png') 
            
    print('\nPlot and save: ', plot_outfile)
    
    plot_predicted_Ca(slide_id, 
              overlayed_image,
              dpi=dpi,
              filename=plot_outfile,
              xlabel=xlabel)
    
    per_slide_exe2 = datetime.now() - per_slide_exe1
    print(slide_id, " exe time: ", per_slide_exe2)
 
    
#%%
stop = datetime.now() - start

print("\nDone, Execution times: ", stop)

#%%




































