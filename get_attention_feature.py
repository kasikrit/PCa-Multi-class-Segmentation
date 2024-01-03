print("Val_total_loss batch_size 8\n")
# import mymodels

import utility
import models

import os, sys
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
    dataset_directory = Path('/Volumes/MyPassportB/slides/')
    # test_dataset_path = Path('/Volumes/MyPassportB/test_slides/')   
    backup_model_path = Path('/Users/kasikritdamkliang/Library/CloudStorage/OneDrive-PrinceofSongklaUniversity/PCa/models/NT/5folds')
    slide_to_predict_path = Path('/Users/kasikritdamkliang/PCa/svs/')
    # slide_to_predict_path = Path('/Users/kasikritdamkliang/Datasets/PKG - Biobank_CMB-PCA_v2/CMB-PCA/')
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
    model_best_path = Path(os.path.abspath(r'C:\Users\kasikrit\OneDrive - Prince of Songkla University\PCa\models'))
    dataset_directory = Path(os.path.abspath(r'E:\slides\svs'))
    slide_no_pos = 3
    file_name_pos = slide_no_pos + 2
    sys.path.append(r'C:\Users\kasikrit\Dropbox\PCa-Annotated\100slides')
    model_file = 'C:/Users/kasikrit/OneDrive - Prince of Songkla University/PCa/models/backup_model_fold_4-train-3c-exp-N-GP3-GP4-dilated-att-res-unet-120ep-20231124-2054.hdf5'
    best_model = tf.keras.models.load_model(model_file, 
                                compile=False)
    
#%%
print("Build and config model...")
PREP_BACKBONE = 'seresnet50'
preprocess_input = sm.get_preprocessing(PREP_BACKBONE)
n_classes = 3
patch_size = 256
target_size = patch_size
fold=2
batch_size = 16
patch_size = 256
ME = True
# ME = False

# preprocess_input1 = sm.get_preprocessing(BACKBONE1)
# model = sm.Unet(BACKBONE1, 
#                     input_shape=(target_size, target_size, 3),
#                    # input_shape=(target_size, target_size, 3) #PSPNet
#                    )
#%

#%%

# print('\nLoading models...')
# list_models = []
# list_models = list(glob.glob(os.path.join(backup_model_path, '*.hdf5'), 
#                         recursive=True ))
# list_models.sort()

# for model in list_models:
#     print(model, end='\n\n')

# model_1 = tf.keras.models.load_model(list_models[4], compile=False)
# model_2 = tf.keras.models.load_model(list_models[0], compile=False)
# model_3 = tf.keras.models.load_model(list_models[1], compile=False)
# model_4 = tf.keras.models.load_model(list_models[2], compile=False)
# model_5 = tf.keras.models.load_model(list_models[3], compile=False)

# print('Models loaded')
# model_list = [model_1, model_2, model_3, model_4, model_5]
# best_model = model_4

#%%
import pickle

print(f"\n\n{fold}")
val_pair_exp_file = f'val_pair_exp_fold_{fold}_N_GP3_GP4_relative_path.dat'
print(val_pair_exp_file)
with open (val_pair_exp_file, 'rb') as fp:
    val_pair_exp = pickle.load(fp)
print('\nLoaded val_pair_exp: ', len(val_pair_exp))

print('\nCreate val_generator_inf')
val_generator_inf = utility.DataGenerator(
    dataset_directory,
    val_pair_exp,
    num_classes=n_classes,
    batch_size=batch_size, 
    dim=(patch_size, patch_size, 3),
    # dim=(384, 384, 3), #PSPNet
    shuffle=True,
    inference=True,
    augmentation=utility.get_training_augmentation(),
    # preprocessing=utility.get_preprocessing(preprocess_input),
    rescale=None,
)
val_steps_inf = val_generator_inf.__len__()
print('val_steps_inf: ', val_steps_inf)

#%%
image_number = random.randint(0, val_steps_inf-1)
print('Random batch number: ', image_number)
X_val, y_val, pair_idx_val = val_generator_inf.__getitem__(image_number)
print(f"sanity check {X_val.shape}, {y_val.shape}")
y_val_argmax = np.argmax(y_val, axis=3).astype('uint8')

# utility.sanity_check(X_val, y_val_argmax,
#         note='Val exp ', 
#         batch_size=batch_size)


X_val_to_pred = preprocess_input(X_val)
print(X_val_to_pred.dtype, X_val_to_pred.shape)

#%%
# up_sampling2d_2, layer.output.shape=TensorShape([None, 32, 32, 1]), 
# up_sampling2d_4, layer.output.shape=TensorShape([None, 64, 64, 1]),
# up_sampling2d_6, layer.output.shape=TensorShape([None, 128, 128, 1]),
# up_sampling2d_8, layer.output.shape=TensorShape([None, 256, 256, 1]),

layer_list = ['up_sampling2d_2',
              'up_sampling2d_4',
              'up_sampling2d_6',
              'up_sampling2d_8',
              'activation_37'
              ]

#%
print(len(best_model.layers))
for index, layer in enumerate(best_model.layers):
    if layer.name in layer_list:
        print(f"{index}, {layer.name}, {layer.output.shape}\n\n")


#%%
activation_list = []
for selected_layer_name in layer_list:
    #selected_layer_name = layer_list[i]
    print(selected_layer_name)
    selected_layer = best_model.get_layer(selected_layer_name)
    
    # Create a model that will return the outputs of the selected_layer
    visualization_model = tf.keras.models.Model(
        inputs = best_model.input,
        outputs = [selected_layer.output, best_model.output])
    
    # Preprocess your image and expand its dimensions to match the input shape of the model
    # For example, if your image is loaded in a variable named 'img'
    # preprocessed_img = preprocess_input(img) # preprocess_input is your preprocessing function
    # expanded_img = np.expand_dims(preprocessed_img, axis=0)
    
    # Get the activations
    activations, _ = visualization_model.predict(
        X_val_to_pred, verbose=1)
    activation_list.append(activations)

#% Visualize the activations
# Each activation map is visualized separately
num_samples = 2 + len(activation_list) # Number of samples, 8 in your case
activations_0 = activation_list[0]
activations_1 = activation_list[1]
activations_2 = activation_list[2]
activations_3 = activation_list[3]
activations_4 = activation_list[4]
activations_pred = np.argmax(activations_4, axis=3)
# Iterate over each sample
for i in range(batch_size):    
    fig = plt.figure(figsize=(18, 8), dpi=600)
    # Plot the original image
    plt.subplot(1, num_samples, 1) # Position in a grid of 3 rows and 'num_samples' columns
    plt.imshow(X_val[i], cmap='gray')
    plt.axis('on') # Hide axis
    plt.title('Image')

    # Plot the corresponding mask
    plt.subplot(1, num_samples, 2 ) # Shift position by 'num_samples'
    (unique, counts) = np.unique(y_val_argmax[i], return_counts=True)
    xlabel = str(unique)
    # plt.xlabel(xlabel)
    plt.imshow(y_val_argmax[i], cmap='gray')
    plt.axis('on') 
    plt.title('Mask: ' + xlabel)

    # Plot the activation for this sample
    ax3 = plt.subplot(1, num_samples, 3) 
    activation_img = plt.imshow(activations_0[i, :, :, 0], 
                cmap='inferno',
                # cmap='gist_heat'
                # cmap='plasma',
                ) # Use index 0 to select the channel
    plt.axis('on') # Hide axis
    plt.title('Attention: ' + layer_list[0])
    
    # Plot the activation for this sample
    ax3 = plt.subplot(1, num_samples, 4) 
    activation_img = plt.imshow(activations_1[i, :, :, 0], 
                cmap='inferno',
                # cmap='gist_heat'
                # cmap='plasma',
                ) # Use index 0 to select the channel
    plt.axis('on') # Hide axis
    plt.title('Attention: ' + layer_list[1])
    
    # Plot the activation for this sample
    ax3 = plt.subplot(1, num_samples, 5) 
    activation_img = plt.imshow(activations_2[i, :, :, 0], 
                cmap='inferno',
                # cmap='gist_heat'
                # cmap='plasma',
                ) # Use index 0 to select the channel
    plt.axis('on') # Hide axis
    plt.title('Attention: ' + layer_list[2])
    
    # Plot the activation for this sample
    ax3 = plt.subplot(1, num_samples, 6) 
    activation_img = plt.imshow(activations_3[i, :, :, 0], 
                cmap='inferno',
                # cmap='gist_heat'
                # cmap='plasma',
                ) # Use index 0 to select the channel
    plt.axis('on') # Hide axis
    plt.title('Attention: ' + layer_list[3])
    
    # Plot the activation for this sample
    (unique, counts) = np.unique(activations_pred[i], return_counts=True)
    pred = str(unique)
    ax3 = plt.subplot(1, num_samples, 7) 
    activation_img = plt.imshow(
                activations_4[i, :, :, 0],     
                cmap='inferno',
                # cmap='gist_heat'
                # cmap='plasma',
                ) # Use index 0 to select the channel
    plt.axis('on') # Hide axis
    # plt.title('Attention: ' + layer_list[4])
    plt.title('Prediction: ' + pred)
   
    # plt.colorbar(activation_img, ax=ax3, orientation='vertical')
    
    fig.tight_layout()   
    plt.show()

      
#%%
stop = datetime.now() - start

print("\nDone, Execution times: ", stop)

#%%
features_max = patch_size * patch_size * 3
cnt = 0
for layer in best_model.layers:
    features = layer.output.shape[1]*layer.output.shape[2]*layer.output.shape[3]
    print(f"{layer.name}, {layer.output.shape}, {features}\n\n")
    if features > features_max:
        features_max = features
        layer_max = cnt
    cnt = cnt+1
    
print(f"{features_max}, {layer_max}")
print(f"{best_model.layers[layer_max].output.shape}")

#%%
import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects, ax,  fontsize=10):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 3)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 12),  #vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=fontsize)


#%%
# Provided data values
# metrics = ['Jaccard_coef', 'Dice_coef', 'precision', 'recall', 'accuracy']

# standard_values = [0.566, 0.718, 0.725, 0.716, 0.828]
# me_values = [0.712, 0.831, 0.836, 0.828, 0.893]

# standard_errors = [0.036,
#                     0.029,
#                     0.027,
#                     0.034,
#                     0.018,]  
# me_errors = [0.013,
#             0.009,
#             0.016,
#             0.006,
#             0.007,]  

# x = np.arange(len(metrics))  # the label locations
# width = 0.35  # the width of the bars

# # Define the exact colors for the bars
# # Matplotlib does not have a 'deep sea blue' color, so we use a custom color close to it
# colors = ['green', '#00688B']  # Deep sea blue hex color

# fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
# ax.set_ylim(0.0, 1.0)

# rects1 = ax.bar(x - width/2, standard_values, width, 
#                 yerr=standard_errors, label='Standard',
#                 capsize=5, color=colors[0])
# rects2 = ax.bar(x + width/2, me_values, width, 
#                 yerr=me_errors, label='ME', 
#                 capsize=5, color=colors[1])

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores', fontsize=12)
# ax.set_title('Performance Metrics Comparison', fontsize=12)
# ax.set_xticks(x)
# ax.set_xticklabels(metrics, fontsize=12)
# ax.legend(loc='upper left')

# autolabel(rects1, ax, fontsize=12)
# autolabel(rects2, ax, fontsize=12)

# fig.tight_layout()

# plt.show()

# fig.savefig("val-performance-comparision-stadard-me.png")

#%% 
# standard_values = [0.712,
#                 0.831,
#                 0.836,
#                 0.828,
#                 0.893]
# me_values = [0.687,
#             0.814,
#             0.826,
#             0.805,
#             0.909]

# standard_errors = [0.013,
#             0.009,
#             0.016,
#             0.006,
#             0.007]  

# me_errors = [0.017,
#             0.012,
#             0.031,
#             0.018,
#             0.007]  

# x = np.arange(len(metrics))  # the label locations
# width = 0.35  # the width of the bars

# # Redefine the color palette to match user's request
# colors = ['#00688B', '#4169E1']  # Deep sea blue and Royal blue

# fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
# ax.set_ylim(0.0, 1.0)

# # Create the bars with the updated colors
# rects1 = ax.bar(x - width/2, standard_values, width, yerr=standard_errors,
#                 label='Average all classes', capsize=5, color=colors[0])
# rects2 = ax.bar(x + width/2, me_values, width, yerr=me_errors,
#                 label='Average positive classes', capsize=5, color=colors[1])

# # Set labels and title
# ax.set_ylabel('Scores', fontsize=12)
# ax.set_title('Performance Metrics Comparison',
#              fontsize=12)
# ax.set_xticks(x)
# ax.set_xticklabels(metrics, fontsize=12)
# ax.legend()

# # Call the autolabel function to annotate the bars
# autolabel(rects1, ax,  fontsize=12)
# autolabel(rects2, ax,  fontsize=12)

# # Adjust the layout and display the plot
# fig.tight_layout()
# plt.show()
# fig.savefig("val-all-vs-positive-classes.png")

#%% get samples of validation set
batch_list = []
for i in range(15): #10*batch size
    image_number = random.randint(0, val_steps_inf-1)
    print('Random batch number: ', image_number)
    X_val, y_val, pair_idx_val = val_generator_inf.__getitem__(image_number)
    print(f"sanity check {X_val.shape}, {y_val.shape}")
    y_val_argmax = np.argmax(y_val, axis=3).astype('uint8')
    batch_list.append(X_val)
    
X_data = np.concatenate(batch_list, axis=0)
print(f"{X_data.shape}")

#%% t-SNE analysis
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf


# Extract the output from the specified layer
layer_output = best_model.layers[layer_max].output

# Create a model that will return these outputs, given the model input
intermediate_model = tf.keras.models.Model(
    inputs=best_model.input, outputs=layer_output)

# Assuming X_val is your validation data
intermediate_output = intermediate_model.predict(
    # X_val,
    X_data,
    verbose=1)

print("Reshape the output to be 2D")
reshaped_output = intermediate_output.reshape(-1, 
                      intermediate_output.shape[-1])

reshaped_output = intermediate_output.reshape(
    intermediate_output.shape[0], 
    intermediate_output.shape[1]* \
    intermediate_output.shape[2]* \
    intermediate_output.shape[3]
    ) 

print(f"{reshaped_output.shape}")

#%% Apply t-SNE
# tsne = TSNE(n_components=2, random_state=1337,
#             verbose=1)
# tsne_results = tsne.fit_transform(reshaped_output)

# # Visualize with scatter plot
# plt.figure(figsize=(12, 8))
# plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
# plt.title("t-SNE visualization of layer features")
# plt.xlabel("t-SNE feature 1")
# plt.ylabel("t-SNE feature 2")
# plt.show()


#%% OPEN TSNE
from openTSNE import TSNE
import numpy as np

from openTSNE.callbacks import Callback

class PrintIteration(Callback):
    def __init__(self):
        super().__init__()

    def __call__(self, iteration, error, embedding):
        print(f"Iteration {iteration}, error: {error}")
# Assuming `reshaped_output` is your data reshaped into a 2D array
# For instance, reshaped_output.shape could be (number_of_samples, features)

# Initialize TSNE with desired parameters
tsne = TSNE(
    n_components=3,
    perplexity=30,  # Adjust based on your dataset size
    # learning_rate='auto',
    learning_rate=1e-4,
    # learning_rate=100,
    n_jobs=-1,  # Use all available CPU cores
    random_state=42,
    verbose=True
)

# Run TSNE
tsne_results = tsne.fit(reshaped_output)
print(tsne_results.shape)

#%
# `embedding` is a numpy array with the shape (number_of_samples, 2)
# # Visualize with scatter plot
# plt.figure(figsize=(12, 8), dpi=300)
# plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
# plt.title("t-SNE visualization of layer features")
# plt.xlabel("t-SNE feature 1")
# plt.ylabel("t-SNE feature 2")
# plt.show()

#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

labels = np.random.randint(0, 3, 
                   tsne_results.shape[1])

# Define a color for each label
color_map = {0: 'green', 1: 'orange', 2: 'blue'}
colors = [color_map[label] for label in labels]

# Create a new matplotlib figure and axes
fig = plt.figure(figsize=(10, 7), dpi=300)
ax = fig.add_subplot(111, projection='3d')

# Scatter plot using the first three dimensions of the t-SNE output, colored by category
scatter = ax.scatter(
    tsne_results[:, 0],
    tsne_results[:, 1],
    tsne_results[:, 2],
    #cmap='viridis',
    c=colors,
    s=20, alpha=0.5
)

# Set labels for axes
ax.set_xlabel('t-SNE feature 0')
ax.set_ylabel('t-SNE feature 1')
ax.set_zlabel('t-SNE feature 2')

# Set title
ax.set_title('3D t-SNE Visualization')

# Show the plot
plt.show()

#%%

