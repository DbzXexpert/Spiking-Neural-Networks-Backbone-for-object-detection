import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
#from ml import TRAINING_DIR, MODEL_DIR, base_path as dir
#from ml.split_training_validation_data import get_files

img_dim = 224
image_size = (img_dim, img_dim, 3)
epochs = 20
batch_size = 16

#remove after done 
dir = ".\\data" 
# Storage directory
STORAGE_DIR = ".\\storage\\train_test_data"
MODEL_DIR = ".\\storage\\trained_models"
VIZ_DIR =".\\storage\\visualizations" 
TRAINING_DIR = ".\\data\\training"



def get_files(dirName): 
    list_files_and_subdirectories = os.listdir(dirName)
    files = list()
    for elt in list_files_and_subdirectories:
        pathName = os.path.join(dirName, elt)
        # Get all the files contained in elt if elt is a subdirectoty
        if os.path.isdir(pathName):
            files = files + get_files(pathName)
        # Get only the files
        elif pathName.endswith(".jpg"):
               files.append(pathName)
    #return all files
    return files



# Define parameters
#the depth of the convolution filter matches the depth of the image
batch_size = 32
img_height = 1500
img_width = 1509

train_ds = tf.keras.utils.image_dataset_from_directory(
  dir,
  validation_split= 0.1,
  subset="training",
  seed=123,
  image_size=(img_height, img_width), 
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  dir,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

model= tf.keras.applications.resnet.ResNet101(
    include_top=False, 
    weights='imagenet', 
    input_shape=None,         #(img_height, img_width,3),
    pooling=max)

model.save(os.path.join(MODEL_DIR,"resnet_model"))

files = get_files(TRAINING_DIR)

X_features = list()

def get_features(files : list):

    for f in files:
        img_data = Image.open(f).convert('RGB')
        image_as_array = np.array(img_data, np.uint8)
        x = np.expand_dims(image_as_array, axis=0)
        x = tf.keras.applications.imagenet_utils.preprocess_input(x)
        features = model.predict(x)
        features_reduce = features.squeeze()
        X_features.append(features_reduce)
    return X_features

if __name__ == "__main__":
    features = get_features(files[:3])
