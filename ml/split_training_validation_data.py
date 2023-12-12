import numpy as np
import tensorflow as tf
import os, shutil
from os import listdir
from os.path import isfile, join
import sklearn
import sklearn.model_selection
import pickle

from ml import base_path

tf.random.set_seed(0)
np.random.seed(0)

def split_data():

    class_names = [join(base_path,f) for f in listdir(base_path) if isfile(join(base_path, f))]
    np.random.shuffle(class_names)
    trainset, testset = sklearn.model_selection.train_test_split(class_names, train_size=len(class_names) - int(len(class_names)/4), test_size=int(len(class_names)/4))
    return trainset, testset

def move_files(trainset, testset):

    directories = dict(training=trainset, validation=testset)
    for dir, dataset in zip(directories.keys(), directories.values()):
        os.mkdir(os.path.join(base_path, dir))
        for f in dataset:
            shutil.move(f, os.path.join(base_path, dir))

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

def save_data(object_to_save, file_name):
    open_file = open(file_name, "wb")
    pickle.dump(object_to_save, open_file)
    open_file.close()

def load_data(file_name):
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list


if __name__ == "__main__":
    trainset, testset = split_data()
    move_files(trainset=trainset, testset=testset)