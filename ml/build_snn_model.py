import os
import tensorflow as tf
#from ml import base_path, STORAGE_DIR, MODEL_DIR
import numpy as np
import keras_spiking
import pickle #remove when done
#from ml.split_training_validation_data import save_data

# repeat the images for n_steps
n_steps = 10
# set a batch size
BATCH_SIZE = 32
# seed
SEED = 123
# set image size
IMG_SIZE = (128,128)
# set validation split
VAL = 0.2



#remove when done debugging
base_path = ".\\data" 
# Storage directory
STORAGE_DIR = ".\\storage\\train_test_data"
MODEL_DIR = ".\\storage\\trained_models"
VIZ_DIR =".\\storage\\visualizations" 
TRAINING_DIR = ".\\data\\training"



def save_data(object_to_save, file_name):
    open_file = open(file_name, "wb")
    pickle.dump(object_to_save, open_file)
    open_file.close()







def prepare_data(base_dir_path):

    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)
    if not os.path.exists(VIZ_DIR):
        os.makedirs(VIZ_DIR)
    if not os.path.exists(TRAINING_DIR):
        os.makedirs(TRAINING_DIR)

    train_ds = tf.keras.utils.image_dataset_from_directory(base_dir_path, 
                                                            validation_split=VAL, 
                                                            subset="training", 
                                                            seed=SEED, 
                                                            image_size=IMG_SIZE, 
                                                            batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(base_dir_path, 
                                                            validation_split=VAL, 
                                                            subset="validation", 
                                                            seed=SEED, 
                                                            image_size=IMG_SIZE, 
                                                            batch_size=BATCH_SIZE)

    train_images = np.concatenate([x for x, _ in train_ds], axis=0)
    train_labels = np.concatenate([y for _, y in train_ds], axis=0)
    test_images = np.concatenate([x for x, _ in val_ds], axis=0)
    test_labels = np.concatenate([y for _, y in val_ds], axis=0)

    # normalize images so values are between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0


    train_sequences = np.tile(train_images[:, None], (1, n_steps, 1, 1, 1))
    test_sequences = np.tile(test_images[:, None], (1, n_steps, 1, 1, 1))

    return train_sequences, train_labels, test_sequences, test_labels, train_images, test_images


def train(model, train_x, test_x, train_y, test_y):

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(train_x, train_y, epochs=10)

    model.save(os.path.join(MODEL_DIR,"spiking_model"))

    _, test_acc = model.evaluate(test_x, test_y, verbose=2)

    return test_acc


model = tf.keras.Sequential(
    [
    # add temporal dimension to the input shape; we can set it to None,
    # to allow the model to flexibly run for different lengths of time
    tf.keras.layers.Reshape((-1, 128*128), input_shape=(None, 128, 128, 3)),
    # we can use Keras' TimeDistributed wrapper to allow the Dense layer
    # to operate on temporal data
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128)),
    # replace the "relu" activation in the non-spiking model with a
    # spiking equivalent
    keras_spiking.SpikingActivation("relu", spiking_aware_training=False),
    # use average pooling layer to average spiking output over time
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(10),
    ]
)


# train the model, identically to the non-spiking version,
# except using the time sequences as inputs
if __name__ == "__main__":
    print("start")
    train_sequences, train_labels, test_sequences, test_labels, train_images, test_images = prepare_data(base_dir_path=base_path)
    print("start2")
    train(model, train_sequences, test_sequences, train_labels, test_labels)
    print("start3")
    save_data(train_sequences, os.path.join(STORAGE_DIR, "train_sequences"))
    print("start4")
    save_data(train_labels, os.path.join(STORAGE_DIR, "train_labels"))
    print("start5")
    save_data(test_sequences, os.path.join(STORAGE_DIR,"test_sequences"))
    print("start6")
    save_data(test_labels, os.path.join(STORAGE_DIR,"test_labels"))
    print("start7")
    save_data(train_images, os.path.join(STORAGE_DIR,"train_images"))
    print("start8")
    save_data(test_images, os.path.join(STORAGE_DIR,"test_images"))
    print("end")