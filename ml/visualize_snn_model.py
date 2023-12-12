import os
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_spiking
import numpy as np
#from ml.split_training_validation_data import load_data
#from ml import STORAGE_DIR, VIZ_DIR, MODEL_DIR
import pickle



# remove when done
base_path = ".\\data" 
# Storage directory
STORAGE_DIR = ".\\storage\\train_test_data"
MODEL_DIR = ".\\storage\\trained_models"
VIZ_DIR =".\\storage\\visualizations" 
TRAINING_DIR = ".\\data\\training"


def load_data(file_name):
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list

# visualize surface defect of the first 25 train images

def visualize_defect(train_images):

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.axis("off")
        plt.savefig()

def check_output(seq_model, test_sequences, test_labels, test_images, modify_dt=None):
    """
    This code is only used for plotting purposes, and isn't necessary to
    understand the rest of this demo
    """

    # rebuild the model with the functional API, so that we can
    # access the output of intermediate layers
    inp = x = tf.keras.Input(batch_shape=seq_model.layers[0].input_shape)

    has_global_average_pooling = False
    for layer in seq_model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling1D):
            # remove the pooling so that we can see the model's
            # output over time
            has_global_average_pooling = True
            continue

        if isinstance(layer, (keras_spiking.SpikingActivation, keras_spiking.Lowpass)):
            cfg = layer.get_config()
            # update dt, if specified
            if modify_dt is not None:
                cfg["dt"] = modify_dt
            # always return the full time series so we can visualize it
            cfg["return_sequences"] = True

            layer = type(layer)(**cfg)
            print(layer)

        if isinstance(layer, keras_spiking.SpikingActivation):
            # save this layer so we can access it later
            spike_layer = layer

        x = layer(x)

    func_model = tf.keras.Model(inp, [x, spike_layer.output])

    # copy weights to new model
    func_model.set_weights(seq_model.get_weights())

    # run model
    output, spikes = func_model.predict(test_sequences)
    print(output.shape)

    if has_global_average_pooling:
        # check test accuracy using average output over all timesteps
        predictions = np.argmax(output.mean(axis=1), axis=-1)
    else:
        # check test accuracy using output from only the last timestep
        predictions = np.argmax(output[:, -1], axis=-1)
    accuracy = np.equal(predictions, test_labels).mean()
    print(f"Test accuracy: {100 * accuracy:.2f}%")

    time = test_sequences.shape[1] * spike_layer.dt
    n_spikes = spikes * spike_layer.dt
    rates = np.sum(n_spikes, axis=1) / time

    print(
        f"Spike rate per neuron (Hz): min={np.min(rates):.2f} "
        f"mean={np.mean(rates):.2f} max={np.max(rates):.2f}"
    )

    # plot output
    for ii in range(4):
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 3, 1)
        plt.title(f"image {ii}")
        plt.title("test")
        plt.imshow(test_images[ii])
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Spikes per neuron per timestep")
        bin_edges = np.arange(int(np.max(n_spikes[ii])) + 2) - 0.5
        plt.hist(np.ravel(n_spikes[ii]), bins=bin_edges)
        x_ticks = plt.xticks()[0]
        plt.xticks(
            x_ticks[(np.abs(x_ticks - np.round(x_ticks)) < 1e-8) & (x_ticks > -1e-8)]
        )
        plt.xlabel("# of spikes")
        plt.ylabel("Frequency")

        plt.subplot(1, 3, 3)
        plt.title("Output predictions")
        plt.plot(
            np.arange(test_sequences.shape[1]) * spike_layer.dt,
            tf.nn.softmax(output[ii].reshape(10,-1)),
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Probability")
        plt.ylim([-0.05, 1.05])

        plt.tight_layout()
        if modify_dt:
            plt.savefig(os.path.join(os.path.join(VIZ_DIR,"snn"), f"plot-{ii}-modify_dt"))
        else:
           plt.savefig(os.path.join(os.path.join(VIZ_DIR,"snn"),f"plot-{ii}")) 

if __name__ == "__main__":
    train_sequences = load_data(os.path.join(STORAGE_DIR, "train_sequences"))
    train_labels = load_data(os.path.join(STORAGE_DIR, "train_labels"))
    test_sequences = load_data( os.path.join(STORAGE_DIR,"test_sequences"))
    test_labels = load_data(os.path.join(STORAGE_DIR,"test_labels"))
    train_images = load_data(os.path.join(STORAGE_DIR,"train_images"))
    test_images = load_data(os.path.join(STORAGE_DIR,"test_images"))
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR,"spiking_model"))
    visualize_defect(train_images)
    check_output(seq_model=model, test_sequences=test_sequences, test_labels=test_labels, test_images=test_images)
    check_output(seq_model=model, test_sequences=test_sequences, test_labels=test_labels, test_images=test_images, modify_dt=0.1)

