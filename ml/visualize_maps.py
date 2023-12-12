from pyexpat import model
import os
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
#from ml import MODEL_DIR, VIZ_DIR remove

image_path= ".\\data\\training\\1000_B_165-81698_PkDown_Push_RC_17042018_1.jpg"


# remove when done
base_path = ".\\data" 
# Storage directory
STORAGE_DIR = ".\\storage\\train_test_data"
MODEL_DIR = ".\\storage\\trained_models"
VIZ_DIR =".\\storage\\visualizations" 
TRAINING_DIR = ".\\data\\training"


def visualize_feature_maps(model):
    print('start')
    mymodel = tf.keras.models.load_model(os.path.join(MODEL_DIR, model))

    layer_names = [layer.name for layer in mymodel.layers]
    layer_outputs = [layer.output for layer in mymodel.layers]
    feature_map_model = tf.keras.models.Model(inputs=mymodel.inputs, outputs=layer_outputs)

    img = image.load_img(image_path, target_size=(128, 128))  
    input = image.img_to_array(img)                           
    input = input.reshape((1,1,) + input.shape)                   
    input /= 255.0

    feature_maps = feature_map_model.predict(input)

    if model == "spiking_model":
        layers_lenght = 3
    else:
       layers_lenght = 4 

    for layer_name, feature_map in zip(layer_names, feature_maps):  
        
        print(f"The shape of the {layer_name} is =======>> {feature_map.shape}")
        if len(feature_map.shape) == layers_lenght:
            k = feature_map.shape[-1]  
            size=feature_map.shape[1]
            image_belt = np.zeros((size, size*k))
            for i in range(k):
                feature_image = feature_map[0, :, i]
                feature_image-= feature_image.mean()
                feature_image/= feature_image.std ()
                feature_image*=  64
                feature_image+= 128
                feature_image= np.clip(feature_image, 0, 255).astype('uint8')
                image_belt[:, i * size : (i + 1) * size] = feature_image    
            scale = 20. / k
            if model == "spiking_model":
                ysize = scale * 20
            else:
                ysize = scale  
            plt.figure( figsize=(scale * k, ysize) )
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(image_belt, aspect='auto')
            plt.savefig(os.path.join(os.path.join(VIZ_DIR, model), f"{model}-feature_maps-{layer_name}.png"))
            print('finish')



if __name__ == "__main__":
    #visualize_feature_maps("spiking_model")
    visualize_feature_maps("resNet_model")
