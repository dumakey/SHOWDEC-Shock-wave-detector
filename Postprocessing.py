import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from Preprocessing import ImageTransformer
import tensorflow as tf

def monitor_hidden_layers(img, model, case_dir, figs_per_row=5, rows_to_cols_ratio=1, idx=None):

    if idx:
        storage_dir = os.path.join(case_dir,'Results','pretrained_model','Hidden_activations','Sample_' + str(idx))
    else:
        storage_dir = os.path.join(case_dir,'Results','pretrained_model','Hidden_activations')

    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)

    # Input preprocessing
    img = img/127.5
    img = img - 1
    img = np.expand_dims(img,axis=0)
    img_tensor = tf.convert_to_tensor(img)

    # Activation model setup
    idx_0 = 1
    idx_f = [i for i,layer in enumerate(model.layers) if layer.name == 'flatten'][0]
    layer_outputs = [layer.output for layer in model.layers[idx_0:idx_f]]
    layer_names = [layer.name for layer in model.layers[idx_0:idx_f]]
    activation_model = tf.keras.Model(inputs=model.input,outputs=layer_outputs)
    activations = activation_model.predict(img_tensor,steps=1)

    # Plotting
    layer_idx = 1
    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        channel_idx = 0
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        height = layer_activation.shape[1]
        width = layer_activation.shape[2]
        n_rows = int(np.ceil(n_features/figs_per_row))
        display_grid = np.zeros((height*n_rows,figs_per_row*width))
        for row in range(n_rows):
            for col in range(figs_per_row):
                channel_image = layer_activation[0,:,:,channel_idx]
                channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image,0,255).astype('uint8')
                display_grid[row*height:(row + 1)*height,col*width:(col + 1)*width] = channel_image
                channel_idx += 1

                if (channel_idx + 1) > n_features:
                    break

        if n_rows < int(rows_to_cols_ratio*figs_per_row):
            plt.figure(figsize=(25,10))
            plt.suptitle('Layer: {}'.format(layer_name),fontsize=18)
            plt.axis('off')
            plt.imshow(display_grid,aspect='auto',cmap='viridis')  # cmap: plasma / viridis
            plt.savefig(os.path.join(storage_dir,'Layer_{:d}_{}_activations.png'.format(layer_idx,layer_name)),dpi=150)
            plt.close()
        else:
            n_rows_old = n_rows
            n_rows = rows_to_cols_ratio*figs_per_row
            for j in range(int(np.ceil(n_rows_old/n_rows))):
                plt.figure(figsize=(25,10))
                plt.suptitle('Layer: {}'.format(layer_name),fontsize=18)
                plt.axis('off')
                plt.imshow(display_grid[j*height*n_rows:(j+1)*height*n_rows,:],aspect='auto',cmap='viridis')   # cmap: plasma / viridis
                plt.savefig(os.path.join(storage_dir,'Layer_{:d}_{}_activations_{}.png'.
                                         format(layer_idx,layer_name,(j+1))), dpi=150)
                plt.close()

        layer_idx += 1


