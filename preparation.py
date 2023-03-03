"""
Data preparation and visualisation functions
"""

from tensorflow import keras
import os
import numpy as np
# from tqdm import tqdm

import matplotlib.pyplot as plt

# Constants
# """The size of the image once it is processed"""
# TARGET_SIZE = (128,128)
# """The standard dimension of the input, that is the target size with the channels"""
# IMG_DIM = (*TARGET_SIZE,3)

def load_dataset(directory, target_size, target_range = 'sigmoid', color_mode='rgb'):
    """
    Loads a dataset from the given directory, and preprocesses it

    Params:
    - directory: the directory where the images are
    - target_size: the size of the images to be loaded in
    - target_range: the range of values of the image, one of `sigmoid` or `tanh`. If `tanh`, the values will be normalised to be in the (-1,1) range
        otherwise, they will be re-scaled to the (0,1) interval. Defaults to `sigmoid`
    - color_mode: the color mode (rbg or grayscale)
    """

    # Get all the file names
    filenames = sorted(os.listdir(directory))
    paths = [ os.path.join(directory, fn) for fn in filenames]

    data_array = []
    for image_path in paths:
        image = keras.utils.load_img(image_path, color_mode=color_mode, target_size=target_size)
        input_arr = keras.utils.img_to_array(image)
        input_arr = input_arr.astype('float32') 

        # Normalise to the appropriate range
        if target_range == 'tanh':
            input_arr = (input_arr / 255.0 - 0.5) * 2
            input_arr = np.clip(input_arr, -1, 1)
        else:
            input_arr = input_arr / 255.0

        
        data_array.append(np.asarray(input_arr))

    data_array = np.asarray(data_array)
    return data_array

def rescale(ls):
    return [ (x * 0.5) + 0.5 for x in ls]


def plot_side_by_side(
    original, reconstructions, target_size,  n=5, nb_channels=3,
    labels=('Original','Reconstructed'),colors=('green','blue'),
    resize_from_tanh=False,
):
    """
    Plots the predictions below to their original.
    """
    if resize_from_tanh:
        [original, reconstructions] = rescale([original, reconstructions])

    # width = 2*n if 2*n*target_size[0] < 20*target_size[0] else 2*

    plt.figure(figsize=(2*n, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.title(labels[0],color=colors[0])
        plt.imshow(original[i].reshape(*target_size,nb_channels))
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.title(labels[1], color=colors[1])
        plt.imshow(reconstructions[i].reshape(*target_size,nb_channels))
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plot_synthesis(original, predicted, target, target_size, nb_channels=3, n=10):
    """
    Plots the face-to-sketch/sketch-to-face.
    """
    plt.figure(figsize=(2*n, 6))
    # plt.title(title, fontsize=15)
    for i in range(n):
        # Display original
        ax = plt.subplot(3, n, i + 1)
        plt.title('Original',color='black')
        plt.imshow(original[i].reshape(*target_size,nb_channels))
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display predictions
        ax = plt.subplot(3, n, i + 1 + n)
        plt.title('Predicted',color='blue')
        plt.imshow(predicted[i].reshape(*target_size,3))
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display target 
        ax = plt.subplot(3, n, i+1+2*n)
        plt.title('Target',color='green')
        plt.imshow(target[i].reshape(*target_size,3))
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

