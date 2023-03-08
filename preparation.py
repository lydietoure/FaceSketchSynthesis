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

def normalise_images(images, target_range='unit'):
    """
    Normalise the images to be in a given range. 
    
    Params:
        - images
        - target_range: One of `unit` (default) and `tanh`
    """
    if target_range == 'unit':
        output = images / 255.0
    else:
        output = (output / 127.5) - 1
    return output

def rescale_images(images, source_range, target_range):
    """
    Rescale images from one range to the other
    """

    if isinstance(source_range, str):
        if source_range == 'tanh':
            source_range = (-1.0,1.0)
        else:
            source_range=(0.0,1.)
    if isinstance(target_range, str):
        if target_range == 'tanh':
            target_range = (-1.0,1.0)
        else:
            target_range=(0.0,1.0)

    low1, hi1 = source_range
    low2, hi2 = target_range

    outputs = (images - low1) / (hi1-low1) * (hi2-low2) + low2
    return outputs

def load_dataset(directory, target_size, color_mode='rgb', target_range='sigmoid'):
    """
    Loads a dataset from the given directory, and resizes it to the given target size

    Params:
    - directory: the directory where the images are
    - target_size: the size of the images to be loaded in
    - target_range: the range of values of the image, one of `sigmoid` or `tanh`, or a tuple of floats (low,hi). If `tanh`, the values will be normalised to be in the (-1,1) range
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
        input_arr = rescale_images(input_arr, (0,255.0), target_range)
        # if target_range == 'tanh':
        #     input_arr = (input_arr / 255.0 - 0.5) * 2
        #     input_arr = np.clip(input_arr, -1, 1)
        # else:
        #     input_arr = input_arr / 255.0

        
        data_array.append(np.asarray(input_arr))

    data_array = np.asarray(data_array)
    return data_array

def rescale(ls):
    return [ (x * 0.5) + 0.5 for x in ls]


def plot_side_by_side(
    original, reconstructions, target_size,  n=5, nb_channels=3,
    labels=('Original','Reconstructed'),colors=('green','blue'),
    rescale_direction=None,
):
    """
    Plots the predictions below to their original.

    Params:
        - original
        - reconstructions
        - target_size
        - nb_channels
        - labels
        - colors
        - rescale_direction: A tuple ((low1,hi1),(low2,hi2)) of ranges. Whether to rescale the data from
    """
    if rescale_direction:
        source_range, target_range = rescale_direction
        original = rescale_images(original, source_range, target_range)
        reconstructions = rescale_images(reconstructions, source_range, target_range)

    # original = original.reshape(*target_size, nb_channels)
    # reconstructions = reconstructions.reshape(*target_size, nb_channels)

    plt.figure(figsize=(2*n, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.title(labels[0],color=colors[0])
        plt.imshow(original[i].reshape(*target_size, nb_channels))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.title(labels[1], color=colors[1])
        plt.imshow(reconstructions[i].reshape(*target_size, nb_channels))
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plot_synthesis(original, predicted, target, target_size, nb_channels=3, n=5,
                   labels=('Original','Predicted','Ground Truth'),
                   rescale_direction=None):
    """
    Plots the face-to-sketch/sketch-to-face.
    """

    if rescale_direction:
        source_range, target_range = rescale_direction
        original = rescale_images(original, source_range, target_range)
        predicted = rescale_images(predicted, source_range, target_range)
        target = rescale_images(target, source_range, target_range)


    plt.figure(figsize=(2*n, 6))
    # plt.title(title, fontsize=15)
    for i in range(n):
        # Display original
        ax = plt.subplot(3, n, i + 1)
        plt.title(labels[0],color='black')
        plt.imshow(original[i].reshape(*target_size, nb_channels))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display predictions
        ax = plt.subplot(3, n, i + 1 + n)
        plt.title(labels[1],color='blue')
        plt.imshow(predicted[i].reshape(*target_size, nb_channels))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display target 
        ax = plt.subplot(3, n, i+1+2*n)
        plt.title(labels[2],color='green')
        plt.imshow(target[i].reshape(*target_size, nb_channels))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def plot_dict_as_table(results):
    """
    Plots a dictionary in a table
    """
    data = list(zip(results.keys(), results.values()))

    _, ax = plt.subplots()
    table = ax.table(cellText=data, loc='left', cellLoc='left')
    
    table.set_fontsize(12)
    table.scale(1,4)
    ax.axis('off')

    plt.show()