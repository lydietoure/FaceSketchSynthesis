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

def load_dataset(directory, target_size):
    """
    Loads a dataset from the given directory, and 
    """

    # Get all the file names
    filenames = sorted(os.listdir(directory))
    paths = [ os.path.join(directory, fn) for fn in filenames]

    data_array = []
    for image_path in paths:
        image = keras.utils.load_img(image_path, color_mode='rgb', target_size=target_size)
        input_arr = keras.utils.img_to_array(image)
        input_arr = input_arr.astype('float32') / 255.0
        
        data_array.append(np.asarray(input_arr))

    data_array = np.asarray(data_array)
    return data_array



def plot_side_by_side(
    original, reconstructions, target_size,  n=5, 
    labels=('Original','Reconstructed'),colors=('green','blue')
):
    """
    Plots the predictions below to their original.
    """
    plt.figure(figsize=(2*n, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.title(labels[0],color=colors[0])
        plt.imshow(original[i].reshape(*target_size,3))
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.title(labels[1], color=colors[1])
        plt.imshow(reconstructions[i].reshape(*target_size,3))
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plot_synthesis(original, predicted, target, target_size, n=10):
    """
    Plots the face-to-sketch/sketch-to-face.
    """
    plt.figure(figsize=(2*n, 6))
    # plt.title(title, fontsize=15)
    for i in range(n):
        # Display original
        ax = plt.subplot(3, n, i + 1)
        plt.title('Original',color='black')
        plt.imshow(original[i].reshape(*target_size,3))
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

