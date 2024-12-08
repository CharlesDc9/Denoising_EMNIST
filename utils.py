import numpy as np
import zipfile
import pandas as pd
from matplotlib import pyplot as plt
import torch

def load_images_from_zip(zip_filename='data/corrupted_emnist.zip'):
    """
    Load a numpy array from a .zip archive containing a .npz file.

    Args:
        zip_filename (str): The name of the .zip file to read.

    Returns:
        numpy.ndarray: The loaded array.
    """
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        npz_filename = zipf.namelist()[0]  # Get the name of the .npz file
        zipf.extract(npz_filename, path='data')  # Extract the .npz file
    data = np.load("data/" + npz_filename)['all_imagesfinal']  # Load the array
    return data / 255.0  # Normalize the pixel values

def load_true_emnist(filename='data/emnist-balanced-train.csv'):
    """
    Load the true Train EMNIST dataset.

    Args:
        filename (str): The name of the .csv file to read.

    Returns:
        numpy.ndarray: The images.
        numpy.ndarray: The labels.
    """
    
    data = pd.read_csv(filename)
    images = np.array(data.iloc[:, 1:])
    images = images.reshape(-1, 28, 28) / 255
    
    labels = np.array(data.iloc[:, 0])
    return images, labels

def to_img(x, base_dim=32):
    "Displays array as an image."
    if isinstance(x, torch.Tensor):
        x = x.cpu().data.numpy()
    x = np.clip(x, 0, 1)
    x = x.reshape([-1, base_dim, base_dim])
    return x


def plot_images(data, base_dim=32, labels=None):
    """
    Plots the images to visualize many of them at once.
    """
    # Flatten images to vectors
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()

    true_imgs = data
    true_imgs = to_img(true_imgs, base_dim=base_dim)

    n = 10  # Number of images to display at the same time
    for j in range(0, len(data), 10):
        plt.figure(figsize=(20, 4))
        for i in range(n):
            if i+j >= len(data):
                break
            # Display original images
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(true_imgs[i + j])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            if labels is not None:
                ax.set_title(str(labels[i + j]), fontsize=10, color='black', pad=5)


            

def plot_reconstructions(model, data, base_dim=32):
    """
    Plot reconstructions from a dataset set.
    The top row is the original digits,
    the middle row is the encoded vector,
    and the bottom row is the decoder reconstruction.
    """
    
    true_imgs = data
    # Encode and then decode the images
    encoded_imgs = model.encoder(data)
    decoded_imgs = model.decoder(encoded_imgs)

    # Convert images for plotting
    true_imgs = to_img(true_imgs, base_dim=base_dim)
    decoded_imgs = to_img(decoded_imgs, base_dim=base_dim)
    encoded_imgs = encoded_imgs.cpu().data.numpy()

    n = 10  # Number of images to display at the same time
    for j in range(0, len(data), 10):
        plt.figure(figsize=(20, 4))
        for i in range(n):
            if i+j >= len(data):
                break
            # Display original images
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(true_imgs[i + j])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display encoded representations
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(encoded_imgs[i + j].reshape(-1, 4))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstructed images
            ax = plt.subplot(3, n, i + 1 + 2 * n)
            plt.imshow(decoded_imgs[i + j])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)