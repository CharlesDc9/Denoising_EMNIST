# EMNIST Denoising with Autoencoders and Variational Autoencoders

This project implements various Autoencoder (AE) and Variational Autoencoder (VAE) architectures for denoising EMNIST dataset and detecting anomalies and study of the mode collapse phenomenon in GANs. It is a project for the Deep Learning course at X-HEC Data Science master. The contributors of the project are Charles De Cian, Matthieu Delsart and Tim Valencony. The results can be found in the notebook `DelsartValenconyDecian.ipynb`.

## Project Overview

The project explores several main VAE architectures:
1. A simple AE, 
2. A fully connected (FC) VAE, 
3. A Convolutional Neural Network (CNN) VAE,
4. A GAN with a VAE discriminator,
5. A GAN with a CNN discriminator.

Models are trained to:
- Denoise EMNIST dataset
- Learn a meaningful latent representation
- Detect anomalies based on reconstruction loss

### Prerequisites
- Python 3.11 or higher
- pip (Python package installer)

### Installation and Setup
1. Clone this repository
2. Create a virtual environment (we used UV package manager)
3. Install required packages using our pyproject.toml file

