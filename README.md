# Image-Pixelation-Detection-And-Correction

This repository offers a comprehensive solution for detecting and removing image pixelation using deep learning models. It leverages MobileNetV2 for pixelation detection and ESPCN for high-quality depixelation.

# Libraries Required
- TensorFlow 
- OpenCV
- Keras
- Numpy
- Matplotlib
- Scikit-learn
- Seaborn

# Dataset Used

The dataset used in this project is derived from the DIV2K collection available through TensorFlow Datasets (TFDS). It comprises high-resolution images with dimensions of 1200x1200 pixels, which serve as the ground truth for model training. To create the corresponding low-resolution images, each high-resolution image is downscaled by a factor of 6, resulting in images with a resolution of 200x200 pixels. This degradation process simulates real-world pixelation and loss of detail, enabling the model to learn effective image enhancement techniques. The dataset thus includes pairs of high-resolution and degraded low-resolution images for supervised training and evaluation of the image super-resolution and pixelation removal models.

# MobileNetV2 for Pixelation Classification

The classification model utilizes MobileNetV2, a lightweight and efficient convolutional neural network architecture designed for mobile and edge devices. MobileNetV2 is known for its use of depthwise separable convolutions, which significantly reduce computational complexity while maintaining high accuracy. In this implementation, MobileNetV2 serves as the base model for feature extraction, with its pre-trained weights on ImageNet providing a strong foundation for detecting pixelation in images.

# ESPCN Model for Pixelation Removal

The ESPCN (Efficient Sub-Pixel Convolutional Neural Network) model is designed for enhancing low-resolution images by removing pixelation. Below is an overview of its architecture and key components:

# Architecture

![architecture](https://github.com/user-attachments/assets/fa3bd6ed-a7ae-4ec7-ac4a-71e7df13a97f)

1. Convolutional Layers for Feature Extraction:

    - Layer 1: Convolutional layer with 8 filters and a 5x5 kernel size. This layer is responsible for initial feature extraction from the input image.
    - Layer 2: Convolutional layer with 8 filters and a 3x3 kernel size. It continues the feature extraction process by refining the features obtained from the previous layer.
    - Layer 3: Convolutional layer with 8 filters and a 2x2 kernel size. This layer further refines the features, preparing them for the subsequent stages of the model.

2. Residual Connections:

    The model incorporates residual connections between the convolutional layers. These connections help mitigate the challenges associated with dimension reduction and preserve essential features from the pixelated input image patches.

3. Single Pixel Attention Block:

    After feature extraction, the model passes the output through a single pixel attention block. This block focuses on important local pixel information, allowing the model to enhance crucial areas of the image.

4. Sub-Pixel Convolution Layer:
   
    The final stage of the model involves a sub-pixel convolution layer, which performs upsampling of the image patches. This layer helps reconstruct the high-resolution output from the processed feature maps.

The make_classif_mob.ipynb file uses MobileNet V2 to classify whether a given input image is pixelated or not. If it is pixelated, the image is sent to the ESPCN model defined in espcn_model (1).ipynb. 

The model works by creating patches of the input pixelated images and enhancing each patch of the image one by one. Finally, the patches from the model are processed to remove any duplicated features and are combined together which would be depixelated.
