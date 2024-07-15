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

The classifier.ipynb file uses MobileNet V2 to classify whether a given input image is pixelated or not. If it is pixelated, the image is sent to the ESPCN model defined in espcn_model.ipynb.

