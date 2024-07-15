import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, auc
import numpy as np
import matplotlib.pyplot as plt

# Define directories
base_dir = 'C:/Users/joela/OneDrive/Documents/Python'
low_res_dir = os.path.join(base_dir, 'low_res_images')
high_res_dir = os.path.join(base_dir, 'high_res_images')

# Parameters
img_height, img_width = 224, 224  # MobileNetV2 typically uses 224x224 input size
batch_size = 16
epochs = 3

# Collect all image file paths and labels
low_res_images = [(os.path.join(low_res_dir, fname), 0) for fname in os.listdir(low_res_dir)]
high_res_images = [(os.path.join(high_res_dir, fname), 1) for fname in os.listdir(high_res_dir)]

# Combine and split the dataset into training and validation sets
all_images = low_res_images + high_res_images
random.shuffle(all_images)

train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)

# Data generators
def create_datagen(image_list, batch_size, img_height, img_width):
    while True:
        for i in range(0, len(image_list), batch_size):
            batch_images = image_list[i:i + batch_size]
            images = []
            labels = []
            for img_path, label in batch_images:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)
            yield np.array(images), np.array(labels)

train_generator = create_datagen(train_images, batch_size, img_height, img_width)
val_generator = create_datagen(val_images, batch_size, img_height, img_width)

# Load the MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the convolutional base
for layer in base_model.layers:
    layer.trainable = False

# Create the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer=Adam(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


steps_per_epoch = len(train_images) // batch_size
validation_steps = len(val_images) // batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=validation_steps
)

#model.save('mob_clssifc.h5')
model.save('saved_model/mob_classifc')


val_images, val_labels = zip(*val_images)
val_images = np.array([tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(img, target_size=(img_height, img_width))) / 255.0 for img in val_images])
val_labels = np.array(val_labels)

predictions = model.predict(val_images)
predicted_labels = (predictions > 0.5).astype(int).flatten()

f1 = f1_score(val_labels, predicted_labels)
print(f'F1 Score: {f1}')

