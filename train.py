import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
import cv2
import keras

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import pathlib

import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

data_path = 'C:\\Users\\Nandan M\\OneDrive\\Desktop\\Food Classification\\Food\\FINAL_DATASET'
# Initialize empty lists for storing the images and labels
images = []
labels = []

# Loop over the subfolders in the dataset
for subfolder in os.listdir(data_path):

    subfolder_path = os.path.join(data_path, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

  # Loop over the images in the subfolder
    for image_filename in os.listdir(subfolder_path):
       # Load the image and store it in the images list
        image_path = os.path.join(subfolder_path, image_filename)
        images.append(image_path)

        # Store the label for the image in the labels list
        labels.append(subfolder)

 # Create a pandas DataFrame from the images and labels
data = pd.DataFrame({'image': images, 'label': labels})

DIR = data_path

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(DIR, validation_split=0.1, 
                subset="training", seed=42,batch_size=16)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(DIR, validation_split=0.1,
                subset="validation", seed=42,batch_size=16)

classes = train_dataset.class_names
numClasses = len(train_dataset.class_names)
print(classes, numClasses)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.inception_v3.preprocess_input
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

model = tf.keras.Sequential([
     tf.keras.layers.Rescaling(1./255),
     tf.keras.layers.RandomFlip("horizontal"),
     tf.keras.layers.RandomRotation(0.1),
     tf.keras.layers.RandomZoom(0.1),
     tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
     tf.keras.layers.MaxPooling2D(),
    
     tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
     tf.keras.layers.MaxPooling2D(),
    
     tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
     tf.keras.layers.MaxPooling2D(),
    
     tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
     tf.keras.layers.MaxPooling2D(),
    
     tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu'),
     tf.keras.layers.MaxPooling2D(),
    
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(numClasses, activation='softmax')
 ])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer,
                        loss=loss,
                        metrics=['accuracy'])

epochs=250
callbacks = tf.keras.callbacks.EarlyStopping(patience=2)
history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)
    # , callbacks=[callbacks])

# plt.plot(range(0, epochs), history.history["loss"], color="b", label="Loss")
# plt.plot(range(0, epochs), history.history["val_loss"], color="r", label="Test Loss")
# plt.legend()
# plt.show()

model.save('food_0250.h5')

model.save('food_0250.keras')