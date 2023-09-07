import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

import os
import numpy as np
import matplotlib.pyplot as plt

batch_size = 100
epochs = 25
IMG_HEIGHT = 150
IMG_WIDTH = 150
TRAIN_PATH = "C:/Users/super/All CS Work/ML Class Work/7 CNN/train/"
TEST_PATH = "C:/Users/super/All CS Work/ML Class Work/7 CNN/test1/test1"
VAL_PATH = "C:/Users/super/All CS Work/ML Class Work/7 CNN/testwlabels"

train_image_generator = ImageDataGenerator(rescale = 1./255, rotation_range = 45, width_shift_range = .15, height_shift_range = .15, horizontal_flip = True, zoom_range = .3)

validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,directory=TRAIN_PATH,shuffle=True,          target_size=(IMG_HEIGHT, IMG_WIDTH),                      class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size, directory = VAL_PATH, target_size = (IMG_HEIGHT, IMG_WIDTH), class_mode = 'binary')

#Build the model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])# Compile the model
model.compile(optimizer='adam',              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),              metrics=['accuracy'])# print the model architecture
model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=220,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=20
)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()