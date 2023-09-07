import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

import os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import livelossplot

batch_size = 500
epochs = 25
NUM_CLASSES = 10
IMG_HEIGHT = 28
IMG_WIDTH = 28
# TRAIN_PATH = "C:/Users/super/All CS Work/ML Class Work/5 MNIST/mnist_train.csv/"
# VAL_PATH = "C:/Users/super/All CS Work/ML Class Work/5 MNIST/mnist_test.csv/"\
plot_losses = livelossplot.PlotLossesKeras()
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], IMG_HEIGHT * IMG_WIDTH))
X_train = X_train.astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], IMG_HEIGHT * IMG_WIDTH))
X_test = X_test.astype('float32') / 255
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

#Build the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(IMG_WIDTH * IMG_HEIGHT,)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])# print the model architecture
model.summary()



history = model.fit(X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[],
    verbose=1,
    validation_data=(X_test, y_test)
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
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()