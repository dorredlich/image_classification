from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
IMG_HEIGHT = 256
IMG_WIDTH = 256

train_path = 'dogs vs cats/training_set/'
validation_path = 'dogs vs cats/validation_set/'
test_path = 'dogs vs cats/test_set'

# train = ImageDataGenerator(rescale=1. / 255,
#                            width_shift_range=0.1,
#                            height_shift_range=0.1,
#                            horizontal_flip=True)
datagen = ImageDataGenerator(rescale=1. / 255)

train_dataset = datagen.flow_from_directory(train_path,
                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                            batch_size=batch_size,
                                            class_mode='binary')

validation_dataset = datagen.flow_from_directory(validation_path,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 batch_size=batch_size,
                                                 class_mode='binary')

test_dataset = datagen.flow_from_directory(test_path,
                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                           batch_size=batch_size,
                                           class_mode='binary')

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_fit = model.fit_generator(train_dataset,
                                steps_per_epoch=20,
                                epochs=11,
                                validation_data=test_dataset,
                                validation_steps=len(test_dataset))


acc = model_fit.history['acc']
val_acc = model_fit.history['val_acc']

loss = model_fit.history['loss']
val_loss = model_fit.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Test Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Test Loss')
plt.show()
