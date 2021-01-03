from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD


import matplotlib.pyplot as plt


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def plotAccuracyLossForValidationAndTrain(model_fit_history):
    acc = model_fit_history.history['acc']
    val_acc = model_fit_history.history['val_acc']

    loss = model_fit_history.history['loss']
    val_loss = model_fit_history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def plotAccuracyLossForTestAndTrain(model_fit_history):
    acc = model_fit_history.history['acc']
    val_acc = model_fit_history.history['val_acc']

    loss = model_fit_history.history['loss']
    val_loss = model_fit_history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Test Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Test Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Test Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Test Loss')
    plt.show()


epochs = 45
batch_size = 32
IMG_SIZE = 150


train_path = 'dogs vs cats/training_set/'
validation_path = 'dogs vs cats/validation_set/'
test_path = 'dogs vs cats/test_set/'


def run_test_harness():

    model = define_model()
    train_datagen = ImageDataGenerator(rotation_range=15,
                                       rescale=1./255,
                                       shear_range=0.1,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1)

    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    # prepare iterators
    train_dataset = train_datagen.flow_from_directory(train_path,
                                                      class_mode='binary',
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      target_size=(IMG_SIZE, IMG_SIZE))

    validation_dataset = validation_datagen.flow_from_directory(validation_path,
                                                                class_mode='binary',
                                                                batch_size=batch_size,
                                                                target_size=(IMG_SIZE, IMG_SIZE))

    test_dataset = test_datagen.flow_from_directory(test_path,
                                                    class_mode='binary',
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    target_size=(IMG_SIZE, IMG_SIZE))

    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    # fit model

    history = model.fit_generator(train_dataset,
                                  steps_per_epoch=len(train_dataset),
                                  validation_data=test_dataset,
                                  validation_steps=len(test_dataset),
                                  epochs=epochs,
                                  callbacks=[monitor])


    # evaluate model
    _, acc = model.evaluate_generator(test_dataset, steps=len(test_dataset))
    print('> %.3f' % (acc * 100.0))
    # learning curves
    plotAccuracyLossForTestAndTrain(history)

 #   plotAccuracyLossForValidationAndTrain(history)


run_test_harness()



