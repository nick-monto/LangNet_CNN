import os
import fnmatch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint

img_width, img_height = 200, 125  # output of mlab.specgram

num_epochs = 10

training_data_dir = 'Input_spectrogram/Training'  # directory for training data
test_data_dir = 'Input_spectrogram/Test'  # directory for test data

datagen = ImageDataGenerator(rescale=1./255)


checkpoint = ModelCheckpoint('./weights.best.hdf5', monitor='accuracy',
                             verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# # set up checkpoints for weights
# filepath="weights-improvement-{epoch:02d}-{accuracy:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath,
#                              monitor='accuracy',
#                              verbose=1,
#                              save_best_only=True,
#                              mode='max')
# callbacks_list = [checkpoint]

# Define the model: 4 convolutional layers, 2 max pools
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # converts 3D feature mapes to 1D feature vectors
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))  # reset half of the weights to zero

model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


print("Initializing the model...")
# # fits the model on batches with real-time data augmentation:
# model.fit_generator(datagen.flow(specs_train_input,
#                                  labels_train,
#                                  batch_size=1),
#                     steps_per_epoch=len(X_train) / 1,
#                     epochs=num_epochs,
#                     verbose=1,
#                     callbacks=callbacks_list)


# this generator will read pictures found in a sub folder
# it will indefinitely generate batches of augmented image data
train_generator = datagen.flow_from_directory(
        training_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        shuffle=True,
        class_mode='categorical')  # need categorical labels

# validation_generator = test_datagen.flow_from_directory(
#         test_data_dir,
#         target_size=(img_width, img_height),
#         batch_size=32,
#         class_mode='categorical')

model.fit_generator(
        train_generator,
        samples_per_epoch=8000,
        nb_epoch=num_epochs,
        # validation_data=validation_generator,
        # nb_val_samples=num_val_samples,
        verbose=1,
        callbacks=callbacks_list
        )

# model.save_weights("model_trainingWeights_final.h5")
# print("Saved model weights to disk")
#
# model.predict_generator(
#         test_generator,
#         val_samples=nb_test_samples)
