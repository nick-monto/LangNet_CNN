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

img_height, img_width = 120, 200

num_epochs = 5


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result[0]


def load_image(path):
    img = Image.open(path).convert('L')  # read in as grayscale
    img = img.resize((img_width, img_height))
    # img.load()  # loads the image into memory
    img_data = np.asarray(img, dtype="float")
    return img_data


stim_train = pd.read_table('img_set_16k.txt',
                           delim_whitespace=True,
                           names=['stimulus', 'language'])

stim = stim_train['stimulus']

labels = pd.get_dummies(stim_train['language'])

# generate a train and validate set
X_train, X_val, y_train, y_val = train_test_split(stim,
                                                  labels,
                                                  test_size=0.2)

labels_train = y_train.values
labels_val = y_val.values

training_data_dir = 'Input_spectrogram_16k/Training'  # directory for training data
# test_data_dir = 'Input_spectrogram/Test'  # directory for test data

print("Preparing the input and labels...")
specs_train_input = []
for i in range(len(X_train)):
    specs_train_input.append(load_image(find(X_train.iloc[i],
                                             training_data_dir)))
specs_train_input = np.asarray(specs_train_input)
specs_train_input = specs_train_input.reshape((len(X_train),
                                               img_height, img_width, 1))
print('There are a total of {} training stimuli!'.format(specs_train_input.shape[0]))

specs_val_input = []
for i in range(len(X_val)):
    specs_val_input.append(load_image(find(X_val.iloc[i],
                                           training_data_dir)))
specs_val_input = np.asarray(specs_val_input)
specs_val_input = specs_val_input.reshape((len(X_val),
                                           img_height, img_width, 1))
print('There are a total of {} validation stimuli!'.format(specs_val_input.shape[0]))
print("Done!")


# set of augments that will be applied to the training data
datagen = ImageDataGenerator(rescale=1./255)


checkpoint = ModelCheckpoint('./weights_16k.best.hdf5', monitor='val_acc',
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

# Define the model: 4 convolutional layers, 4 max pools
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(img_height, img_width, 1),
                 kernel_initializer='TruncatedNormal',
                 activation='relu',
                 name='conv1'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))

model.add(Conv2D(64, (3, 3), padding='same',
                 activation='relu',
                 kernel_initializer='TruncatedNormal',
                 name='conv2'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu',
                 kernel_initializer='TruncatedNormal',
                 name='conv3'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))

model.add(Conv2D(256, (3, 3), padding='same',
                 activation='relu',
                 kernel_initializer='TruncatedNormal',
                 name='conv4'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool4'))

model.add(Flatten(name='flat1'))  # converts 3D feature mapes to 1D feature vectors
model.add(Dense(256, activation='relu',
                kernel_initializer='TruncatedNormal', name='fc1'))
model.add(Dropout(0.5, name='do1'))  # reset half of the weights to zero

model.add(Dense(8, activation='softmax', name='output'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# compute quantities required for featurewise normalization
datagen.fit(specs_train_input)
datagen.fit(specs_val_input)

print("Initializing the model...")
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(specs_train_input,
                                 labels_train,
                                 batch_size=16),
                    steps_per_epoch=len(X_train) / 16,
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks_list,
                    validation_data=datagen.flow(specs_val_input,
                                                 labels_val,
                                                 batch_size=16),
                    validation_steps=len(X_val) / 16)

model.save('LangNet_4Conv.h5')
# # this generator will read pictures found in a sub folder
# # it will indefinitely generate batches of augmented image data
# train_generator = train_datagen.flow_from_directory(
#         training_data_dir,
#         target_size=(img_width, img_height),
#         batch_size=32,
#         class_mode='categorical')  # need categorical labels
#
# validation_generator = test_datagen.flow_from_directory(
#         test_data_dir,
#         target_size=(img_width, img_height),
#         batch_size=32,
#         class_mode='categorical')

# model.fit_generator(
#         train_generator,
#         samples_per_epoch=num_train_samples,
#         nb_epoch=num_epoch,
#         validation_data=validation_generator,
#         nb_val_samples=num_val_samples,
#         verbose=1,
#         callbacks=callbacks_list
#         )

# model.save_weights("model_trainingWeights_final.h5")
# print("Saved model weights to disk")
#
# model.predict_generator(
#         test_generator,
#         val_samples=nb_test_samples)
