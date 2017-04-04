from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint

img_width, img_height = 384, 252  # standard dim output of mlab.specgram

training_data_dir = 'Input_spectrogram/training'  # directory for training data
test_data_dir = 'Input_spectrogram/validation'  # directory for test data

num_train_samples =  # TODO: get the total number of training files
num_val_samples =   # TODO: get the total number of validation files
num_epoch = 25

checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_acc',
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# # set up checkpoints for weights
# filepath="weights-improvement-{epoch:02d}-{accuracy:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath,
#                              monitor='accuracy',
#                              verbose=1,
#                              save_best_only=True,
#                              mode='max')
# callbacks_list = [checkpoint]

# model creation: Three convolutional layers
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_width, img_height, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # converts 3D feature mapes to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))  # reset half of the weights to zero

model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# set of augments that will be applied to the training data
# only turn images grayscale
train_datagen = ImageDataGenerator(rescale=1./255)

# only color rescale for test set
test_datagen = ImageDataGenerator(rescale=1./255)

# this generator will read pictures found in a sub folder
# it will indefinitely generate batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        training_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')  # need categorical labels

validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        samples_per_epoch=num_train_samples,
        nb_epoch=num_epoch,
        validation_data=validation_generator,
        nb_val_samples=num_val_samples,
        verbose=1,
        callbacks=callbacks_list
        )

# model.save_weights("model_trainingWeights_final.h5")
# print("Saved model weights to disk")
#
# model.predict_generator(
#         test_generator,
#         val_samples=nb_test_samples)
