"""Visualization of the filters based off of:
https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
"""
from __future__ import print_function

from scipy.misc import imsave
import numpy as np
import time
import os
import fnmatch
from PIL import Image
from keras.models import load_model
from keras import backend as K

# dimensions of the generated pictures for each filter.
img_width = 200
img_height = 120

# the name of the layer we want to visualize
layer_name = 'conv4'

# input directory
INPUT_FOLDER = 'Input_spectrogram_16k/Training/'


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
    img.load()  # loads the image into memory
    img_data = np.asarray(img, dtype="float")
    img_data = img_data / 255.
    img_data = img_data.reshape(1, img_height, img_width, 1)
    return img_data


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


# build the VGG16 network with ImageNet weights
model = load_model('LangNet_4Conv.h5')
print('Model loaded.')

model.summary()

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


kept_filters = []
for filter_index in range(0, 200):
    # we only scan through the first 100 filters,
    # but there are actually 256 of them
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # load in the desired image
    input_img_data = load_image(find('ces-0a71b112_converted_0.jpeg',
                                     INPUT_FOLDER))

    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

len(kept_filters)

# we will stich the best 25 filters on a 8 x 8 grid.
n = 5

# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 25 filters.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((height, width, 3))

# fill the picture with our saved filters
if width > height:
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stitched_filters[(img_height + margin) * i: (img_height + margin) * i + img_height,
                             (img_width + margin) * j: (img_width + margin) * j + img_width, :] = img
else:
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
# save the result to disk
imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)
