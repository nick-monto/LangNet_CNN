import pandas as pd
import tensorflow as tf
import numpy as np
from scipy.misc import imread
from sklearn.model_selection import train_test_split
import os
import fnmatch

SCRIPT_DIR = os.getcwd()
INPUT_FOLDER = 'Input_spectrogram/Training/'
languages = os.listdir(INPUT_FOLDER)
languages.sort()

stim_train = pd.read_table('img_set.txt',
                           delim_whitespace=True,
                           names=['stimulus', 'language'])

# df = pd.concat([stim_train, pd.get_dummies(stim_train['language'])], axis=1); df

stim = stim_train['stimulus']

labels = pd.get_dummies(stim_train['language'])

# generate a train and validate set
X_train, X_test, y_train, y_test = train_test_split(stim,
                                                    labels,
                                                    test_size=0.2)

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result[0]

height, width, channels = imread(find(X_train.iloc[0], INPUT_FOLDER)).shape

def load_image(path):
    img = imread(path, flatten=True, mode='L').flatten()  # opens the image file, one channel
    data_scaled = img / 255  # converts each value to between 0 and 1
    return data_scaled

print("Loading images...")

specs_input = np.zeros((len(X_train), height*width*1))
for i in range(len(X_train)):
    specs_input[i] = load_image(find(X_train.iloc[i], INPUT_FOLDER))

specs_test_input = np.zeros((len(X_test), height*width*1))
for i in range(len(X_test)):
    specs_test_input[i] = load_image(find(X_test.iloc[i], INPUT_FOLDER))

print("Loading labels...")

labels_output = y_train.values
labels_test_output = y_test.values

# Initiate the session for TensorFlow
sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, height*width*1])
x_image = tf.reshape(x, [-1, height, width, 1])

# The number of labels is based on number of languages

y_ = tf.placeholder("float", shape=[None, len(languages)])


def weight_variable(shape):  # function to initialize weights with noise
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # function to initialize with zero biases
    initial = tf.zeros(shape)
    return tf.Variable(initial)


def convolve(x_in, weights):  # function for convolution with output and input the same size
    return tf.nn.conv2d(x_in, weights, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x_in):  # function for max pooling over 2x2 blocks
    return tf.nn.max_pool(x_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# LAYER 1 ::: Takes 1 image and results in 32 feature maps

W_conv1 = weight_variable([3, 3, 1, 32])  # 3x3 patch, 1 input channels (colors), 32 output channels (activation maps)
b_conv1 = bias_variable([32])  # 32 output channels (activation maps)

h_conv1 = convolve(x_image, W_conv1) + b_conv1  # perform convolution of image on weights using function and add biases
h_act1 = tf.nn.relu(h_conv1)
h_pool1 = max_pool(h_act1)

# LAYER 2 ::: Takes 32 activation maps (shape 32x16) and results in 64 activation maps
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = convolve(h_pool1, W_conv2) + b_conv2
h_act2 = tf.nn.relu(h_conv2)
h_pool2 = max_pool(h_act2)

# LAYER 3 ::: Takes 64 feature maps (shape 16x8) and results in 128 activation maps
W_conv3 = weight_variable([3, 3, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = convolve(h_pool2, W_conv3) + b_conv3
h_act3 = tf.nn.relu(h_conv3)
h_pool3 = max_pool(h_act3)

# DENSELY (FULLY) CONNECTED LAYER
# Before these calculations, we flatten the feature maps using reshape
# We also use matrix multiplication, instead of convolution
# There is no max pooling at this level either
h_flat3 = tf.reshape(h_pool3, [-1, 8 * 4 * 128])  # flatten the feature maps

W_fc1 = weight_variable([8 * 4 * 128, 500])  # 8x4x128 for the 128 flattened activation maps, 500 output channels (nodes)
b_fc1 = bias_variable([500])  # 500 output channels (nodes)

h_mm1 = tf.matmul(h_flat3, W_fc1) + b_fc1  # perform matrix multiplication and add biases
h_fc1 = tf.nn.relu(h_mm1)

# READOUT (OUTPUT) LAYER ::: Takes 500 nodes and results in a decision of size 8 (number of languages)
W_fc2 = weight_variable([500, 8])  # 500 input channels (nodes), 8 output channels (number of languages)
b_fc2 = bias_variable([8])  # 2 output channels (word or nonword)

h_mm2 = tf.matmul(h_fc1, W_fc2) + b_fc2
y = tf.nn.softmax(h_mm2)  # apply SoftMax to activations for the final output

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # our cost function is cross entropy between target and prediction
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # training should minimize cross entropy

# We also need some code to determine if the model is making accurate decisions. First, we will take the node with
# the highest activation in the prediction (y) and the label (y_) and see if they are the same. We then convert this
# to a numerical (float) value and take the mean of all of the input items being tested on.

high_pred = tf.argmax(y, 1)  # gives the node with highest activation from the model
high_real = tf.argmax(y_, 1)  # gives the node that should have the highest activation from the label

correct_prediction = tf.equal(high_pred, high_real)  # checks if they are equal and gives a boolean (TRUE or FALSE)
correct_float = tf.cast(correct_prediction, "float")  # converts to float (TRUE = 1, FALSE = 0)
accuracy = tf.reduce_mean(correct_float)  # calculates the mean accuracy of all items in the test set

# After the graph is complete, we need to initialize all of the variables.
sess.run(tf.global_variables_initializer())


# Time to see how this puppy does
print("Starting training...")

for i in range(len(X_train)):
    if i % 100 == 0:  # how frequently to check the accuracy
        test_accuracy = accuracy.eval(feed_dict={x: specs_input, y_: labels_output})
        print("Step %d accuracy %g" % (i, test_accuracy))

    train_step.run(feed_dict={x: [specs_input[i]], y_: [labels_output[i]]})

# After training is complete, we test on the entire set of unique input items to determine the final accuracy.

print("Final accuracy %g" % accuracy.eval(feed_dict={x: specs_test_input, y_: labels_test_output}))
