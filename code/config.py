#!/usr/bin/env python

import tensorflow as tf

##### DIRECTORIES:
# workspace = "/home/njk/Courses/EECS349/Project/data/LUNA2016/"
# image_dir = workspace + "subset1/"
# output_dir = workspace + "output/"
# image_list = glob(image_dir + "*.mhd")
# pprint(image_list)


##### CONSTANTS:
IMAGE_SIZE = 40 # image width / height (must be square)
NUM_CHANNELS = 1 # grayscale = 1, RGB = 3
NUM_CLASSES = 2 # binary classification


NUM_TRAIN_EXAMPLES = 99
NUM_VALIDATION_EXAMPLES = 11
NUM_TEST_EXAMPLES = 22

BATCH_SIZE = 64 # size of the subset of examples to use when performing gradient descent during training
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
DROPOUT_RATE = 0.35

##### NETWORK ARCHITECTURE:
CONV1_NUM_FILTERS = 32
CONV1_KERNEL_SIZE = [5, 5]
CONV1_PADDING = 'same'
CONV1_ACTIV_FUNC = tf.nn.relu
POOL1_FILTER_SIZE = [2, 2]
POOL1_STRIDE = 2
CONV2_NUM_FILTERS = 64
CONV2_KERNEL_SIZE = [5, 5]
CONV2_PADDING = 'same'
CONV2_ACTIV_FUNC = tf.nn.relu
POOL2_FILTER_SIZE = [2, 2]
POOL2_STRIDE = 2
FC1_NUM_NEURONS = 1024
FC1_ACTIV_FUNC = tf.nn.relu


if __name__ == '__main__':
    print "WARNING: this file has no executable code and should NOT be invoked standalone"
