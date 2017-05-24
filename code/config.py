#!/usr/bin/env python

import tensorflow as tf


##### DIRECTORIES:
WORKSPACE = "/home/njk/Courses/EECS349/Project/"
DATA_DIR = WORKSPACE + "data/LUNA2016/"
INPUT_IMAGE_DIR = DATA_DIR + "subset0/"
JPEG_IMAGE_DIR = DATA_DIR + "images/"
MODEL_SAVE_DIR = WORKSPACE + "models/"


##### IMAGE DATA CONSTANTS:
IMAGE_SIZE = 40 # image width / height (must be square)
NUM_CHANNELS = 1 # grayscale = 1, RGB = 3
NUM_CLASSES = 2 # binary classification, easier to set up model with 2 instead of 1


##### TRAINING CONSTANTS:
NUM_TRAIN_EXAMPLES = 99
NUM_VALIDATION_EXAMPLES = 11
NUM_TEST_EXAMPLES = 22
BATCH_SIZE = 10
DROPOUT_RATE = 0.35
NUM_ITERS=10000


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


##### TENSORFLOW FLAGS AND VARIABLES:
FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)
tf.app.flags.DEFINE_integer('batch_size', BATCH_SIZE, """number of images to process in a batch""")
FLAGS.model_dir = MODEL_SAVE_DIR


if __name__ == '__main__':
    print "WARNING: this file has no executable code and should NOT be invoked standalone"
