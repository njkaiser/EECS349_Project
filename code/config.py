#!/usr/bin/env python
''' parameters - much easier to manage all parameters from one file '''

import tensorflow as tf
import os


MODEL_NAME = "test_test_test"


##### DIRECTORIES:
WORKSPACE = str(os.getcwd()) + '/'
# WORKSPACE = "home/chainatee/Spring/349_Machine_Learning/git_repo/EECS349_Project/"
# WORKSPACE = "/Users/Adam/github/EECS349_Project/"
# WORKSPACE = "/home/njk/Courses/EECS349/Project/"
DATA_DIR = WORKSPACE + "data/LUNA2016/"
INPUT_IMAGE_DIR = DATA_DIR + "subset3/"
JPEG_IMAGE_DIR = DATA_DIR + "images/"
MODEL_SAVE_DIR = WORKSPACE + "model_tmp2/"
LOG_DIR = WORKSPACE + "log_tmp2/"
TRAINING_LOG_DIR = LOG_DIR + "train"
VALIDATION_LOG_DIR = LOG_DIR + "validation"


# check for / set up correct workspace directories:
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Un-comment if training single model
# if not os.path.exists(TRAINING_LOG_DIR):
#     os.makedirs(TRAINING_LOG_DIR)
# if not os.path.exists(VALIDATION_LOG_DIR):
#     os.makedirs(VALIDATION_LOG_DIR)

if not os.path.isdir(DATA_DIR) or \
    not os.path.isdir(JPEG_IMAGE_DIR) or \
    not os.path.isdir(MODEL_SAVE_DIR) or \
    not os.path.isdir(LOG_DIR):
    print "Expected directory structure does not exist - are you running from the root of the workspace?"
    exit(-1)


##### IMAGE DATA CONSTANTS:
IMAGE_SIZE = 40 # image width / height (must be square)
NUM_CHANNELS = 1 # grayscale = 1, RGB = 3
NUM_CLASSES = 2 # binary classification, easier to set up model with 2 instead of 1


##### TRAINING CONSTANTS:
NUM_TRAIN_EXAMPLES = 2064
NUM_VALIDATION_EXAMPLES = 442
NUM_TEST_EXAMPLES = 442
BATCH_SIZE = 104
NUM_ITERS = 251


##### HYPERPARAMETERS
DROPOUT_RATE = 0.2
KEEP_PROB = 1 - DROPOUT_RATE
LEARNING_RATE = 0.0001


##### BASIC ARCHITECTURE:
# CONV1_NUM_FILTERS = 16
# CONV1_KERNEL_SIZE = [3, 3]
# CONV1_STRIDE = 1
# CONV1_PADDING = 'SAME'
# CONV1_ACTIV_FUNC = tf.nn.relu
# POOL1_FILTER_SIZE = [2, 2]
# POOL1_STRIDE = 2
# POOL1_PADDING = 'SAME'
# CONV2_NUM_FILTERS = 32
# CONV2_KERNEL_SIZE = [7, 7]
# CONV2_STRIDE = 1
# CONV2_PADDING = 'SAME'
# CONV2_ACTIV_FUNC = tf.nn.relu
# POOL2_FILTER_SIZE = [2, 2]
# POOL2_STRIDE = 2
# POOL2_PADDING = 'SAME'
# FC1_NUM_NEURONS = 1024
# FC1_ACTIV_FUNC = tf.nn.relu

##### ARCHITECTURE 2:
# CONV1_NUM_FILTERS = 16
# CONV1_KERNEL_SIZE = [3, 3]
# CONV1_STRIDE = 1
# CONV1_PADDING = 'SAME'
# CONV1_ACTIV_FUNC = tf.nn.relu
# POOL1_FILTER_SIZE = [2, 2]
# POOL1_STRIDE = 2
# POOL1_PADDING = 'SAME'
# FC1_NUM_NEURONS = 1024
# FC1_ACTIV_FUNC = tf.nn.relu

##### ARCHITECTURE 3:
# CONV1_NUM_FILTERS = 16
# CONV1_KERNEL_SIZE = [3, 3]
# CONV1_STRIDE = 1
# CONV1_PADDING = 'SAME'
# CONV1_ACTIV_FUNC = tf.nn.relu
# POOL1_FILTER_SIZE = [2, 2]
# POOL1_STRIDE = 2
# POOL1_PADDING = 'SAME'
# CONV2_NUM_FILTERS = 32
# CONV2_KERNEL_SIZE = [5, 5]
# CONV2_STRIDE = 1
# CONV2_PADDING = 'SAME'
# CONV2_ACTIV_FUNC = tf.nn.relu
# POOL2_FILTER_SIZE = [2, 2]
# POOL2_STRIDE = 2
# POOL2_PADDING = 'SAME'
# CONV3_NUM_FILTERS = 64
# CONV3_KERNEL_SIZE = [7, 7]
# CONV3_STRIDE = 1
# CONV3_PADDING = 'SAME'
# CONV3_ACTIV_FUNC = tf.nn.relu
# POOL3_FILTER_SIZE = [2, 2]
# POOL3_STRIDE = 2
# POOL3_PADDING = 'SAME'
# FC1_NUM_NEURONS = 1024
# FC1_ACTIV_FUNC = tf.nn.relu

##### ARCHITECTURE 3:
CONV1_NUM_FILTERS = 16
CONV1_KERNEL_SIZE = [3, 3]
CONV1_STRIDE = 1
CONV1_PADDING = 'SAME'
CONV1_ACTIV_FUNC = tf.nn.relu
POOL1_FILTER_SIZE = [2, 2]
POOL1_STRIDE = 2
POOL1_PADDING = 'SAME'
CONV2_NUM_FILTERS = 32
CONV2_KERNEL_SIZE = [5, 5]
CONV2_STRIDE = 1
CONV2_PADDING = 'SAME'
CONV2_ACTIV_FUNC = tf.nn.relu
POOL2_FILTER_SIZE = [2, 2]
POOL2_STRIDE = 2
POOL2_PADDING = 'SAME'
CONV3_NUM_FILTERS = 64
CONV3_KERNEL_SIZE = [7, 7]
CONV3_STRIDE = 1
CONV3_PADDING = 'SAME'
CONV3_ACTIV_FUNC = tf.nn.relu
POOL3_FILTER_SIZE = [2, 2]
POOL3_STRIDE = 2
POOL3_PADDING = 'SAME'
CONV4_NUM_FILTERS = 64
CONV4_KERNEL_SIZE = [7, 7]
CONV4_STRIDE = 1
CONV4_PADDING = 'SAME'
CONV4_ACTIV_FUNC = tf.nn.relu
POOL4_FILTER_SIZE = [2, 2]
POOL4_STRIDE = 2
POOL4_PADDING = 'SAME'
FC1_NUM_NEURONS = 1024
FC1_ACTIV_FUNC = tf.nn.relu


##### TENSORFLOW FLAGS AND VARIABLES:
FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)
tf.app.flags.DEFINE_integer('batch_size', BATCH_SIZE, """number of images to process in a batch""")
#FLAGS.model_dir = MODEL_SAVE_DIR


if __name__ == '__main__':
    print "WARNING: this file has no executable code and should NOT be invoked standalone"
