#!/usr/bin/env python
''' converts individual png files to mini-batched tensor for use by tensorflow '''

import os
import numpy as np
import cv2
# print "OpenCV Version:", cv2.__version__
import tensorflow as tf
from pprint import pprint
# from tensorflow.contrib import learn
# from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
# tf.logging.set_verbosity(tf.logging.INFO)
from glob import glob


# directories:
workspace = "/home/njk/Courses/EECS349/Project/data/LUNA2016/"
image_dir = workspace + "output/"
image_list = glob(image_dir + "*.png")
# print num_images
# pprint(imt i, filename


##### USEFUL CONSTANTS:
IMAGE_SIZE = 40 # input image width / height (must be square)
NUM_IMAGES = len(image_list)
NUM_CLASSES = 2 # binary classification
BATCH_SIZE = 128 # size of the subset of examples to use when performing gradient descent during training
# IMAGE_WIDTH = 32 # example image width
# IMAGE_HEIGHT = 32 # example image height
# CHANNELS = 1 # 0 = same as PNG, 1 = grayscale
# DTYPE = tf.uint8 # [optional]: tf.uint8 or tf.uint16
# NAME = 'NATEISTHEBEST' # [optional]: name to identify the operation
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


##### BASIC MODEL PARAMETERS:
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', BATCH_SIZE, """number of images to process in a batch""")
# tf.app.flags.DEFINE_string('image_dir', '/tmp/cifar10_data', """path to image data""")
# tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")



# TODO: IMPORT:
# tf.image.decode_png(
#     contents,
#     channels=CHANNELS,
#     dtype=DTYPE,
#     name=NAME
# )

def import_data():
    images = np.zeros((NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE, 1))
    labels = np.zeros((NUM_IMAGES, NUM_CLASSES))

    for i, filename in enumerate(image_list):
        # print i, filename
        # import image and convert to numpy array of proper size:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image = image.astype(float)
        image = 1 - image/255
        # image.reshape((IMAGE_SIZE, IMAGE_SIZE, 1))
        # image = image[:, :, np.newaxis] # reshape isn't working for this?
        images[i] = image.reshape((IMAGE_SIZE, IMAGE_SIZE, 1))
        # print image.shape
        # print image
        # print image.max()
        # print image.min()

        # import labels and assign to correct classification:
        classification = filename[-7:-4]
        if classification == 'pos':
            labels[i] = np.array([1, 0]).reshape((1, 2))
            # print labels[i].shape
        elif classification == 'neg':
            labels[i] = np.array([0, 1]).reshape((1, 2))
            # print labels[i].shape
        else:
            print "ERROR: classification cannot be determined from filename:", filename
            assert False

    return images, labels

    # # if not eval_data:
    # # filenames = [os.path.join(image_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    # # num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    # # else:
    # #     filenames = [os.path.join(image_dir, 'test_batch.bin')]
    # #     num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    #
    # for f in image_list:
    #     if not tf.gfile.Exists(f):
    #         raise ValueError('Failed to find file: ' + f)
    #
    # # Create a queue that produces the filenames to read.
    # filename_queue = tf.train.string_input_producer(image_list)
    #
    # # Read examples from files in the filename queue.
    # read_input = read_cifar10(filename_queue)
    # reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    #
    # height = IMAGE_SIZE
    # width = IMAGE_SIZE
    #
    # # Image processing for training the network. Note the many random
    # # distortions applied to the image.
    #
    # # Randomly crop a [height, width] section of the image.
    # distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
    #
    # # Randomly flip the image horizontally.
    # distorted_image = tf.image.random_flip_left_right(distorted_image)
    #
    # # Because these operations are not commutative, consider randomizing
    # # the order their operation.
    # distorted_image = tf.image.random_brightness(distorted_image,
    # max_delta=63)
    # distorted_image = tf.image.random_contrast(distorted_image,
    # lower=0.2, upper=1.8)
    #
    # # Subtract off the mean and divide by the variance of the pixels.
    # float_image = tf.image.per_image_standardization(distorted_image)
    #
    # # Set the shapes of tensors.
    # float_image.set_shape([height, width, 3])
    # read_input.label.set_shape([1])
    #
    # # Ensure that the random shuffling has good mixing properties.
    # min_fraction_of_examples_in_queue = 0.4
    # min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
    # min_fraction_of_examples_in_queue)
    # print ('Filling queue with %d CIFAR images before starting to train. '
    # 'This will take a few minutes.' % min_queue_examples)
    #
    # num_preprocess_threads = 16
    # images, label_batch = tf.train.batch(
    # [image, label],
    # batch_size=BATCH_SIZE,
    # num_threads=num_preprocess_threads,
    # capacity=min_queue_examples + 3 * BATCH_SIZE)
    #
    # # Display the training images in the visualizer.
    # tf.summary.image('images', images)
    #
    # return images, tf.reshape(label_batch, [BATCH_SIZE])


if __name__ == '__main__':
    tf.app.run()
