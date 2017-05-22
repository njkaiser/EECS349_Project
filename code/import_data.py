#!/usr/bin/env python
''' converts individual png files to 4D image tensor for input into tensorflow '''

import os
import numpy as np
# import cv2
# print "OpenCV Version:", cv2.__version__
from scipy import misc

import tensorflow as tf
from pprint import pprint
from config import IMAGE_SIZE, NUM_CHANNELS, NUM_CLASSES, NUM_TRAIN_EXAMPLES, NUM_VALIDATION_EXAMPLES, NUM_TEST_EXAMPLES
from glob import glob

import matplotlib.pyplot as plt

##### DIRECTORIES:
workspace = "/home/njk/Courses/EECS349/Project/data/LUNA2016/"
example_image_dir = workspace + "output/"
image_list = glob(example_image_dir + "*.png")


##### CONSTANTS:
BATCH_SIZE = 64 # size of the subset of examples to use when performing gradient descent during training
NUM_IMAGES = len(image_list)
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


##### TENSORFLOW MODEL PARAMETERS:
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', BATCH_SIZE, """number of images to process in a batch""")
# tf.app.flags.DEFINE_string('example_image_dir', '/tmp/cifar10_data', """path to image data""")
# tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")


def import_data():
    images = np.zeros((NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), dtype=np.float32)
    labels = np.zeros((NUM_IMAGES, NUM_CLASSES), dtype=np.float32)

    for i, filename in enumerate(image_list):
        # print i, filename
        # import image and convert to numpy array of proper size:
        # image = tf.image.decode_png(filename, channels=3, dtype=tf.uint8, name=None)
        # image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image = misc.imread(filename)
        # print image.shape
        # print type(image)
        # print type(image[0][0])
        # image = image.astype(np.float32)
        # print type(image[0][0])


        # img_tf = tf.Variable(image)
        # # print img_tf.get_shape().as_list()
        #
        # init = tf.global_variables_initializer()
        # sess = tf.Session()
        # sess.run(init)
        # im = sess.run(img_tf)

        # fig = plt.figure()
        # fig.add_subplot(1,2,1)
        # plt.imshow(im)
        # fig.add_subplot(1,2,2)
        # plt.imshow(image)
        # plt.show()
        # assert False




        image = image.astype(np.float32)
        image = 1 - image/255 # convert pixels from range [0, 255] [0.0, 1.0]

        # image.reshape((IMAGE_SIZE, IMAGE_SIZE, 1))
        # image = image[:, :, np.newaxis] # reshape isn't working for this?
        # image.reshape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        image = image[:, :, np.newaxis] # reshape isn't working how I want
        # print "shape1:", image.shape
        # print "type1:", type(image[0][0][0])
        # image.reshape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32)
        # image = image.astype(np.float32)
        # print "shape2:", image.shape
        # print "type2:", type(image[0][0][0])

        # tf.image.convert_image_dtype(image, tf.float32, saturate=True, name=None)
        # tf.image.convert_image_dtype(image.reshape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)), tf.float32, saturate=True, name=None)
        # image = tf.cast(image.reshape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)), tf.float32)
        # tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
        # print tf.shape(image)
        # images[i] = tf.cast(image.reshape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)), tf.float32)

        # print "image  type:", type(image[0][0][0])
        images[i] = image
        # print "images type:", type(images[i][0][0][0])
        # images[i] = image[:, :, np.newaxis]
        # print image.shape
        # print image
        # print "image.max():", image.msax(), "image.min()", image.min()

        # import labels and assign to correct classification:
        classification = filename[-7:-4]
        if classification == 'pos':
            labels[i] = np.array([1]).reshape((NUM_CLASSES, NUM_CHANNELS)) # not necessary, but ensures we know where the error is if we ever change NUM_CLASSES or NUM_CHANNELS
            # print labels[i].shape
        elif classification == 'neg':
            labels[i] = np.array([0]).reshape((NUM_CLASSES, NUM_CHANNELS)) # not necessary, but ensures we know where the error is if we ever change NUM_CLASSES or NUM_CHANNELS
            # print labels[i].shape
        else:
            print "ERROR: classification cannot be determined from filename:", filename
            assert False

    # zero-center data over entire dataset (ONLY FOR TRAINING SET, subtract same mean value for test/validation sets)
    # print "images.max():", images.max(), "images.min()", images.min()
    # print "average pixel value:", np.mean(images)
    # print "images.shape:", images.shape

    # print "type1:", type(images[0][0][0][0])
    images -= np.mean(images[0:NUM_TRAIN_EXAMPLES, :, :, :])
    # print "type2:", type(images[0][0][0][0])
    print "np.std(images[0:NUM_TRAIN_EXAMPLES, :, :, :]) =", np.std(images[0:NUM_TRAIN_EXAMPLES, :, :, :])
    # print "type3:", type(images[0][0][0][0])
    # print "after subtraction: images.max():", images.max(), "images.min()", images.min()

    # split image data into train, validation, and test sets:
    idx1 = NUM_TRAIN_EXAMPLES
    idx2 = NUM_TRAIN_EXAMPLES + NUM_VALIDATION_EXAMPLES
    idx3 = NUM_TRAIN_EXAMPLES + NUM_VALIDATION_EXAMPLES + NUM_TEST_EXAMPLES

    train_data = images[0:idx1, :, :, :]
    validation_data = images[idx1:idx2, :, :, :]
    test_data = images[idx2:idx3, :, :, :]

    train_labels = labels[0:idx1]
    validation_labels = labels[idx1:idx2]
    test_labels = labels[idx2:idx3]
    # print "type3:", type(train_data[0][0][0][0])

    # print "train, validation, and test data {{{ label }}} shapes:"
    # print train_data.shape, "{{{", train_labels.shape, "}}}"
    # print validation_data.shape, "{{{", validation_labels.shape, "}}}"
    # print test_data.shape, "{{{", test_labels.shape, "}}}"

    return train_data, validation_data, test_data, train_labels, validation_labels, test_labels





# TODO: IMPORT:
# tf.image.decode_png(
#     contents,
#     channels=CHANNELS,
#     dtype=DTYPE,
#     name=NAME
# )

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
    print "WARNING: this file should be invoked by tensorflow, not standalone"
    import_data()
