#!/usr/bin/env python
''' converts individual png files to 4D image tensor for input into tensorflow '''

import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob

from config import IMAGE_SIZE, NUM_CHANNELS, NUM_CLASSES, NUM_TRAIN_EXAMPLES, NUM_VALIDATION_EXAMPLES, NUM_TEST_EXAMPLES, BATCH_SIZE, DATA_DIR, JPEG_IMAGE_DIR


def import_data():
    image_list = glob(JPEG_IMAGE_DIR + "*.png")
    num_images = len(image_list)
    images = np.zeros((num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), dtype=np.float32)
    labels = np.zeros((num_images, NUM_CLASSES), dtype=np.float32)

    for i, filename in enumerate(image_list):
        # import image and convert to numpy array of proper size:
        image = misc.imread(filename)
        image = image.astype(np.float32)
        image = 1 - image/255 # convert pixels from range [0, 255] to [0.0, 1.0]
        image = image[:, :, np.newaxis] # np.reshape isn't working how I want
        images[i] = image

        # import labels and assign to correct classification:
        classification = filename[-7:-4]
        if classification == 'pos':
            labels[i] = np.array([1.0, 0.0])
        elif classification == 'neg':
            labels[i] = np.array([0.0, 1.0])
        else:
            print "ERROR: classification cannot be determined from filename:", filename
            assert False

    # zero-center data over entire dataset (ONLY FOR TRAINING SET, subtract same mean value for test/validation sets)
    images -= np.mean(images[0:NUM_TRAIN_EXAMPLES, :, :, :])

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

    test_image_filenames = image_list[idx2:idx3]

    # print "train_data:\n", train_data
    # print "validation_data:\n", validation_data
    # print "test_data:\n", test_data

    return train_data, validation_data, test_data, train_labels, validation_labels, test_labels, test_image_filenames



if __name__ == '__main__':
    print "WARNING: this file should be invoked by tensorflow, not standalone"
    import_data()
