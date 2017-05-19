#!/usr/bin/env python
''' converts individual png files to the required tensor data structure '''

import numpy as np
import tensorflow as tf
# from tensorflow.contrib import learn
# from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
# tf.logging.set_verbosity(tf.logging.INFO)


The attr channels indicates the desired number of color channels for the decoded image.

Accepted values are:






##### USEFUL CONSTANTS:
# NUM_CLASSES = 2 # binary classification
# BATCH_SIZE = 128 # size of the subset of examples to use when performing gradient descent during training
# IMAGE_WIDTH = 32 # example image width
# IMAGE_HEIGHT = 32 # example image height
CHANNELS = 1 # 0 = same as PNG, 1 = grayscale
DTYPE = tf.uint8 # [optional]: tf.uint8 or tf.uint16
NAME = 'NATEISTHEBEST' # [optional]: name to identify the operation


tf.image.decode_png(
    contents,
    channels=CHANNELS,
    dtype=DTYPE,
    name=NAME
)



if __name__ == '__main__':
    # tf.app.run()
    pass
