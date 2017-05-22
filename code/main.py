#!/usr/bin/env python
''' main function for project - invoked by tensorflow to call all auxiliary functions '''

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
# tf.logging.set_verbosity(tf.logging.INFO)
from import_data import import_data
from model import build_model
# from pprint import pprint
# from config import


# Variables must be initialized by running an `init` Op after having
# launched the graph. We first have to add the `init` Op to the graph.
init_op = tf.global_variables_initializer()
# TODO: ADD ANY OTHER NECESARY TF VARIABLES HERE


def main(argv):
    with tf.Session() as sess:

        # STEP 1: INITIALIZE ALL NECESSARY TENSORFLOW VARIABLES
        sess.run(init_op)


        # STEP 2: GRAB 4D TENSOR DATA FOR TRAINING, VALIDATION, AND TESTING
        train_data, validation_data, test_data, train_labels, validation_labels, test_labels = import_data()
        # print validation_data
        # print validation_labels


        # STEP 3: BUILD MODEL(S)
        train_model = build_model(train_data, train_labels, learn.ModeKeys.TRAIN)
        # TODO:
            # do we actually need to build 2 models? 1 for train and 1 for test? What about validate?


        # STEP 4: TRAIN
        # TODO:
            # make sure we train for a period, pause, and test on validation


        # STEP 5: TEST
        # TODO:
            # final test run against test data
            # export data to file and run any processing scripts we want to write



if __name__ == '__main__':
    tf.app.run()
