#!/usr/bin/env python
''' main function for project - invoked by tensorflow to call all auxiliary functions '''

import numpy as np
import tensorflow as tf
# from tensorflow.contrib import learn
# from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
# tf.logging.set_verbosity(tf.logging.INFO)
from import_data import import_data
# from pprint import pprint


# Variables must be initialized by running an `init` Op after having
# launched the graph. We first have to add the `init` Op to the graph.
init_op = tf.global_variables_initializer()
# TODO: ADD ANY OTHER NECESARY TF VARIABLES HERE


def main(argv):
    with tf.Session() as sess:

        # STEP 1: INITIALIZE ALL NECESSARY TENSORFLOW VARIABLES
        sess.run(init_op)


        # STEP 2: BUILD MODEL(S)
        # from model import build_model
        # model = build_model()
        # TODO:
            # do we actually need to build 2 models? 1 for train and 1 for test?


        # STEP 3: IMPORT IMAGE DATA AS 4D TENSOR
        data, labels = import_data()
        # print data
        print labels


        # STEP 4: TRAIN
        # TODO:
            # make sure we train for a period, pause, and test on validation


        # STEP 5: TEST
        # TODO:
            # final test run against test data
            # export data to file and run any processing scripts we want to write



if __name__ == '__main__':
    tf.app.run()
