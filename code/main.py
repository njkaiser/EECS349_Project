#!/usr/bin/env python
''' main function for project - invoked by tensorflow to call all auxiliary functions '''

import numpy as np
import tensorflow as tf
# from tensorflow.contrib import learn
# from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
# tf.logging.set_verbosity(tf.logging.INFO)
from import_data import import_data
# from pprint import pprint


def main(argv):
    data, labels = import_data()
    print data
    print labels



if __name__ == '__main__':
    tf.app.run()
