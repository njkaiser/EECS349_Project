#!/usr/bin/env python
''' main function for project - invoked by tensorflow to call all auxiliary functions '''

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from import_data import import_data
from model import build_model

from config import FLAGS, BATCH_SIZE, NUM_ITERS, MODEL_SAVE_DIR


# Variables must be initialized by running an `init` Op after having
# launched the graph. We first have to add the `init` Op to the graph.
init_op = tf.global_variables_initializer()
CONFIG = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# TODO: ADD ANY OTHER NECESARY TF VARIABLES HERE
# tf.logging.set_verbosity(tf.logging.INFO)


def main(argv):
    with tf.Session(config=CONFIG) as sess:

        ##### STEP 1: INITIALIZE ALL NECESSARY TENSORFLOW VARIABLES
        sess.run(init_op)


        ##### STEP 2: GRAB 4D TENSOR DATA FOR TRAINING, VALIDATION, AND TESTING
        train_data, validation_data, test_data, train_labels, validation_labels, test_labels = import_data()
        # print validation_data
        # print validation_labels


        ##### STEP 3: BUILD MODEL(S)
        # train_model = build_model(train_data, train_labels, learn.ModeKeys.TRAIN)
        # saver = tf.train.Saver()
        # saver.save(sess, MODEL_SAVE_DIR + '/test.chkp')
        # TODO:
            # do we actually need to build 2 models? 1 for train and 1 for test? What about validate?

        # Set up logging for predictions
        # Log the values in the "Softmax" tensor with label "probabilities"
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)


        ##### STEP 4: TRAIN
        train_classifier = learn.SKCompat(learn.Estimator(model_fn=build_model, model_dir=MODEL_SAVE_DIR))

        # train_model.fit(input_fn=build_model, steps=200)
        # results = train_model.evaluate(input_fn=build_model, steps=1)
        # for key in sorted(results):
        #     print("%s: %s" % (key, results[key]))

        # print type(train_data)
        # print type(train_labels)
        # print type([logging_hook])

        # train_model.fit(x=train_data, y=train_labels, batch_size=BATCH_SIZE, steps=51, monitors=[logging_hook])

        train_classifier.fit(x=train_data, y=train_labels, batch_size=BATCH_SIZE, steps=51, monitors=[logging_hook])
        #
        # train_model.train(build_model, hooks=None, steps=NUM_ITERS, max_steps=None)
        # TODO:
            # make sure we train for a period, pause, and test on validation

# Train the model
# mnist_classifier.fit(
#     x=train_data,
#     y=train_labels,
#     batch_size=100,
#     steps=20000,
#     monitors=[logging_hook])


        ##### STEP 5: TEST
        # Configure the accuracy metric for evaluation
        # metrics = {"accuracy": learn.MetricSpec( metric_fn=tf.metrics.accuracy, prediction_key="classes"), }
        #
        # test_results = mnist_classifier.evaluate(x=test_data, y=test_labels, metrics=metrics)
        #
        # print(test_results)
        # TODO:
            # final test run against test data
            # export data to file and run any processing scripts we want to write


if __name__ == '__main__':
    tf.app.run()
