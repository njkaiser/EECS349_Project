#!/usr/bin/env python
''' main function for project - invoked by tensorflow to call all auxiliary functions '''

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from import_data import import_data
from model import build_model

from config import FLAGS, BATCH_SIZE, NUM_ITERS, MODEL_SAVE_DIR, WORKSPACE


def main(argv):
    with tf.Session() as sess:


        ##### STEP 1: INITIALIZE ALL NECESSARY TENSORFLOW VARIABLES
        init_op = tf.global_variables_initializer()
        train_writer = tf.summary.FileWriter(WORKSPACE + 'tensorboard_log/' + 'train/', sess.graph)
        valid_writer = tf.summary.FileWriter(WORKSPACE + 'tensorboard_log/' + 'valid/', sess.graph)
        # TODO: ADD ANY OTHER NECESSARY TF STUFF HERE
        sess.run(init_op)


        ##### STEP 2: GRAB 4D TENSOR DATA FOR TRAINING, VALIDATION, AND TESTING
        train_data, validation_data, test_data, train_labels, validation_labels, test_labels = import_data()


        ##### STEP 3: SET UP LOGGING
        # set up logging for predictions
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

        validation_metrics = {
            "accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"),
            }

        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            validation_data,
            validation_labels,
            every_n_steps=50,
            metrics=validation_metrics,
            early_stopping_metric="loss",
            early_stopping_metric_minimize=True,
            early_stopping_rounds=200)


        ##### STEP 4: TRAIN
        train_classifier = learn.Estimator(model_fn=build_model, model_dir=MODEL_SAVE_DIR)
        train_classifier.fit(x=train_data, y=train_labels, batch_size=BATCH_SIZE, steps=NUM_ITERS, monitors=[validation_monitor])


        ##### STEP 5: TEST
        # Configure the accuracy metric for evaluation
        metrics = {
            "accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"),
            }

        # Evaluate the model and print results
        eval_results = train_classifier.evaluate(x=test_data, y=test_labels, metrics=metrics)
        print(eval_results)
        # TODO:
            # final test run against test data
            # export data to file and run any processing scripts we want to write


if __name__ == '__main__':
    tf.app.run()
