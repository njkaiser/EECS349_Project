#!/usr/bin/env python
''' sets up and creates a tensorflow CNN structure using tf layers '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from config import CONV1_NUM_FILTERS, CONV1_KERNEL_SIZE, CONV1_PADDING, CONV1_ACTIV_FUNC, POOL1_FILTER_SIZE, POOL1_STRIDE, CONV2_NUM_FILTERS, CONV2_KERNEL_SIZE, CONV2_PADDING, CONV2_ACTIV_FUNC, POOL2_FILTER_SIZE, POOL2_STRIDE, FC1_NUM_NEURONS, FC1_ACTIV_FUNC, NUM_CLASSES, BATCH_SIZE, DROPOUT_RATE, NUM_CHANNELS



def build_model(input_data, input_labels, mode):
    # print "SHAPE input =", input_data.shape

    tf.summary.image("1_before_conv1", input_data[0:1, :, :, 0:1], max_outputs=NUM_CHANNELS, collections=None)

    conv1 = tf.layers.conv2d(
        inputs=input_data,
        filters=CONV1_NUM_FILTERS,
        kernel_size=CONV1_KERNEL_SIZE,
        padding=CONV1_PADDING,
        activation=CONV1_ACTIV_FUNC)
    # print "SHAPE conv1 =", conv1.get_shape()

    tf.summary.image("2_after_conv1", conv1[0:1, :, :, 0:1], max_outputs=NUM_CHANNELS, collections=None)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=POOL1_FILTER_SIZE, strides=POOL1_STRIDE)
    # print "SHAPE pool1 =", pool1.get_shape()

    tf.summary.image("3_after_pool1", pool1[0:1, :, :, 0:1], max_outputs=NUM_CHANNELS, collections=None)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=CONV2_NUM_FILTERS,
        kernel_size=CONV2_KERNEL_SIZE,
        padding=CONV2_PADDING,
        activation=CONV2_ACTIV_FUNC)
    # print "SHAPE conv2 =", conv2.get_shape()

    tf.summary.image("4_after_conv2", conv2[0:1, :, :, 0:1], max_outputs=NUM_CHANNELS, collections=None)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=POOL2_FILTER_SIZE, strides=POOL2_STRIDE)
    # print "SHAPE pool2 =", pool2.get_shape()

    tf.summary.image("5_after_pool2", pool2[0:1, :, :, 0:1], max_outputs=NUM_CHANNELS, collections=None)

    p2s = pool2.get_shape().as_list()
    pool2_flat = tf.reshape(pool2, [-1, p2s[1] * p2s[2] * CONV2_NUM_FILTERS])
    # print "SHAPE pool2_flat =", pool2_flat.get_shape()

    dense = tf.layers.dense(inputs=pool2_flat, units=FC1_NUM_NEURONS, activation=FC1_ACTIV_FUNC)
    # print "SHAPE dense =", dense.get_shape()

    dropout = tf.layers.dropout(inputs=dense, rate=DROPOUT_RATE, training=mode == learn.ModeKeys.TRAIN)
    # print "SHAPE dropout =", dropout.get_shape()

    logits = tf.layers.dense(inputs=dropout, units=NUM_CLASSES)
    # print "SHAPE logits =", logits.get_shape()

    loss = None
    train_op = None
    # calculate loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=input_labels, depth=NUM_CLASSES)
        # print "SHAPE onehot_labels =", onehot_labels.get_shape()

        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # configure training op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

    # generate predictions
    predictions = {
        "classes": tf.argmax(
        input=logits, axis=1),
        "probabilities": tf.nn.softmax(
        logits, name="softmax_tensor")
        }

    # return a ModelFnOps object
    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


# THIS IS THROWING AN ERROR - WHY?
# if __name__ == '__main__':
#     print "WARNING: this file should be invoked by tensorflow, not standalone"
#     build_model()








# def cnn_model_fn(features, labels, mode):
#   """Model function for CNN."""
#   # Input Layer
#   # Reshape X to 4-D tensor: [batch_size, width, height, channels]
#   # MNIST images are 28x28 pixels, and have one color channel
#   input_layer = tf.reshape(features, [-1, 28, 28, 1])
#
#   # Convolutional Layer #1
#   # Computes 32 features using a 5x5 filter with ReLU activation.
#   # Padding is added to preserve width and height.
#   # Input Tensor Shape: [batch_size, 28, 28, 1]
#   # Output Tensor Shape: [batch_size, 28, 28, 32]
#   conv1 = tf.layers.conv2d(
#       inputs=input_layer,
#       filters=32,
#       kernel_size=[5, 5],
#       padding="same",
#       activation=tf.nn.relu)
#
#   # Pooling Layer #1
#   # First max pooling layer with a 2x2 filter and stride of 2
#   # Input Tensor Shape: [batch_size, 28, 28, 32]
#   # Output Tensor Shape: [batch_size, 14, 14, 32]
#   pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
#
#   # Convolutional Layer #2
#   # Computes 64 features using a 5x5 filter.
#   # Padding is added to preserve width and height.
#   # Input Tensor Shape: [batch_size, 14, 14, 32]
#   # Output Tensor Shape: [batch_size, 14, 14, 64]
#   conv2 = tf.layers.conv2d(
#       inputs=pool1,
#       filters=64,
#       kernel_size=[5, 5],
#       padding="same",
#       activation=tf.nn.relu)
#
#   # Pooling Layer #2
#   # Second max pooling layer with a 2x2 filter and stride of 2
#   # Input Tensor Shape: [batch_size, 14, 14, 64]
#   # Output Tensor Shape: [batch_size, 7, 7, 64]
#   pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#
#   # Flatten tensor into a batch of vectors
#   # Input Tensor Shape: [batch_size, 7, 7, 64]
#   # Output Tensor Shape: [batch_size, 7 * 7 * 64]
#   pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
#
#   # Dense Layer
#   # Densely connected layer with 1024 neurons
#   # Input Tensor Shape: [batch_size, 7 * 7 * 64]
#   # Output Tensor Shape: [batch_size, 1024]
#   dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
#
#   # Add dropout operation; 0.6 probability that element will be kept
#   dropout = tf.layers.dropout(
#       inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
#
#   # Logits layer
#   # Input Tensor Shape: [batch_size, 1024]
#   # Output Tensor Shape: [batch_size, 10]
#   logits = tf.layers.dense(inputs=dropout, units=10)
#
#   loss = None
#   train_op = None
#
#   # Calculate Loss (for both TRAIN and EVAL modes)
#   if mode != learn.ModeKeys.INFER:
#     onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
#     loss = tf.losses.softmax_cross_entropy(
#         onehot_labels=onehot_labels, logits=logits)
#
#   # Configure the Training Op (for TRAIN mode)
#   if mode == learn.ModeKeys.TRAIN:
#     train_op = tf.contrib.layers.optimize_loss(
#         loss=loss,
#         global_step=tf.contrib.framework.get_global_step(),
#         learning_rate=0.001,
#         optimizer="SGD")
#
#   # Generate Predictions
#   predictions = {
#       "classes": tf.argmax(
#           input=logits, axis=1),
#       "probabilities": tf.nn.softmax(
#           logits, name="softmax_tensor")
#   }
#
#   # Return a ModelFnOps object
#   return model_fn_lib.ModelFnOps(
#       mode=mode, predictions=predictions, loss=loss, train_op=train_op)
#
#
# def main(unused_argv):
#   # Load training and eval data
#   mnist = learn.datasets.load_dataset("mnist")
#   train_data = mnist.train.images  # Returns np.array
#   train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#   eval_data = mnist.test.images  # Returns np.array
#   eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
#
#   # Create the Estimator
#   mnist_classifier = learn.Estimator(
#       model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
#
#   # Set up logging for predictions
#   # Log the values in the "Softmax" tensor with label "probabilities"
#   tensors_to_log = {"probabilities": "softmax_tensor"}
#   logging_hook = tf.train.LoggingTensorHook(
#       tensors=tensors_to_log, every_n_iter=50)
#
#   # Train the model
#   mnist_classifier.fit(
#       x=train_data,
#       y=train_labels,
#       batch_size=100,
#       steps=20000,
#       monitors=[logging_hook])
#
#   # Configure the accuracy metric for evaluation
#   metrics = {
#       "accuracy":
#           learn.MetricSpec(
#               metric_fn=tf.metrics.accuracy, prediction_key="classes"),
#   }
#
#   # Evaluate the model and print results
#   eval_results = mnist_classifier.evaluate(
#       x=eval_data, y=eval_labels, metrics=metrics)
#   print(eval_results)
