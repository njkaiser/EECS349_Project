import tensorflow as tf
import cv2
import numpy as np
import os
from import_data import import_data

from config import IMAGE_SIZE, NUM_CLASSES, CONV1_NUM_FILTERS, CONV1_KERNEL_SIZE, CONV1_PADDING, CONV1_ACTIV_FUNC, POOL1_FILTER_SIZE, POOL1_STRIDE, CONV2_NUM_FILTERS, CONV2_KERNEL_SIZE, CONV2_PADDING, CONV2_ACTIV_FUNC, POOL2_FILTER_SIZE, POOL2_STRIDE, FC1_NUM_NEURONS, FC1_ACTIV_FUNC, DROPOUT_RATE, NUM_CHANNELS, CONV1_STRIDE, CONV2_STRIDE, POOL1_PADDING, POOL2_PADDING, MODEL_SAVE_DIR, TRAINING_LOG_DIR, VALIDATION_LOG_DIR, MODEL_NAME, LEARNING_RATE

train_data, validation_data, test_data, train_labels, validation_labels, test_labels = import_data()

train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
validation_data = np.reshape(validation_data, (validation_data.shape[0], validation_data.shape[1] * validation_data.shape[2]))

print "Train Data Size", train_data.shape
print "Train Labels Size", train_labels.shape
print "Validation Data Size", validation_data.shape
print "Validation Labels Size", validation_labels.shape

sess = tf.InteractiveSession()

# create a placeholder for input
x = tf.placeholder(tf.float32, [None, IMAGE_SIZE * IMAGE_SIZE])

# to implement cross-entropy we need to add a placeholder to input the correct answers
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# NOTE: info on convolutional neural networks: http://cs231n.github.io/convolutional-networks/

# create functions to initialize weights with a slightly positive initial bias to avoid "dead neurons"
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# convolution and pooling
def conv2d(x, W, stride, padding):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

def max_pool_2x2(x, ksize, stride, padding):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1], strides=[1, stride, stride, 1], padding=padding)

# first convolutional layer
# 32 features for each 5x5 patch
# weight tensor will have a shape of [5,5,1,32]
# first two dimensions are patch size (5x5)
# next is the number of input channels (1)
# last is the number of output channels (32)
W_conv1 = weight_variable([CONV1_KERNEL_SIZE[0],CONV1_KERNEL_SIZE[1],1,CONV1_NUM_FILTERS], 'W_conv1')
b_conv1 = bias_variable([CONV1_NUM_FILTERS], 'b_conv1')

# reshape x to a 4d tensor, 2nd and 3rd dimensions are image width and height, final dimension is the number of color channels
x_image = tf.reshape(x, [-1,IMAGE_SIZE,IMAGE_SIZE,1])

# convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool
h_conv1 = CONV1_ACTIV_FUNC(conv2d(x_image, W_conv1, CONV1_STRIDE, CONV1_PADDING) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1, POOL1_FILTER_SIZE, POOL1_STRIDE, POOL1_PADDING)

# second convolutional layer
# 64 features for each 5x5 patch
W_conv2 = weight_variable([CONV2_KERNEL_SIZE[0],CONV2_KERNEL_SIZE[1],CONV1_NUM_FILTERS,CONV2_NUM_FILTERS], 'W_conv2')
b_conv2 = bias_variable([CONV2_NUM_FILTERS], 'b_conv2')

# convolve the result of h_pool1 with the weight tensor, add the bias, apply the ReLU function, and finally max pool
h_conv2 = CONV2_ACTIV_FUNC(conv2d(h_pool1, W_conv2, CONV2_STRIDE, CONV2_PADDING) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2, POOL2_FILTER_SIZE, POOL2_STRIDE, POOL2_PADDING)

p2s = h_pool2.get_shape().as_list()

# densely connected layer
# image size has been reduced to 10x10 so we will add a fully-connected layer with 1024 neurons
W_fc1 = weight_variable([p2s[1]*p2s[2]*CONV2_NUM_FILTERS, FC1_NUM_NEURONS], 'W_fc1')
b_fc1 = bias_variable([FC1_NUM_NEURONS], 'b_fc1')

h_pool2_flat = tf.reshape(h_pool2, [-1, p2s[1]*p2s[2]*CONV2_NUM_FILTERS])
h_fc1 = FC1_ACTIV_FUNC(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
# to reduce overfitting, we apply dropout before the readout layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([FC1_NUM_NEURONS, NUM_CLASSES], 'W_fc2')
b_fc2 = bias_variable([NUM_CLASSES], 'b_fc2')

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# train and evaluate the model
# nearly identical to SoftMax code
# differences:
# replace gradient descent with ADAM optimizer
# we will include the additional parameter keep_prob in feed_dict to control the dropout rate
# we will add logging to every 100th iteration in the training process
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("cross_entropy", cross_entropy)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(TRAINING_LOG_DIR, sess.graph)
validation_writer = tf.summary.FileWriter(VALIDATION_LOG_DIR, sess.graph)

saver = tf.train.Saver()

# for i in range(1):
for i in range(20000):
    # If you want to run from a previous model, do so here:
    # saver.restore(sess, "models/model_1/test.ckpt")

    # TODO: Figure out batch training
    # batch = [train_data[i:i+50], train_labels[i:i+50]]

    if i % 10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict={x: validation_data, y_: validation_labels, keep_prob: 1.0})
        validation_writer.add_summary(summary, i)
        print("step %d, validation accuracy %g"%(i, acc))
        save_path = saver.save(sess, MODEL_SAVE_DIR + "/" + MODEL_NAME + ".ckpt")
        print("Saved model %s at Step %d"%(MODEL_NAME,i))
    summary, _ = sess.run([merged, train_step], feed_dict={x: train_data, y_: train_labels, keep_prob: DROPOUT_RATE})
    train_writer.add_summary(summary, i)

acc, y_c = sess.run([accuracy, y_conv], feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})

print("final predictions",y_c)
print("final test accuracy %g"%(acc))

print("weights:", sess.run(W_fc2))
print("biases:", sess.run(b_fc2))

saver = tf.train.Saver()

save_path = saver.save(sess, "model_tmp/test.ckpt")
print("Save to path: ", save_path)
