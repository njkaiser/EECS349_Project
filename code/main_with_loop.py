import tensorflow as tf
import numpy as np
from import_data import import_data
import os
from datetime import datetime


from config import IMAGE_SIZE, NUM_CLASSES, CONV1_NUM_FILTERS, CONV1_KERNEL_SIZE, CONV1_PADDING, CONV1_ACTIV_FUNC, POOL1_FILTER_SIZE, POOL1_STRIDE, CONV2_NUM_FILTERS, CONV2_KERNEL_SIZE, CONV2_PADDING, CONV2_ACTIV_FUNC, POOL2_FILTER_SIZE, POOL2_STRIDE, FC1_NUM_NEURONS, FC1_ACTIV_FUNC, DROPOUT_RATE, NUM_CHANNELS, CONV1_STRIDE, CONV2_STRIDE, POOL1_PADDING, POOL2_PADDING, MODEL_SAVE_DIR, TRAINING_LOG_DIR, VALIDATION_LOG_DIR, MODEL_NAME, LEARNING_RATE, NUM_ITERS, BATCH_SIZE

train_data, validation_data, test_data, train_labels, validation_labels, test_labels = import_data()

train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
validation_data = np.reshape(validation_data, (validation_data.shape[0], validation_data.shape[1] * validation_data.shape[2]))
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1] * test_data.shape[2]))

print "Train Data Size", train_data.shape
print "Train Labels Size", train_labels.shape
print "Validation Data Size", validation_data.shape
print "Validation Labels Size", validation_labels.shape
print "Test Data Size", test_data.shape
print "Test Labels Size", test_labels.shape

# sess = tf.InteractiveSession()

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


config_list = []
conv1_kernel_size = [[3, 3],[5, 5], [7, 7]]
conv2_kernel_size = [[3, 3],[5, 5], [7, 7]]
conv1_num_filters = [16, 32]
conv2_num_filters = [32, 64]
dropout_rates     = [0.1, 0.2, 0.3]
for c1ks in conv1_kernel_size:
    for c2ks in conv2_kernel_size:
        for c1nf in conv1_num_filters:
            for c2nf in conv2_num_filters:
                for dr in dropout_rates:
                    config_list.append([c1ks, c2ks, c1nf, c2nf, dr])


for config in config_list:
    with tf.Session() as sess:
        # first convolutional layer
        # 32 features for each 5x5 patch
        # weight tensor will have a shape of [5,5,1,32]
        # first two dimensions are patch size (5x5)
        # next is the number of input channels (1)
        # last is the number of output channels (32)
        W_conv1 = weight_variable([config[0][0], config[0][1], NUM_CHANNELS, config[2]], 'W_conv1')
        b_conv1 = bias_variable([config[2]], 'b_conv1')
        tf.summary.image('W_conv1', tf.transpose(W_conv1, [3, 0, 1, 2]), max_outputs=config[2])

        # reshape x to a 4d tensor, 2nd and 3rd dimensions are image width and height, final dimension is the number of color channels
        x_image = tf.reshape(x, [-1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])

        # convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool
        h_conv1 = CONV1_ACTIV_FUNC(conv2d(x_image, W_conv1, CONV1_STRIDE, CONV1_PADDING) + b_conv1)
        # h_pool1 = max_pool_2x2(h_conv1, POOL1_FILTER_SIZE, POOL1_STRIDE, POOL1_PADDING)

        ## 2 SETS OF LAYERS
        # second convolutional layer
        # 64 features for each 5x5 patch
        W_conv2 = weight_variable([config[1][0], config[1][1], config[2], config[3]], 'W_conv2')
        b_conv2 = bias_variable([config[3]], 'b_conv2')

        tf.summary.image('W_conv2', tf.transpose(W_conv2[:, :, 0:1, :], [3, 0, 1, 2]), max_outputs=config[3])

        # convolve the result of h_pool1 with the weight tensor, add the bias, apply the ReLU function, and finally max pool
        h_conv2 = CONV2_ACTIV_FUNC(conv2d(h_conv1, W_conv2, CONV2_STRIDE, CONV2_PADDING) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2, POOL2_FILTER_SIZE, POOL2_STRIDE, POOL2_PADDING)

        # densely connected layer
        # image size has been reduced to 10x10 so we will add a fully-connected layer with 1024 neurons
        p2s = h_pool2.get_shape().as_list()
        W_fc1 = weight_variable([p2s[1]*p2s[2]*config[3], FC1_NUM_NEURONS], 'W_fc1')
        b_fc1 = bias_variable([FC1_NUM_NEURONS], 'b_fc1')

        h_pool2_flat = tf.reshape(h_pool2, [-1, p2s[1]*p2s[2]*config[3]])
        h_fc1 = FC1_ACTIV_FUNC(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


        ## 1 SET OF LAYERS
        # p1s = h_pool1.get_shape().as_list()
        #
        # # densely connected layer
        # # image size has been reduced to 10x10 so we will add a fully-connected layer with 1024 neurons
        # W_fc1 = weight_variable([p1s[1]*p1s[2]*CONV1_NUM_FILTERS, FC1_NUM_NEURONS], 'W_fc1')
        # b_fc1 = bias_variable([FC1_NUM_NEURONS], 'b_fc1')
        #
        # h_pool1_flat = tf.reshape(h_pool1, [-1, p1s[1]*p1s[2]*CONV1_NUM_FILTERS])
        # h_fc1 = FC1_ACTIV_FUNC(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)


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

        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("cross_entropy", cross_entropy)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(TRAINING_LOG_DIR, sess.graph)
        validation_writer = tf.summary.FileWriter(VALIDATION_LOG_DIR, sess.graph)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())


        step_list = []
        loss_list = []
        val_acc_list = []
        k = 0
        for i in range(NUM_ITERS):
            # If you want to run from a previous model, do so here:
            # saver.restore(sess, "models/model_1/test.ckpt")

            # TODO: Figure out batch training
            # batch = [train_data[i:i+50], train_labels[i:i+50]]

            j = 0
            while (j + BATCH_SIZE <= train_data.shape[0]):
                # print i,j,k
                batch = [train_data[j:j+BATCH_SIZE], train_labels[j:j+BATCH_SIZE]]
                summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: config[4]})
                train_writer.add_summary(summary, k)
                j += BATCH_SIZE
                k += 1

            if i % 10 == 0:
                summary, acc = sess.run([merged, accuracy], feed_dict={x: validation_data, y_: validation_labels, keep_prob: 1.0})
                validation_writer.add_summary(summary, k)
                print("step %d, validation accuracy %g"%(i, acc))
                step_list.append(i)
                val_acc_list.append(acc)
                save_path = saver.save(sess, MODEL_SAVE_DIR + "/" + MODEL_NAME + ".ckpt")
                print("Saved model %s at Step %d"%(MODEL_NAME,i))
            # summary, _ = sess.run([merged, train_step], feed_dict={x: train_data, y_: train_labels, keep_prob: DROPOUT_RATE})
            # train_writer.add_summary(summary, i)

        acc, y_c = sess.run([accuracy, y_conv], feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})

        print("final predictions",y_c)
        print("final test accuracy %g"%(acc))

        print("weights:", sess.run(W_fc2))
        print("biases:", sess.run(b_fc2))

        save_path = saver.save(sess, MODEL_SAVE_DIR + "/" + MODEL_NAME + ".ckpt")
        print("Saved final model to path: ", save_path)



        # ADDED BY NATE FOR EXPERIMENT
        print("step_list:\n", step_list)
        print("loss_list:\n", loss_list)
        print("val_acc_list:\n", val_acc_list)

        with open("experiment_output.txt", 'w') as f:
            f.write("experiment end time:" + str(datetime.now()) + '\n')
            f.write("configuration:\n")
            f.write("(conv1_kernel_size, conv2_kernel_size, conv1_num_filters, conv2_num_filters, dropout_rate)\n")
            for c in config:
                f.write(str(c) + ', ')
                f.seek(-2, os.SEEK_CUR)
                f.truncate()
            f.write("step:\n")
            for s in step_list:
                f.write(str(s) + ', ')
            f.seek(-2, os.SEEK_CUR)
            f.truncate()
            # f.write("\nloss:\n")
            # for l in loss_list:
            #     f.write(l + ', ')
            # f.seek(-2, os.SEEK_CUR)
            # f.truncate()
            f.write("\nvalidation accuracy:\n")
            for a in val_acc_list:
                f.write(str(a) + ', ')
            f.seek(-2, os.SEEK_CUR)
            f.truncate()

        # IMPORTANT: DO WE NEED TO RUN THIS TO START A NEW SESSION?
        # sess.close() #?????
