import tensorflow as tf
import numpy as np
from import_data import import_data
import os
from datetime import datetime
from collections import OrderedDict
import shutil
import time

from config import WORKSPACE, LOG_DIR, MODEL_SAVE_DIR, IMAGE_SIZE, NUM_CLASSES, NUM_CHANNELS, CONV1_NUM_FILTERS, CONV1_KERNEL_SIZE, CONV1_ACTIV_FUNC, CONV1_STRIDE, CONV1_PADDING, POOL1_FILTER_SIZE, POOL1_STRIDE, POOL1_PADDING, CONV2_NUM_FILTERS, CONV2_KERNEL_SIZE, CONV2_ACTIV_FUNC, CONV2_STRIDE, CONV2_PADDING, POOL2_FILTER_SIZE, POOL2_STRIDE, POOL2_PADDING, FC1_ACTIV_FUNC, FC1_NUM_NEURONS, LEARNING_RATE, NUM_ITERS, BATCH_SIZE, DROPOUT_RATE

print "WORKSPACE:",WORKSPACE

train_data, validation_data, test_data, train_labels, validation_labels, test_labels, test_images_filenames = import_data()

train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
validation_data = np.reshape(validation_data, (validation_data.shape[0], validation_data.shape[1] * validation_data.shape[2]))
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1] * test_data.shape[2]))

print "Train Data Size", train_data.shape
print "Train Labels Size", train_labels.shape
print "Validation Data Size", validation_data.shape
print "Validation Labels Size", validation_labels.shape
print "Test Data Size", test_data.shape
print "Test Labels Size", test_labels.shape

model_base_name = "model_config_"
log_base_name = "log_config_"

# create a placeholder for input
x = tf.placeholder(tf.float32, [None, IMAGE_SIZE * IMAGE_SIZE])

# to implement cross entropy we need to add a placeholder to input the correct answers
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# placeholder for keep probability for the dropout layer
keep_prob = tf.placeholder(tf.float32)

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

def train_model_loop(iteration, config, config_names, output_file):
    with tf.Session() as sess:
        model_name = model_base_name + str(iteration)
        model_dir = MODEL_SAVE_DIR + model_name + '/'
        log_dir = LOG_DIR + log_base_name + str(iteration) + '/'

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print "PREPARING TO TRAIN:", model_name
        print "Current configuration is:", config
	initialTime = datetime.now()

        # first convolutional layer
        W_conv1 = weight_variable([config[0][0], config[0][1], NUM_CHANNELS, config[2]], 'W_conv1')
        b_conv1 = bias_variable([config[2]], 'b_conv1')
        tf.summary.image('W_conv1', tf.transpose(W_conv1, [3, 0, 1, 2]), max_outputs=config[2])

        # reshape x to a 4d tensor
        x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

        # convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool
        h_conv1 = CONV1_ACTIV_FUNC(conv2d(x_image, W_conv1, CONV1_STRIDE, CONV1_PADDING) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1, POOL1_FILTER_SIZE, POOL1_STRIDE, POOL1_PADDING)

        # second convolutional layer
        W_conv2 = weight_variable([config[1][0], config[1][1], config[2], config[3]], 'W_conv2')
        b_conv2 = bias_variable([config[3]], 'b_conv2')

        tf.summary.image('W_conv2', tf.transpose(W_conv2[:, :, 0:1, :], [3, 0, 1, 2]), max_outputs=config[3])

        # convolve the result of h_pool1 with the weight tensor, add the bias, apply the ReLU function, and finally max pool
        h_conv2 = CONV2_ACTIV_FUNC(conv2d(h_pool1, W_conv2, CONV2_STRIDE, CONV2_PADDING) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2, POOL2_FILTER_SIZE, POOL2_STRIDE, POOL2_PADDING)

        # densely connected layer
        p2s = h_pool2.get_shape().as_list()
        W_fc1 = weight_variable([p2s[1]*p2s[2]*config[3], FC1_NUM_NEURONS], 'W_fc1')
        b_fc1 = bias_variable([FC1_NUM_NEURONS], 'b_fc1')

        h_pool2_flat = tf.reshape(h_pool2, [-1, p2s[1]*p2s[2]*config[3]])
        h_fc1 = FC1_ACTIV_FUNC(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # readout layer
        W_fc2 = weight_variable([FC1_NUM_NEURONS, NUM_CLASSES], 'W_fc2')
        b_fc2 = bias_variable([NUM_CLASSES], 'b_fc2')

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # setup for training
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # setup to save accuracy and loss to logs
        # tf.summary.scalar("accuracy", accuracy)
        # tf.summary.scalar("cross_entropy", cross_entropy)
        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter(log_dir + 'train/', sess.graph)
        # validation_writer = tf.summary.FileWriter(log_dir + 'validation', sess.graph)

        # saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        step_list = []
        loss_list = []
        val_acc_list = []
        k = 0
	initialStepTime = datetime.now()
        for i in range(NUM_ITERS):
            j = 0
            while (j + BATCH_SIZE <= train_data.shape[0]):
                batch = [train_data[j:j+BATCH_SIZE], train_labels[j:j+BATCH_SIZE]]
                _ = sess.run([train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1-config[4]})
                # train_writer.add_summary(summary, k)
                j += BATCH_SIZE
                k += 1

            if i % 10 == 0:
                loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: validation_data, y_: validation_labels, keep_prob: 1.0})
                # validation_writer.add_summary(summary, k)
                print("Step %d, validation accuracy %g, step time %s"%(i, acc, str(datetime.now() - initialStepTime)))
		initialStepTime = datetime.now()
                step_list.append(i)
                loss_list.append(loss)
                val_acc_list.append(acc)
                # save_path = saver.save(sess, model_dir + model_name + ".ckpt")
                # print("Saved model %s at Step %d"%(model_name, i))

        acc, y_c = sess.run([accuracy, y_conv], feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})

        ##################### Save incorrectly classified images
        eq_arr = np.equal(np.argmax(y_c, axis = 1), np.argmax(test_labels, axis = 1))
        test_images_filenamesWithResult = ["%s_result%s.png" % t for t in zip(list(test_images_filenames), map(str,test_labels))]
        test_results = dict(zip(test_images_filenames, eq_arr))
        test_results_rename = dict(zip(test_images_filenames,test_images_filenamesWithResult))
        timenow = time.strftime("%d_%m_%Y-%H_%M_%S")
        test_result_path = MODEL_SAVE_DIR + "/" + model_name  + "/validation_result" + timenow

        if not os.path.isdir(test_result_path):
            os.makedirs(test_result_path)
        else:
            shutil.rmtree(test_result_path)
            os.makedirs(test_result_path)
        test_result_filenames = []

        for filename, result in test_results.iteritems():
            if not result:
                shutil.copy(filename, test_result_path)
                test_result_filenames.append(filename)
        ##################### (END) Save incorrectly classified images

        ##################### Create confusion matrix.txt
        pos_images, neg_images, pos_misclass_images, neg_misclass_images = 0,0,0,0
        for filename in test_images_filenames:
            if filename.find('pos') > 0:
                pos_images += 1
            else:
                neg_images += 1
        for filename in test_result_filenames:
            if filename.find('pos') > 0:
                pos_misclass_images += 1
            else:
                neg_misclass_images += 1
        f= open(test_result_path + "/confusion_"+timenow+".txt","w+")
        f.write("CONFUSION MATRIX for " + timenow + "\r\n")
        f.write("Ground truth|  1  |  0  |\r\n")
        f.write("Classified  |-----|-----|\r\n")
        f.write("     1      |  "+ str(pos_images-pos_misclass_images) + "  | " + str(neg_misclass_images) + "  |\r\n")
        f.write("     0      |  "+ str(pos_misclass_images) + "  | " + str(neg_images-neg_misclass_images) + "  |\r\n")
        f.write("misclassified images:\r\n")
        for filename in test_result_filenames:
            f.write(filename + "\r\n")
        print("Saved incorrectly classified image and confusion matrix to path:", test_result_path)
        f.close()
        ##################### (END) Create confusion matrix.txt

        # print("Final predictions",y_c)
        print("Final test accuracy for %s is %g"%(model_name, acc))
	print("Time to train model %s is %s"%(model_name, str(datetime.now() - initialTime)))

        # save_path = saver.save(sess, model_dir + model_name + ".ckpt")
        # print("Saved final %s to path %s: "%(model_name, save_path))

        # print current lists of experiment values
        print "step_list:", step_list
        print "loss_list:", loss_list
        print "val_acc_list:", val_acc_list

        print "\n"

        # write experiment output to file -- flag 'a' for append
        config_dict = OrderedDict(zip(config_names, config))
        with open(output_file, 'a') as f:
            f.write(model_name + '\n\n')
            f.write("experiment end time: " + str(datetime.now()) + '\n\n')
            f.write("configuration:\n")
            for c in config_dict:
                f.write(c + ' = ' + str(config_dict[c]) + '\n')
            f.write("\nstep:\n")
            f.write(','.join([str(s) for s in step_list]))
            f.write("\nloss:\n")
            f.write(','.join([str(l) for l in loss_list]))
            f.write("\nvalidation accuracy:\n")
            f.write(','.join([str(a) for a in val_acc_list]))
            f.write("\ntest accuracy:\n")
            f.write(str(acc))
            f.write("\n\n")
            f.write("--------------------")
            f.write("\n\n")

        # train_writer.close()
        # validation_writer.close()

def train_model(model_name, configuration, output_file):
    with tf.Session() as sess:
        model_name = model_name
        model_dir = MODEL_SAVE_DIR + model_name + '/'
        log_dir = LOG_DIR + model_name + '/'

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print "PREPARING TO TRAIN:", model_name

        # first convolutional layer
        W_conv1 = weight_variable([CONV1_KERNEL_SIZE[0],CONV1_KERNEL_SIZE[1],NUM_CHANNELS,CONV1_NUM_FILTERS], 'W_conv1')
        b_conv1 = bias_variable([CONV1_NUM_FILTERS], 'b_conv1')
        tf.summary.image('W_conv1', tf.transpose(W_conv1, [3, 0, 1, 2]), max_outputs=CONV1_NUM_FILTERS)

        # reshape x to a 4d tensor
        x_image = tf.reshape(x, [-1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])

        # convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool
        h_conv1 = CONV1_ACTIV_FUNC(conv2d(x_image, W_conv1, CONV1_STRIDE, CONV1_PADDING) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1, POOL1_FILTER_SIZE, POOL1_STRIDE, POOL1_PADDING)

        # second convolutional layer
        W_conv2 = weight_variable([CONV2_KERNEL_SIZE[0],CONV2_KERNEL_SIZE[1],CONV1_NUM_FILTERS,CONV2_NUM_FILTERS], 'W_conv2')
        b_conv2 = bias_variable([CONV2_NUM_FILTERS], 'b_conv2')

        tf.summary.image('W_conv2', tf.transpose(W_conv2[:, :, 0:1, :], [3, 0, 1, 2]), max_outputs=CONV2_NUM_FILTERS)

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

        # setup for training
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # setup to save accuracy and loss to logs
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("cross_entropy", cross_entropy)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + 'train/', sess.graph)
        validation_writer = tf.summary.FileWriter(log_dir + 'validation', sess.graph)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        step_list = []
        loss_list = []
        val_acc_list = []
        k = 0
        for i in range(NUM_ITERS):
            j = 0
            while (j + BATCH_SIZE <= train_data.shape[0]):
                batch = [train_data[j:j+BATCH_SIZE], train_labels[j:j+BATCH_SIZE]]
                summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: KEEP_PROB})
                train_writer.add_summary(summary, k)
                j += BATCH_SIZE
                k += 1

            if i % 10 == 0:
                summary, loss, acc = sess.run([merged, cross_entropy, accuracy], feed_dict={x: validation_data, y_: validation_labels, keep_prob: 1.0})
                validation_writer.add_summary(summary, k)
                print("Step %d, validation accuracy %g"%(i, acc))
                step_list.append(i)
                loss_list.append(loss)
                val_acc_list.append(acc)
                save_path = saver.save(sess, model_dir + model_name + ".ckpt")
                print("Saved model %s at Step %d"%(model_name, i))

        acc, y_c = sess.run([accuracy, y_conv], feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})

        ##################### Save incorrectly classified images
        eq_arr = np.equal(np.argmax(y_c, axis = 1), np.argmax(test_labels, axis = 1))
        test_images_filenamesWithResult = ["%s_result%s.png" % t for t in zip(list(test_images_filenames), map(str,test_labels))]
        test_results = dict(zip(test_images_filenames, eq_arr))
        test_results_rename = dict(zip(test_images_filenames,test_images_filenamesWithResult))
        timenow = time.strftime("%d_%m_%Y-%H_%M_%S")
        test_result_path = log_dir + "/validation_result" + timenow

        if not os.path.isdir(test_result_path):
            os.makedirs(test_result_path)
        else:
            shutil.rmtree(test_result_path)
            os.makedirs(test_result_path)
        test_result_filenames = []

        for filename, result in test_results.iteritems():
            if not result:
                shutil.copy(filename, test_result_path)
                test_result_filenames.append(filename)
        ##################### (END) Save incorrectly classified images

        ##################### Create confusion matrix.txt
        pos_images, neg_images, pos_misclass_images, neg_misclass_images = 0,0,0,0
        for filename in test_images_filenames:
            if filename.find('pos') > 0:
                pos_images += 1
            else:
                neg_images += 1
        for filename in test_result_filenames:
            if filename.find('pos') > 0:
                pos_misclass_images += 1
            else:
                neg_misclass_images += 1
        f= open(test_result_path + "/confusion_"+timenow+".txt","w+")
        f.write("CONFUSION MATRIX for " + timenow + "\r\n")
        f.write("Ground truth|  1  |  0  |\r\n")
        f.write("Classified  |-----|-----|\r\n")
        f.write("     1      |  "+ str(pos_images-pos_misclass_images) + "  | " + str(neg_misclass_images) + "  |\r\n")
        f.write("     0      |  "+ str(pos_misclass_images) + "  | " + str(neg_images-neg_misclass_images) + "  |\r\n")
        f.write("misclassified images:\r\n")
        for filename in test_result_filenames:
            f.write(filename + "\r\n")
        print("Saved incorrectly classified image and confusion matrix to path:", test_result_path)
        f.close()
        ##################### (END) Create confusion matrix.txt

        # print("Final predictions",y_c)
        print("Final test accuracy for %s is %g"%(model_name, acc))

        save_path = saver.save(sess, model_dir + model_name + ".ckpt")
        print("Saved final %s to path %s: "%(model_name, save_path))

        # print current lists of experiment values
        print "step_list:", step_list
        print "loss_list:", loss_list
        print "val_acc_list:", val_acc_list

        print "\n"

        # write experiment output to file -- flag 'a' for append
        with open(output_file, 'a') as f:
            f.write(model_name + '\n\n')
            f.write("experiment end time: " + str(datetime.now()) + '\n\n')
            f.write("configuration:\n")
            f.write(configuration + '\n')
            f.write("\nstep:\n")
            f.write(','.join([str(s) for s in step_list]))
            f.write("\nloss:\n")
            f.write(','.join([str(l) for l in loss_list]))
            f.write("\nvalidation accuracy:\n")
            f.write(','.join([str(a) for a in val_acc_list]))
            f.write("\ntest accuracy:\n")
            f.write(str(acc))
            f.write("\n\n")
            f.write("--------------------")
            f.write("\n\n")

        train_writer.close()
        validation_writer.close()

def architecture_1(model_name, configuration, output_file):
    with tf.Session() as sess:
        model_name = model_name
        model_dir = MODEL_SAVE_DIR + model_name + '/'
        log_dir = LOG_DIR + model_name + '/'

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print "PREPARING TO TRAIN:", model_name

        # first convolutional layer
        W_conv1 = weight_variable([CONV1_KERNEL_SIZE[0],CONV1_KERNEL_SIZE[1],NUM_CHANNELS,CONV1_NUM_FILTERS], 'W_conv1')
        b_conv1 = bias_variable([CONV1_NUM_FILTERS], 'b_conv1')
        tf.summary.image('W_conv1', tf.transpose(W_conv1, [3, 0, 1, 2]), max_outputs=CONV1_NUM_FILTERS)

        # reshape x to a 4d tensor
        x_image = tf.reshape(x, [-1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])

        # convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool
        h_conv1 = CONV1_ACTIV_FUNC(conv2d(x_image, W_conv1, CONV1_STRIDE, CONV1_PADDING) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1, POOL1_FILTER_SIZE, POOL1_STRIDE, POOL1_PADDING)

        ## 1 SET OF LAYERS
        p1s = h_pool1.get_shape().as_list()

        # densely connected layer
        # image size has been reduced to 10x10 so we will add a fully-connected layer with 1024 neurons
        W_fc1 = weight_variable([p1s[1]*p1s[2]*CONV1_NUM_FILTERS, FC1_NUM_NEURONS], 'W_fc1')
        b_fc1 = bias_variable([FC1_NUM_NEURONS], 'b_fc1')

        h_pool1_flat = tf.reshape(h_pool1, [-1, p1s[1]*p1s[2]*CONV1_NUM_FILTERS])
        h_fc1 = FC1_ACTIV_FUNC(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

        # dropout
        # to reduce overfitting, we apply dropout before the readout layer
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # readout layer
        W_fc2 = weight_variable([FC1_NUM_NEURONS, NUM_CLASSES], 'W_fc2')
        b_fc2 = bias_variable([NUM_CLASSES], 'b_fc2')

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # setup for training
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # setup to save accuracy and loss to logs
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("cross_entropy", cross_entropy)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + 'train/', sess.graph)
        validation_writer = tf.summary.FileWriter(log_dir + 'validation', sess.graph)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        step_list = []
        loss_list = []
        val_acc_list = []
        k = 0
        for i in range(NUM_ITERS):
            j = 0
            while (j + BATCH_SIZE <= train_data.shape[0]):
                batch = [train_data[j:j+BATCH_SIZE], train_labels[j:j+BATCH_SIZE]]
                summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: KEEP_PROB})
                train_writer.add_summary(summary, k)
                j += BATCH_SIZE
                k += 1

            if i % 10 == 0:
                summary, loss, acc = sess.run([merged, cross_entropy, accuracy], feed_dict={x: validation_data, y_: validation_labels, keep_prob: 1.0})
                validation_writer.add_summary(summary, k)
                print("Step %d, validation accuracy %g"%(i, acc))
                step_list.append(i)
                loss_list.append(loss)
                val_acc_list.append(acc)
                save_path = saver.save(sess, model_dir + model_name + ".ckpt")
                print("Saved model %s at Step %d"%(model_name, i))

        acc, y_c = sess.run([accuracy, y_conv], feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})

        ##################### Save incorrectly classified images
        eq_arr = np.equal(np.argmax(y_c, axis = 1), np.argmax(test_labels, axis = 1))
        test_images_filenamesWithResult = ["%s_result%s.png" % t for t in zip(list(test_images_filenames), map(str,test_labels))]
        test_results = dict(zip(test_images_filenames, eq_arr))
        test_results_rename = dict(zip(test_images_filenames,test_images_filenamesWithResult))
        timenow = time.strftime("%d_%m_%Y-%H_%M_%S")
        test_result_path = log_dir + "/validation_result" + timenow

        if not os.path.isdir(test_result_path):
            os.makedirs(test_result_path)
        else:
            shutil.rmtree(test_result_path)
            os.makedirs(test_result_path)
        test_result_filenames = []

        for filename, result in test_results.iteritems():
            if not result:
                shutil.copy(filename, test_result_path)
                test_result_filenames.append(filename)
        ##################### (END) Save incorrectly classified images

        ##################### Create confusion matrix.txt
        pos_images, neg_images, pos_misclass_images, neg_misclass_images = 0,0,0,0
        for filename in test_images_filenames:
            if filename.find('pos') > 0:
                pos_images += 1
            else:
                neg_images += 1
        for filename in test_result_filenames:
            if filename.find('pos') > 0:
                pos_misclass_images += 1
            else:
                neg_misclass_images += 1
        f= open(test_result_path + "/confusion_"+timenow+".txt","w+")
        f.write("CONFUSION MATRIX for " + timenow + "\r\n")
        f.write("Ground truth|  1  |  0  |\r\n")
        f.write("Classified  |-----|-----|\r\n")
        f.write("     1      |  "+ str(pos_images-pos_misclass_images) + "  | " + str(neg_misclass_images) + "  |\r\n")
        f.write("     0      |  "+ str(pos_misclass_images) + "  | " + str(neg_images-neg_misclass_images) + "  |\r\n")
        f.write("misclassified images:\r\n")
        for filename in test_result_filenames:
            f.write(filename + "\r\n")
        print("Saved incorrectly classified image and confusion matrix to path:", test_result_path)
        f.close()
        ##################### (END) Create confusion matrix.txt

        # print("Final predictions",y_c)
        print("Final test accuracy for %s is %g"%(model_name, acc))

        save_path = saver.save(sess, model_dir + model_name + ".ckpt")
        print("Saved final %s to path %s: "%(model_name, save_path))

        # print current lists of experiment values
        print "step_list:", step_list
        print "loss_list:", loss_list
        print "val_acc_list:", val_acc_list

        print "\n"

        # write experiment output to file -- flag 'a' for append
        with open(output_file, 'a') as f:
            f.write(model_name + '\n\n')
            f.write("experiment end time: " + str(datetime.now()) + '\n\n')
            f.write("configuration:\n")
            f.write(configuration + '\n')
            f.write("\nstep:\n")
            f.write(','.join([str(s) for s in step_list]))
            f.write("\nloss:\n")
            f.write(','.join([str(l) for l in loss_list]))
            f.write("\nvalidation accuracy:\n")
            f.write(','.join([str(a) for a in val_acc_list]))
            f.write("\ntest accuracy:\n")
            f.write(str(acc))
            f.write("\n\n")
            f.write("--------------------")
            f.write("\n\n")

        train_writer.close()
        validation_writer.close()

def architecture_2(model_name, configuration, output_file):
    with tf.Session() as sess:
        model_name = model_name
        model_dir = MODEL_SAVE_DIR + model_name + '/'
        log_dir = LOG_DIR + model_name + '/'

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print "PREPARING TO TRAIN:", model_name

        # first convolutional layer
        W_conv1 = weight_variable([CONV1_KERNEL_SIZE[0],CONV1_KERNEL_SIZE[1],NUM_CHANNELS,CONV1_NUM_FILTERS], 'W_conv1')
        b_conv1 = bias_variable([CONV1_NUM_FILTERS], 'b_conv1')
        tf.summary.image('W_conv1', tf.transpose(W_conv1, [3, 0, 1, 2]), max_outputs=CONV1_NUM_FILTERS)

        # reshape x to a 4d tensor
        x_image = tf.reshape(x, [-1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])

        # convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool
        h_conv1 = CONV1_ACTIV_FUNC(conv2d(x_image, W_conv1, CONV1_STRIDE, CONV1_PADDING) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1, POOL1_FILTER_SIZE, POOL1_STRIDE, POOL1_PADDING)

        # second convolutional layer
        W_conv2 = weight_variable([CONV2_KERNEL_SIZE[0],CONV2_KERNEL_SIZE[1],CONV1_NUM_FILTERS,CONV2_NUM_FILTERS], 'W_conv2')
        b_conv2 = bias_variable([CONV2_NUM_FILTERS], 'b_conv2')

        tf.summary.image('W_conv2', tf.transpose(W_conv2[:, :, 0:1, :], [3, 0, 1, 2]), max_outputs=CONV2_NUM_FILTERS)

        # convolve the result of h_pool1 with the weight tensor, add the bias, apply the ReLU function, and finally max pool
        h_conv2 = CONV2_ACTIV_FUNC(conv2d(h_pool1, W_conv2, CONV2_STRIDE, CONV2_PADDING) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2, POOL2_FILTER_SIZE, POOL2_STRIDE, POOL2_PADDING)

        p2s = h_pool2.get_shape().as_list()

        # densely connected layer
        # image size has been reduced to 10x10 so we will add a fully-connected layer with 1024 neurons
        W_fc1 = weight_variable([p2s[1]*p2s[2]*CONV2_NUM_FILTERS, FC1_NUM_NEURONS], 'W_fc1')
        b_fc1 = bias_variable([FC1_NUM_NEURONS], 'b_fc1')

        h_pool2_flat = tf.reshape(h_pool2, [-1, p2s[1]*p2s[2]*CONV2_NUM_FILTERS])



        # second convolutional layer
        W_conv3 = weight_variable([CONV3_KERNEL_SIZE[0],CONV3_KERNEL_SIZE[1],CONV2_NUM_FILTERS,CONV3_NUM_FILTERS], 'W_conv3')
        b_conv3 = bias_variable([CONV3_NUM_FILTERS], 'b_conv3')

        tf.summary.image('W_conv3', tf.transpose(W_conv3[:, :, 0:1, :], [3, 0, 1, 2]), max_outputs=CONV3_NUM_FILTERS)

        # convolve the result of h_pool1 with the weight tensor, add the bias, apply the ReLU function, and finally max pool
        h_conv3 = CONV2_ACTIV_FUNC(conv2d(h_pool2, W_conv3, CONV3_STRIDE, CONV3_PADDING) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3, POOL3_FILTER_SIZE, POOL3_STRIDE, POOL3_PADDING)

        p3s = h_pool3.get_shape().as_list()

        # densely connected layer
        W_fc1 = weight_variable([p3s[1]*p3s[2]*CONV3_NUM_FILTERS, FC1_NUM_NEURONS], 'W_fc1')
        b_fc1 = bias_variable([FC1_NUM_NEURONS], 'b_fc1')

        h_pool3_flat = tf.reshape(h_pool3, [-1, p3s[1]*p3s[2]*CONV3_NUM_FILTERS])
        h_fc1 = FC1_ACTIV_FUNC(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

        # dropout
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # readout layer
        W_fc2 = weight_variable([FC1_NUM_NEURONS, NUM_CLASSES], 'W_fc2')
        b_fc2 = bias_variable([NUM_CLASSES], 'b_fc2')

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # setup for training
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # setup to save accuracy and loss to logs
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("cross_entropy", cross_entropy)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + 'train/', sess.graph)
        validation_writer = tf.summary.FileWriter(log_dir + 'validation', sess.graph)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        step_list = []
        loss_list = []
        val_acc_list = []
        k = 0
        for i in range(NUM_ITERS):
            j = 0
            while (j + BATCH_SIZE <= train_data.shape[0]):
                batch = [train_data[j:j+BATCH_SIZE], train_labels[j:j+BATCH_SIZE]]
                summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: KEEP_PROB})
                train_writer.add_summary(summary, k)
                j += BATCH_SIZE
                k += 1

            if i % 10 == 0:
                summary, loss, acc = sess.run([merged, cross_entropy, accuracy], feed_dict={x: validation_data, y_: validation_labels, keep_prob: 1.0})
                validation_writer.add_summary(summary, k)
                print("Step %d, validation accuracy %g"%(i, acc))
                step_list.append(i)
                loss_list.append(loss)
                val_acc_list.append(acc)
                save_path = saver.save(sess, model_dir + model_name + ".ckpt")
                print("Saved model %s at Step %d"%(model_name, i))

        acc, y_c = sess.run([accuracy, y_conv], feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})

        ##################### Save incorrectly classified images
        eq_arr = np.equal(np.argmax(y_c, axis = 1), np.argmax(test_labels, axis = 1))
        test_images_filenamesWithResult = ["%s_result%s.png" % t for t in zip(list(test_images_filenames), map(str,test_labels))]
        test_results = dict(zip(test_images_filenames, eq_arr))
        test_results_rename = dict(zip(test_images_filenames,test_images_filenamesWithResult))
        timenow = time.strftime("%d_%m_%Y-%H_%M_%S")
        test_result_path = log_dir + "/validation_result" + timenow

        if not os.path.isdir(test_result_path):
            os.makedirs(test_result_path)
        else:
            shutil.rmtree(test_result_path)
            os.makedirs(test_result_path)
        test_result_filenames = []

        for filename, result in test_results.iteritems():
            if not result:
                shutil.copy(filename, test_result_path)
                test_result_filenames.append(filename)
        ##################### (END) Save incorrectly classified images

        ##################### Create confusion matrix.txt
        pos_images, neg_images, pos_misclass_images, neg_misclass_images = 0,0,0,0
        for filename in test_images_filenames:
            if filename.find('pos') > 0:
                pos_images += 1
            else:
                neg_images += 1
        for filename in test_result_filenames:
            if filename.find('pos') > 0:
                pos_misclass_images += 1
            else:
                neg_misclass_images += 1
        f= open(test_result_path + "/confusion_"+timenow+".txt","w+")
        f.write("CONFUSION MATRIX for " + timenow + "\r\n")
        f.write("Ground truth|  1  |  0  |\r\n")
        f.write("Classified  |-----|-----|\r\n")
        f.write("     1      |  "+ str(pos_images-pos_misclass_images) + "  | " + str(neg_misclass_images) + "  |\r\n")
        f.write("     0      |  "+ str(pos_misclass_images) + "  | " + str(neg_images-neg_misclass_images) + "  |\r\n")
        f.write("misclassified images:\r\n")
        for filename in test_result_filenames:
            f.write(filename + "\r\n")
        print("Saved incorrectly classified image and confusion matrix to path:", test_result_path)
        f.close()
        ##################### (END) Create confusion matrix.txt

        # print("Final predictions",y_c)
        print("Final test accuracy for %s is %g"%(model_name, acc))

        save_path = saver.save(sess, model_dir + model_name + ".ckpt")
        print("Saved final %s to path %s: "%(model_name, save_path))

        # print current lists of experiment values
        print "step_list:", step_list
        print "loss_list:", loss_list
        print "val_acc_list:", val_acc_list

        print "\n"

        # write experiment output to file -- flag 'a' for append
        with open(output_file, 'a') as f:
            f.write(model_name + '\n\n')
            f.write("experiment end time: " + str(datetime.now()) + '\n\n')
            f.write("configuration:\n")
            f.write(configuration + '\n')
            f.write("\nstep:\n")
            f.write(','.join([str(s) for s in step_list]))
            f.write("\nloss:\n")
            f.write(','.join([str(l) for l in loss_list]))
            f.write("\nvalidation accuracy:\n")
            f.write(','.join([str(a) for a in val_acc_list]))
            f.write("\ntest accuracy:\n")
            f.write(str(acc))
            f.write("\n\n")
            f.write("--------------------")
            f.write("\n\n")

        train_writer.close()
        validation_writer.close()
