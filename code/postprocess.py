#!/usr/bin/env python
''' script for post-processing output text from high-dimensionality experiment loop '''

import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter


def postprocess():
    data = {}
    config = []
    with open("data/new_experiment_output.txt", 'r') as f:
        for line in f:
            line = line.strip().split()
            if line:
                if line[0][0:13] == "model_config_":
                    n = int(line[0][13:])
                    # print n
                    count = 0
                    config = []

            # UNCOMMENT FOR 108 DATA
            # if count == 5 or count == 6:
            #     config.append([int(line[2][1]), int(line[3][0])])
            # elif count == 7:
            #     config.append(int(line[2]))
            # elif count == 8:
            #     config.append(int(line[2]))
            # elif count == 9:
            #     config.append(float(line[2]))
                # print "config:", config
            if count == 12:
                step = [int(x) for x in line[0].split(',')]
                # print "step:", step
            elif count == 14:
                loss = [float(x) for x in line[0].split(',')]
                # print "loss:", loss
            elif count == 16:
                validation_accuracy = [float(x) for x in line[0].split(',')]
                # print "validation_accuracy:", validation_accuracy
            elif count == 18:
                try:
                    test_accuracy = float(line[0])
                except ValueError:
                    test_accuracy = 0.0
                # print test_accuracy

            elif count == 19:
                # we're past all data for this run, now we store it
                data[n] = {"config": config, "step": step, "loss": loss, "validation_accuracy": validation_accuracy, "test_accuracy": test_accuracy}

            count += 1 # increment line count

    # loop through data and determine best runs
    top_N = 10 # number of best runs we want to see
    best_indices = []
    averages = []
    for n in data:
        # print n
        avg = sum(data[n]['validation_accuracy'][21:30])/(30.0-21.0)
        averages.append([avg, n])
    averages.sort(key=lambda x: x[0])

    for a in averages:
        print a

    averages = list(reversed(averages))

    # fig, ax = plt.subplots()
    # for model in averages[0:5]:
    #     ax.plot(data[model[1]]['step'], data[model[1]]['validation_accuracy'], lw=1.0, label=('model ' + str(model[1])))
    # ax.set_title('Validation Accuracy vs Epoch')
    # ax.set_xlim([0, 500])
    # ax.set_ylim([0.8, 1])
    # ax.legend(loc='lower right')
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Validation Accuracy [%]')
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # for model in averages[0:5]:
    #     ax.plot(data[model[1]]['step'], data[model[1]]['loss'], lw=1.0, label=('model ' + str(model[1])))
    # ax.set_title('Validation Accuracy vs Epoch')
    # ax.set_xlim([0, 500])
    # ax.set_ylim([0, 0.5])
    # ax.legend(loc='upper right')
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Validation Accuracy [%]')
    # plt.show()

    # print "data:\n", data
    return data


if __name__ == '__main__':
    postprocess()
