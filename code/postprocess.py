#!/usr/bin/env python
''' script for post-processing output text from high-dimensionality experiment loop '''

import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter


def postprocess():
    data = {}
    config = []
    with open("data/experiment_output_FINAL.txt", 'r') as f:
        for line in f:
            line = line.strip().split()
            if line:
                if line[0][0:13] == "model_config_":
                    n = int(line[0][13:])
                    # print n
                    count = 0
                    config = []

            if count == 5 or count == 6:
                config.append([int(line[2][1]), int(line[3][0])])
            elif count == 7:
                config.append(int(line[2]))
            elif count == 8:
                config.append(int(line[2]))
            elif count == 9:
                config.append(float(line[2]))
                # print "config:", config
            elif count == 12:
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
        avg = sum(data[n]['validation_accuracy'][17:30])/(30.0-17.0)
        averages.append([avg, n])
    # sorted(averages, key=lambda x: x[0])
    # print averages
    # sorted(averages, key=itemgetter(0))
    averages.sort(key=lambda x: x[0])
    for a in averages:
        print a

    # print "data:\n", data
    return data


if __name__ == '__main__':
    postprocess()
