#!/usr/bin/env python
''' script for post-processing output text from high-dimensionality experiment loop '''

import numpy as np
import matplotlib.pyplot as plt


def postprocess():
    data = {}
    config = []
    with open("experiment_output_post_Hanlon.txt", 'r') as f:
        for line in f:
            line = line.strip().split()
            if line:
                if line[0][0:13] == "model_config_":
                    n = int(line[0][13:])
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
                # we're past all data for this run, now we store it
                data[n] = {"config": config, "step": step, "loss": loss, "validation_accuracy": validation_accuracy}

            count += 1 # increment line count

    # print "data:\n", data
    return data


if __name__ == '__main__':
    postprocess()
