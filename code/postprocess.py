#!/usr/bin/env python
''' script for post-processing output text from high-dimensionality experiment loop '''

import numpy as np
import matplotlib.pyplot as plt


def postprocess():
    data = []
    config = []
    with open("practice_experiment_output.txt", 'r') as f:
        for line in f:
        # while 1:
            line = line.split()
            # print line
            if line:
                if line[0][0:13] == "model_config_":
                    n = int(line[0][13:])
                    print n
                    count = 0
                    config = []

            if count == 5:
                config.append(int(line[1]))
                print config
            elif count == 6 or count == 7:
                config.append([int(line[1][1]), int(line[2][0])])
                print config
            elif count == 8:
                config.append(float(line[1]))
                print config
            elif count == 9:
                config.append(int(line[1]))
                print config

            # print config

            count += 1

                # assert False
                # data.append({'config': 0, 'epoch': 0, 'loss': 0, 'validation_accuracy': 0, 'test_accuracy': 0})




if __name__ == '__main__':
    postprocess()
