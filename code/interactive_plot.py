#!/usr/bin/env python
''' creates interactive plot of tensorflow output '''

import numpy as np
import matplotlib.pyplot as plt
from postprocess import postprocess


def interactive_plot():
    # import data from post-processing script
    run_data = postprocess()

    # setup
    accuracies = []
    losses = []
    test_accuracies = []
    step = 0
    acc = 0
    loss = 0
    handle = 0

    #######################################################
    # GRAPHING VALIDATION ACCURACY AS A FUNCTION OF EPOCH #
    #######################################################
    fig, ax = plt.subplots()
    ax.set_title('Validation Accuracy vs Epoch')

    for run_num in run_data:
        # print "processing run", run_num
        step = np.asarray(run_data[run_num]['step'])
        acc = 100 * np.asarray(run_data[run_num]['validation_accuracy'])
        loss = np.asarray(run_data[run_num]['loss'])
        handle, = ax.plot(step, acc, lw=1.0, label=str(run_num))
        accuracies.append(handle)
        test_accuracies.append(run_data[run_num]['test_accuracy'])

    # shrink x axis by 20% so we can fit our huge legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

    # create legend, put to right of plot area
    leg = ax.legend(fancybox=True, shadow=True, ncol=3, loc='upper right', bbox_to_anchor=(1.2, 1.0))
    leg.get_frame().set_alpha(0.4)

    # I have no idea what this does, but I think it links legend entries to the data
    lined = dict()
    for legline, origline in zip(leg.get_lines(), accuracies):
        legline.set_picker(5)  # 5 pts tolerance
        lined[legline] = origline

    # legend click callback
    def onpick(event):
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)

        # change the alpha on the line in the legend so we can see what lines have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', onpick)
    ax.set_xlim([0, 500])
    ax.set_ylim([80, 100])
    plt.xlabel('Epoch Number')
    plt.ylabel('Validation Accuracy [%]')
    plt.show()


    ########################################
    # GRAPHING LOSS AS A FUNCTION OF EPOCH #
    ########################################
    fig, ax = plt.subplots()
    ax.set_title('Softmax Loss vs Epoch')

    for run_num in run_data:
        # print "processing run", run_num
        step = np.asarray(run_data[run_num]['step'])
        loss = np.asarray(run_data[run_num]['loss'])
        handle, = ax.plot(step, loss, lw=1.0, label=str(run_num))
        # losses[run_num] = handle
        losses.append(handle)

    # shrink x axis by 20% so we can fit our huge legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

    # create legend, put to right of plot area
    leg = ax.legend(fancybox=True, shadow=True, ncol=3, loc='upper right', bbox_to_anchor=(1.2, 1.0))
    leg.get_frame().set_alpha(0.4)

    # I have no idea what this does, but I think it links legend entries to the data
    lined = dict()
    for legline, origline in zip(leg.get_lines(), losses):
        legline.set_picker(5)  # 5 pts tolerance
        lined[legline] = origline

    # legend click callback
    def onpick2(event):
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)

        # change the alpha on the line in the legend so we can see what lines have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', onpick2)
    ax.set_xlim([0, 500])
    ax.set_ylim([0.0, 0.5])
    plt.xlabel('Epoch Number')
    plt.ylabel('Softmax Loss')
    plt.show()


    #######################################################
    # GRAPHING FINAL TEST ACCURACY AS A FUNCTION OF EPOCH #
    #######################################################
    plt.plot(test_accuracies)
    plt.show()


if __name__ == '__main__':
    interactive_plot()
