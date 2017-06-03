#!/usr/bin/env python
''' creates interactive plot of tensorflow output lists '''

import numpy as np
import matplotlib.pyplot as plt
from postprocess import postprocess

# NUM_GRAPHS = 108

# TO CREATE GRAPHS STRUCTURE:
# for i in xrange(NUM_GRAPHS):
#     print "g"+ str(i) + "= 0;",
# print
# assert False

# TO CREATE INITIALIZATION LIST:
# print "accuracies = [",
# for i in xrange(NUM_GRAPHS):
#     print "g" + str(i) + ",",
# print "]"
# assert False

def interactive_plot():
    run_data = postprocess()

    # this is necessary because what I'm trying to do is hacky
    # g0= 0; g1= 0; g2= 0; g3= 0; g4= 0; g5= 0; g6= 0; g7= 0; g8= 0; g9= 0; g10= 0; g11= 0; g12= 0; g13= 0; g14= 0; g15= 0; g16= 0; g17= 0; g18= 0; g19= 0; g20= 0; g21= 0; g22= 0; g23= 0; g24= 0; g25= 0; g26= 0; g27= 0; g28= 0; g29= 0; g30= 0; g31= 0; g32= 0; g33= 0; g34= 0; g35= 0; g36= 0; g37= 0; g38= 0; g39= 0; g40= 0; g41= 0; g42= 0; g43= 0; g44= 0; g45= 0; g46= 0; g47= 0; g48= 0; g49= 0; g50= 0; g51= 0; g52= 0; g53= 0; g54= 0; g55= 0; g56= 0; g57= 0; g58= 0; g59= 0; g60= 0; g61= 0; g62= 0; g63= 0; g64= 0; g65= 0; g66= 0; g67= 0; g68= 0; g69= 0; g70= 0; g71= 0; g72= 0; g73= 0; g74= 0; g75= 0; g76= 0; g77= 0; g78= 0; g79= 0; g80= 0; g81= 0; g82= 0; g83= 0; g84= 0; g85= 0; g86= 0; g87= 0; g88= 0; g89= 0; g90= 0; g91= 0; g92= 0; g93= 0; g94= 0; g95= 0; g96= 0; g97= 0; g98= 0; g99= 0; g100= 0; g101= 0; g102= 0; g103= 0; g104= 0; g105= 0; g106= 0; g107= 0;

    # see above comment, this is also necessary
    # accuracies = [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16, g17, g18, g19, g20, g21, g22, g23, g24, g25, g26, g27, g28, g29, g30, g31, g32, g33, g34, g35, g36, g37, g38, g39, g40, g41, g42, g43, g44, g45, g46, g47, g48, g49, g50, g51, g52, g53, g54, g55, g56, g57, g58, g59, g60, g61, g62, g63, g64, g65, g66, g67, g68, g69, g70, g71, g72, g73, g74, g75, g76, g77, g78, g79, g80, g81, g82, g83, g84, g85, g86, g87, g88, g89, g90, g91, g92, g93, g94, g95, g96, g97, g98, g99, g100, g101, g102, g103, g104, g105, g106, g107]
    accuracies = []
    losses = []

    # create plot object
    fig, ax = plt.subplots()
    ax.set_title('All of the data in the entire world:')

    # create plot handles for all entries in file
    step = 0
    acc = 0
    loss = 0
    handle = 0
    for run_num in run_data:
        # print "processing run", run_num
        step = np.asarray(run_data[run_num]['step'])
        acc = np.asarray(run_data[run_num]['validation_accuracy'])
        loss = np.asarray(run_data[run_num]['loss'])
        handle, = ax.plot(step, acc, lw=1.0, label=str(run_num))
        # accuracies[run_num] = handle
        accuracies.append(handle)
        # handle, = ax.plot(step, loss, lw=1.0, label=str(run_num))
        # losses[run_num] = handle
        # losses.append(handle)

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
    ax.set_ylim([0.85, 1.0])
    plt.show()


    fig, ax = plt.subplots()
    ax.set_title('All of the data in the entire world:')

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

    ax.set_ylim([0.0, 0.5])
    plt.show()


if __name__ == '__main__':
    interactive_plot()
