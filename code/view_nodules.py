#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from glob import glob

working_path = "/home/njk/Courses/EECS349/Project/data/LUNA2016/output/"
image_list = glob(working_path + "images_*.npy")

for img_name in image_list:
    img = np.load(img_name)
    nodulemask = np.load(img_name.replace("images", "masks"))
    for i in range(len(img)):
        print "image %d" % i
        fig, ax = plt.subplots(2, 2, figsize=[8, 8])
        ax[0, 0].imshow(img[i], cmap='gray')
        ax[0, 1].imshow(nodulemask[i], cmap='gray')
        ax[1, 0].imshow(img[i] * nodulemask[i], cmap='gray')
        plt.show()
        # raw_input("hit enter to cont : ")
