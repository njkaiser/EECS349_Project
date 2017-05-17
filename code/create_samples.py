#!/usr/bin/env python
# from __future__ import print_function, division
import os
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
from tqdm import tqdm # long waits are more fun with status bars!

from matplotlib import pyplot as plt
from matplotlib import animation
# plt.ion()

from skimage import morphology
from skimage import measure
# from sklearn.cluster import KMeans
from skimage.transform import resize

from pprint import pprint
import time

# directories:
workspace = "/home/njk/Desktop/for_the_last_time/"
image_dir = workspace + "subset0/"
output_dir = workspace + "output/"
image_list = glob(image_dir + "*.mhd")
# pprint(image_list)


class Nodule(object):
    def __init__(self, x, y, z, r):
        self.x = x
        self.y = y
        self.z = z
        self.r = r

    def __repr__(self):
        return "<x: " + str(self.x) + ", y: " + str(self.y) + ", z: " + str(self.z) + ", r: " + str(self.r) + ">"

    def __str__(self):
        return "<x: " + str(self.x) + ", y: " + str(self.y) + ", z: " + str(self.z) + ", r: " + str(self.r) + ">"


patient_data = dict()
with open(workspace + "annotations.csv", mode='r') as labelfile:
    reader = csv.reader(labelfile)
    for uid, x, y, z, d in reader:
        if uid == "seriesuid":
            continue # skip header row
        elif uid not in patient_data:
            patient_data[uid] = [] # create empty list first time
        patient_data[uid].append(Nodule(float(x), float(y), float(z), float(d)/2))

# pprint(patient_data)
# print patient_data["1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860"]



for fname in image_list:
    uid = os.path.splitext(os.path.basename(fname))[0]
    if uid not in patient_data:
        print "no data available for patient", uid
        continue
    else:
        print patient_data[uid]
        sitk_slices = sitk.ReadImage(fname)
        origin = np.array(sitk_slices.GetOrigin())   # x, y, z of origin in world coordinates (mm)
        spacing = np.array(sitk_slices.GetSpacing()) # spacing of voxels in world coordinates (mm)
        slices = sitk.GetArrayFromImage(sitk_slices) # convert to numpy array
        z_slice_count, height, width = slices.shape  # height x width constitute the transverse plane
        print "img_array.shape:", slices.shape

# BEGIN ANIMATION
        # fig = plt.figure() # make figure
        # img = plt.imshow(slices[0], cmap='gray')#, vmin=0, vmax=255)
        #
        # def updatefig(iii): # callback function for FuncAnimation()
        #     img.set_array(slices[iii])
        #     return [img]
        #
        # animation.FuncAnimation(fig, updatefig, frames=range(z_slice_count), interval=1, blit=True, repeat=False)
        # fig.show()
# END ANIMATION

        for nodule in patient_data[uid]:
            print nodule


        # for j in range(len(x)):
        # for i, slc in enumerate(slices):
        #     print i
        #     img = plt.imshow(slc, cmap='gray')#, vmin=0, vmax=255)
        #     # fig.draw()
        #     fig.canvas.draw_idle()
        #     time.sleep(0.02)
        # fig.close()

        # for z in tqdm(range(z_slice_count)):
#         for slc in slices:
#             artists = func()
#             im = plt.imshow(slices[z, :, :], animated=True)
#             fig.canvas.draw_idle()
#             plt.pause(interval)
#
# for d in frames:
#    artists = func(d, *fargs)


        # Animation function
        # def animate(z):
        #     t = slices[z,:,:]
        #     img = plt.contourf(X, Y, t)
        #     return img
        #
        # # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        # anim = animation.FuncAnimation(fig, animate, frames=range(z_slice_count))
        #
        # # plt.imshow(slices[z, :, :], cmap='gray')
        # plt.show()



if __name__ == '__main__':
    # main()
    pass
