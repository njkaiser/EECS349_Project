#!/usr/bin/env python

import os
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
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
workspace = "/home/njk/Courses/EECS349/Project/data/LUNA2016/"
image_dir = workspace + "subset0/"
output_dir = workspace + "output/"
image_list = glob(image_dir + "*.mhd")
# pprint(image_list)


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


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


def generate_positives(uid, slices, v_center, radius, spacing):
    sz = 40
    output = np.zeros([sz, sz]) # create sz x sz pixel output image
    vz = v_center[2]
    vx_min = v_center[0]-sz/2
    vx_max = v_center[0]+sz/2-1
    vy_min = v_center[1]-sz/2
    vy_max = v_center[1]+sz/2+1

    n = int(radius/spacing[2]/2) # divide by 2 to ensure we get a slice of nodule
    # print "number of slices:", 2*n+1
    for slc in tqdm(slices[vz-n:vz+n, :, :]):
        # remember, numpy indices are (z, y, x) order
        # print "taking slice:", vz, str(vy_min) + ":" + str(vy_max), str(vx_min) + ":" +  str(vx_max)
        output = slc[vy_min:vy_max, vx_min:vx_max]
        plt.imshow(output, cmap='gray')
        plt.show()


def main():
    # load patient data from CSV file
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

    # loop through files and create pos/neg train/test/validation samples:
    for fname in image_list:
        uid = os.path.splitext(os.path.basename(fname))[0]
        if uid not in patient_data:
            print color.RED + "patient " + uid + ": NO DATA AVAILABLE" + color.END
            continue
        else:
            print color.BOLD + color.BLUE + "patient " + uid + color.END
            sitk_slices = sitk.ReadImage(fname)
            origin = np.array(sitk_slices.GetOrigin())   # x, y, z of origin (mm)
            spacing = np.array(sitk_slices.GetSpacing()) # spacing of voxels (mm)
            slices = sitk.GetArrayFromImage(sitk_slices) # convert sitk image to numpy array
            num_slices, H, W = slices.shape # height x width in transverse plane
            # print "img_array.shape:", slices.shape
            # print "\tspacing:", spacing

            ##### BEGIN ANIMATION
            # fig = plt.figure() # make figure
            # img = plt.imshow(slices[0], cmap='gray')#, vmin=0, vmax=255)
            #
            # def updatefig(iii): # callback function for FuncAnimation()
            #     img.set_array(slices[iii])
            #     return [img]
            #
            # animation.FuncAnimation(fig, updatefig, frames=range(num_slices), interval=1, blit=True, repeat=False)
            # fig.show()
            ##### END ANIMATION

            for nodule in patient_data[uid]:
                # print nodule
                m_center = np.array([nodule.x, nodule.y, nodule.z]) # location in mm
                v_center = np.rint((m_center-origin)/spacing).astype(int) # location in voxels
                # print "\tm_center:", m_center
                # print "\tv_center:", v_center
                generate_positives(uid, slices, v_center, nodule.r, spacing)



if __name__ == '__main__':
    main()
