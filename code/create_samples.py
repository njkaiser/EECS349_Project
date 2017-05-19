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
import math
import random
random.seed(99)
from PIL import Image
from scipy.misc import imsave

# directories:
workspace = "/home/njk/Courses/EECS349/Project/data/LUNA2016/"
image_dir = workspace + "subset1/"
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


def generate_sample(uid, slices, nodule_index, v_center, radius, spacing, classification):
    sz = 40
    output = np.zeros([sz, sz]) # create sz x sz pixel output image
    vz = v_center[2]
    vx_min = v_center[0]-sz/2
    vx_max = v_center[0]+sz/2
    vy_min = v_center[1]-sz/2
    vy_max = v_center[1]+sz/2

    if classification == "pos":
        n = int(radius/spacing[2]/2) # divide by 2 -safety factor so we don't go beyond bounds of nodule
        if n < 1:
            n = 1
        vz_start = vz-n
        vz_end = vz+n
    else:
        n = 1
        vz_start = vz
        vz_end = vz+1
    print "number of slices:", 2*n-1
    for i, slc in enumerate(slices[vz_start:vz_end, :, :]):
        # remember, numpy indices are (z, y, x) order
        # print "taking slice:", vz, str(vy_min) + ":" + str(vy_max), str(vx_min) + ":" +  str(vx_max)
        output = slc[vy_min:vy_max, vx_min:vx_max]
        imsave(output_dir + uid + 'nod' + str(nodule_index) + 'slc' + str(i) + classification + '.png', output)
        # plt.imshow(output, cmap='gray')
        # plt.show()
##### END OF FUNCTION generate_sample()


def main():
    ##### load patient data from CSV file
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

    ##### loop through files and create pos/neg train/test/validation samples:
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

            ##### create POSITIVE samples:
            for i, nodule in enumerate(patient_data[uid]):
                # print nodule
                m_center = np.array([nodule.x, nodule.y, nodule.z]) # location in mm
                v_center = np.rint((m_center-origin)/spacing).astype(int) # location in voxels
                # print "\tm_center:", m_center
                # print "\tv_center:", v_center
                generate_sample(uid, slices, i, v_center, nodule.r, spacing, "pos")

            ##### create NEGATIVE samples:
            v_center = 0
            for i in range(len(patient_data[uid])):
                # discard outermost strips of image, where there's no lung:
                xmin = int(W/6)
                xmax = W-int(W/6)
                ymin = int(H/6)
                ymax = H-int(H/6)
                zmin = num_slices/8
                zmax = num_slices - num_slices/8

                # loop through randomly chosen locations until we find one away from real nodules:
                while 1:
                    # generate random voxel center:
                    x = random.randint(xmin, xmax)
                    y = random.randint(ymin, ymax)
                    z = random.randint(zmin, zmax)
                    v_center = np.array([x, y, z]) # location in voxels

                    # try again if we're too close to a nodule:
                    min_dist = 80 # pixels, but beware x/y and z are different
                    for nodule in patient_data[uid]:
                        dist = math.sqrt((x-nodule.x)**2 + (y-nodule.y)**2 + (z-nodule.z)**2)
                        # print "looping through nodules, checking distance =", dist
                        if dist < min_dist:
                            # print "too close, recalculating random center"
                            break
                    else: # hacky (but succinct) way to break out of nested loop if sample passes
                        # print "random sample passed"
                        break

                # generate the negative sample image and save:
                generate_sample(uid, slices, 1, v_center, 0, spacing, "neg")



if __name__ == '__main__':
    main()
