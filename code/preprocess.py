#!/usr/bin/env python
import os
import dicom as pydicom
import csv
import time

import pylab # used to visualize pixel data if desired


csvfilename1 = '/home/njk/Courses/EECS349/Project/data/DSB2017/stage1_groundtruth.csv'
csvfilename2 = '/home/njk/Courses/EECS349/Project/data/DSB2017/stage1_solution.csv'
datadirectory = '/home/njk/Courses/EECS349/Project/data/DSB2017/'

print "extracting groundtruth data..."
start = time.time()

# import groundtruth data from CSV file #1:
groundtruth = {}
with open(csvfilename1) as f:
    csvfile = csv.reader(f)
    for line in csvfile:
        groundtruth[line[0]] = line[1]

# import groundtruth data from CSV file #2:
with open(csvfilename2) as f:
    csvfile = csv.reader(f)
    for line in csvfile:
        if line[0] in groundtruth:
            if groundtruth[line[0]] == line[1]:
                continue
            else:
                print "ERROR: CONFLICTING GROUNDTRUTH ENTRIES FOR PATIENT", line[0]
        else:
            groundtruth[line[0]] = line[1]

# for key, value in groundtruth.iteritems():
#     print key, ":", value

end = time.time()
print "...done - elapsed time:", end - start

print "groundtruth data contains", len(groundtruth), "entries"

print "checking image data..."
start = time.time()

filecount = 0
for dirpath, dirnames, filenames in os.walk(datadirectory):
    for filename in filenames:
        if ".dcm" in filename:
            # print dirpath + "/" + filename
            ds = pydicom.read_file(dirpath + "/" + filename)
            name = ds.PatientName

            # pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
            # pylab.show()
            if not name in groundtruth:
                print "ERROR: NAME", name, "DOES NOT EXIST IN GROUNDTRUTH DATA"

            # try:
            #     print "name:", name, "groundtruth:", groundtruth[name]
            #     pass
            # except KeyError:
            #     # print "SOMETHING"
            # print ds

            filecount += 1
            if not filecount % 1000:
                print "iteration", filecount
            # if filecount == 1:
            #     quit()

end = time.time()
print "...done - elapsed time:", end - start

print "Total DCM file count:", filecount
