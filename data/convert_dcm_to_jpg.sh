#!/bin/bash
cd
list=`find /home/njk/Courses/EECS349/Project/data/train/DOI/ -type d`
for directory in $list; do
cd $directory
mogrify -format jpg *.dcm
cd
done
