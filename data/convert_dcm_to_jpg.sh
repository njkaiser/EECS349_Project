#!/bin/bash
cd
list=`find /home/njk/Desktop/1.3.6.1.4.1.14519.5.2.1.7311.5101.158323547117540061132729905711/ -type d`
for directory in $list; do
cd $directory
mogrify -format jpg *.dcm
cd
done
