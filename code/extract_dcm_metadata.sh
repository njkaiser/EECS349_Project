#!/bin/bash
files=`find ~/Desktop/TEMP_PROJECT/3/DOI -type f \( -iname "*.dcm" \)`
for file in $files
do
  dcmdump $file > "${file%.*}.data"
done
