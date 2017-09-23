**Using a Convolutional Neural Network to Predict the Presence of Lung Cancer**
==============
#### *Project for EECS349: Machine Learning, Northwestern University*
#### *Project Members: Adam Pollack, Chainatee Tanakulrungson, Nate Kaiser*

### Instructions for Use:
To use this package, finish this sentence.

### File descriptions:
**config.py**:

**convert_dcm_to_jpg.sh**: a short script for extracting and separating the image and metadata from DCM (medical image) files

**create_samples.py**: crawls the specified directory, for each image generates a 40x40 pixel snippet of all cancer nodules (given the nodules' (x, y, z) positions, and generates an equal number of samples at random (x, y, z) positions after verifying the resulting 40x40 image is A) within the bounds of the lungs and B) a given radius away from any cancer nodules

**extract_dcm_metadata.sh**: a short shell script for extracting metadata from DCM (medical image) files

**import_data.py**:

**interactive_plot.py**: creates two interactive plots of all experimental data (accuracy vs epoch & loss vs epoch)

**main.py**:

**model.py**:

**postprocess.py**: extracts data from custom experiment output format for use by other scripts

**preprocess.py**: crawls the specified directory and verifies each image is valid and has a corresponding data entry in the groundtruth table

**view_lungs.py**: loops through images and shows output of lung mask operation (used for manual verification)

**view_nodules.py**: loops through images and shows output of cancer nodule mask operation (used for manual verification)
