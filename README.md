**Using a Convolutional Neural Network to Predict the Presence of Lung Cancer**
==============
#### *Project for EECS349: Machine Learning, Northwestern University*
#### *Project Members: Adam Pollack, Chainatee Tanakulrungson, Nate Kaiser*

### Instructions for Use:
Install Tensorflow:
`pip install tensorflow` or `pip install tensorflow-gpu`

Run main.py from the base directory
`python code/main.py`

### File descriptions:
**config.py**: define the constants you wish to use with your model architecture in this file

**convert_dcm_to_jpg.sh**: a short script for extracting and separating the image and metadata from DCM (medical image) files

**create_samples.py**: crawls the specified directory, for each image generates a 40x40 pixel snippet of all cancer nodules (given the nodules' (x, y, z) positions, and generates an equal number of samples at random (x, y, z) positions after verifying the resulting 40x40 image is A) within the bounds of the lungs and B) a given radius away from any cancer nodules

**extract_dcm_metadata.sh**: a short shell script for extracting metadata from DCM (medical image) files

**import_data.py**: convert the data from .png to lists in Python which can be used for training

**interactive_plot.py**: creates two interactive plots of all experimental data (accuracy vs epoch & loss vs epoch)

**main.py**: contains code to either iterate over a list of different configurations (defined at the top of the script) or call one of the model architectures in model.py directly (doing this will use the constants defined in config.py)

**model.py**: defines various models which can be used by main.py. the train\_model\_loop function can be used to iterate over a list of model configurations as defined in main.py. the other functions (such as architecture\_6) define individual models which can be trained from main.py using the values defined in config.py 

**postprocess.py**: extracts data from custom experiment output format for use by other scripts

**preprocess.py**: crawls the specified directory and verifies each image is valid and has a corresponding data entry in the groundtruth table

**view_lungs.py**: loops through images and shows output of lung mask operation (used for manual verification)

**view_nodules.py**: loops through images and shows output of cancer nodule mask operation (used for manual verification)
