# dogs-vs-cats
The image classification with Convolutional Neural Network build with MXnet

# Prerequisites
Following packages must be installed. We will consider installation on macOS Sierra.

1. Anaconda for Python 3.6 - install from here https://www.continuum.io/downloads
2. R within Anaconda - installation instructions here: https://www.continuum.io/conda-for-r
3. MXnet for R - installation instructions here: http://mxnet.io/get_started/osx_setup.html
	* make sure to have GFortran installed - install from here https://gcc.gnu.org/wiki/GFortranBinaries#MacOS
	* make sure to have libxml2 installed - (brew install libxml2) or (port install libxml2) 

# Prepare Datasets
Our goal is to generate two files, imgdata_train.rec for training and imgdata_val.rec for validation, and the former contains 95% images.
This can be done with following steps:
1. Make sure that images belonging to the same class are placed in the same directory:
	* cats - into directory named `cat`
	* dogs - into directory named `dog`
2. All these class directories are then in the same root directory `full_train_data`.
3. Then prepare two `.lst` files, which consist of the labels and image paths can be used for generating rec files.
```
python tools/im2rec.py --list True --recursive True --train-ratio 0.95 imgdata full_train_data
```
4. Then generate the `.rec files`. We resize the images such that the short edge is at least 128px and save them with 95/100 quality. We also use 16 threads to accelerate the packing.
```
python tools/im2rec.py --resize 480 --quality 95 --num-thread 16 imgdata full_train_data
```
