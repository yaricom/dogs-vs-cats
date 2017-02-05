# dogs-vs-cats
The image classification with Convolutional Neural Network build with MXnet

## Prerequisites

We will install mxnet from source code.
```
# Clone mxnet repository. In terminal, run the commands WITHOUT "sudo"
git clone https://github.com/dmlc/mxnet.git ~/mxnet --recursive

# If building with GPU, add configurations to config.mk file:
cd ~/mxnet
cp make/config.mk .
echo "USE_CUDA=1" >>config.mk
echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk
echo "USE_CUDNN=1" >>config.mk

# Install MXNet for Python with all required dependencies
cd ~/mxnet/setup-utils
bash install-mxnet-osx-python.sh
```


## Prepare Datasets
Our goal is to generate two files, imgdata_train.rec for training and imgdata_val.rec for validation, and the former contains 95% images.
This can be done with following steps:

Make sure that images belonging to the same class are placed in the same directory:
	* cats - into directory named `data/full_train_data/cat`
	* dogs - into directory named `data/full_train_data/dog`
Make sure that all these class directories are in the same root directory `full_train_data`.

Make sure to create output directory first
```
mkdir data/data_set
```

Then prepare two `.lst` files, which consist of the labels and image paths can be used for generating rec files.
```
python tools/im2rec.py --list True --recursive True --train-ratio 0.95 data/data_set/imgdata data/full_train_data
```
As result two files with image lists will be generated:

* `data/data_set/imgdata_train.lst` - train data list
* `data/data_set/imgdata_val.lst` - validation data list

The class labels will be generated: 0 - cat, 1 - dog

Then generate the `.rec files`. We resize the images such that the short edge is at least 128px and save them with 95/100 quality. We also use 16 threads to accelerate the packing.
```
python tools/im2rec.py --resize 32 --quality 95 --num-thread 16 data/data_set/imgdata data/full_train_data
```
The resulting records will be generated in directory: `data/data_set/`


## Run training
To initiate training run
```
python src/dogs_vs_cats_train.py --data-train "data/data_set/imgdata_train.rec" --data-val "data/data_set/imgdata_val.rec"  --image-shape 3,32,32 --network resnet --num-layers 18 --batch-size 128 --num-examples 25000
```
