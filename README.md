# dogs-vs-cats
In this repository we will try to build image classification prediction model based on [Convolutional Neural Networks][2] architecture using [MXNet library for Deep Learning][1].
The data set consist of 25 000 pictures of dogs and cats. The provided images has different sizes.

## Prerequisites

Make sure that you have Xcode command line tools installed on your macOS. 
Run following command to install if missed:
```
sudo xcode-select --install
```

Next we will install mxnet from source code as it seems the best way to do this now.
The installation script will use Homebrew, so make sure that you have following:
```
echo 'import site; site.addsitedir("/usr/local/lib/python2.7/site-packages")' >> ~/Library/Python/2.7/lib/python/site-packages/homebrew.pth
```
Next download MXNet source code and run installation script
```
# Clone mxnet repository. In terminal, run the commands WITHOUT "sudo"
git clone https://github.com/dmlc/mxnet.git ~/mxnet --recursive

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

Make sure to create output directories first
```
mkdir data/data_set_train
mkdir data/data_set_test
```

Then prepare two `.lst` files, which consist of the labels and image paths can be used for generating rec files.
```
python tools/im2rec.py --list True --recursive True --train-ratio 0.95 data/data_set_train/imgdata data/full_train_data
```
As result two files with image lists will be generated:

* `data/data_set_train/imgdata_train.lst` - train data list
* `data/data_set_train/imgdata_val.lst` - validation data list

The class labels will be generated: 0 - cat, 1 - dog

Then generate the `.rec files`. We resize the images such that the short edge is at least 32px and save them with 95/100 quality. We also use 16 threads to accelerate the packing.
```
python tools/im2rec.py --resize 32 --quality 95 --num-thread 16 data/data_set_train/imgdata data/full_train_data
```
The resulting records will be generated in directory: `data/data_set/`


## Run training
To initiate training run
```
python src/dogs_vs_cats_train.py --data-train "data/data_set_train/imgdata_train.rec" --data-val "data/data_set_train/imgdata_val.rec"  --image-shape 3,32,32 --network resnet --num-layers 18 --batch-size 128 --num-examples 25000
```

## Run prediction over trained model
After model training complete prepare test data set by running following command.

First create list of images in the data set:
```
python tools/im2rec.py --list True --recursive False --shuffle False  --train-ratio 0.0 --test-ratio 1.0 data/data_set_test/imgdata data/test
```

Then run records generation
```
python tools/im2rec.py --resize 32 --quality 95 --num-thread 16 --recursive False --shuffle False  --train-ratio 0.0 --test-ratio 1.0 data/data_set_test/imgdata data/test
```

Finally run prediction over test data set and trained model
```
python src/dogs_vs_cats_predict.py --image-shape 3,32,32 --rec-prefix imgdata_test --model out/model_dogs_vs_cats out/results data/data_set_test
```
The predictions output will be stored as coma separated file into `out/results` directory.


[1]: http://mxnet.io
[2]: http://deeplearning.net/tutorial/lenet.html
