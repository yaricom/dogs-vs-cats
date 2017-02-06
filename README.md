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

The sample output for first four epoch runs are following:
```
python src/dogs_vs_cats_train.py --image-shape 3,32,32 --network resnet --num-layers 18 --batch-size 128 --num-examples 25000
INFO:root:start with arguments Namespace(batch_size=128, benchmark=0, data_nthreads=4, data_train='data/data_set/imgdata_train.rec', data_val='data/data_set/imgdata_val.rec', disp_batches=20, gpus=None, image_shape='3,32,32', kv_store='device', load_epoch=None, lr=0.05, lr_factor=0.1, lr_step_epochs='200,250', max_random_aspect_ratio=0, max_random_h=36, max_random_l=50, max_random_rotate_angle=0, max_random_s=50, max_random_scale=1, max_random_shear_ratio=0, min_random_scale=1, model_prefix='./out/model_dogs_vs_cats', mom=0.9, monitor=0, network='resnet', num_classes=2, num_epochs=300, num_examples=25000, num_layers=18, optimizer='sgd', pad_size=4, random_crop=1, random_mirror=1, rgb_mean='123.68,116.779,103.939', test_io=0, top_k=0, wd=0.0001)
[20:50:28] src/io/iter_image_recordio.cc:221: ImageRecordIOParser: data/data_set/imgdata_train.rec, use 1 threads for decoding..
[20:50:29] src/io/iter_image_recordio.cc:221: ImageRecordIOParser: data/data_set/imgdata_val.rec, use 1 threads for decoding..
INFO:root:Epoch[0] Batch [20]	Speed: 2.54 samples/sec	Train-accuracy=0.524182
INFO:root:Epoch[0] Batch [40]	Speed: 2.59 samples/sec	Train-accuracy=0.538281
INFO:root:Epoch[0] Batch [60]	Speed: 2.63 samples/sec	Train-accuracy=0.536328
INFO:root:Epoch[0] Batch [80]	Speed: 2.53 samples/sec	Train-accuracy=0.600000
INFO:root:Epoch[0] Batch [100]	Speed: 2.50 samples/sec	Train-accuracy=0.625391
INFO:root:Epoch[0] Batch [120]	Speed: 2.55 samples/sec	Train-accuracy=0.626172
INFO:root:Epoch[0] Batch [140]	Speed: 2.50 samples/sec	Train-accuracy=0.561328
INFO:root:Epoch[0] Batch [160]	Speed: 2.55 samples/sec	Train-accuracy=0.637500
INFO:root:Epoch[0] Batch [180]	Speed: 2.46 samples/sec	Train-accuracy=0.619141
INFO:root:Epoch[0] Train-accuracy=0.670312
INFO:root:Epoch[0] Time cost=9360.959
INFO:root:Saved checkpoint to "./out/model_dogs_vs_cats-0001.params"
INFO:root:Epoch[0] Validation-accuracy=0.706250
INFO:root:Epoch[1] Batch [20]	Speed: 2.53 samples/sec	Train-accuracy=0.593378
INFO:root:Epoch[1] Batch [40]	Speed: 2.65 samples/sec	Train-accuracy=0.657813
INFO:root:Epoch[1] Batch [60]	Speed: 2.65 samples/sec	Train-accuracy=0.663281
INFO:root:Epoch[1] Batch [80]	Speed: 2.67 samples/sec	Train-accuracy=0.674219
INFO:root:Epoch[1] Batch [100]	Speed: 2.67 samples/sec	Train-accuracy=0.688281
INFO:root:Epoch[1] Batch [120]	Speed: 2.67 samples/sec	Train-accuracy=0.699219
INFO:root:Epoch[1] Batch [140]	Speed: 2.67 samples/sec	Train-accuracy=0.717578
INFO:root:Epoch[1] Batch [160]	Speed: 2.67 samples/sec	Train-accuracy=0.692187
INFO:root:Epoch[1] Batch [180]	Speed: 2.67 samples/sec	Train-accuracy=0.712109
INFO:root:Epoch[1] Train-accuracy=0.707812
INFO:root:Epoch[1] Time cost=8958.407
INFO:root:Saved checkpoint to "./out/model_dogs_vs_cats-0002.params"
INFO:root:Epoch[1] Validation-accuracy=0.738281
INFO:root:Epoch[2] Batch [20]	Speed: 2.67 samples/sec	Train-accuracy=0.681920
INFO:root:Epoch[2] Batch [40]	Speed: 2.67 samples/sec	Train-accuracy=0.730859
INFO:root:Epoch[2] Batch [60]	Speed: 2.66 samples/sec	Train-accuracy=0.739453
INFO:root:Epoch[2] Batch [80]	Speed: 2.67 samples/sec	Train-accuracy=0.742188
INFO:root:Epoch[2] Batch [100]	Speed: 2.67 samples/sec	Train-accuracy=0.745703
INFO:root:Epoch[2] Batch [120]	Speed: 2.67 samples/sec	Train-accuracy=0.739453
INFO:root:Epoch[2] Batch [140]	Speed: 2.67 samples/sec	Train-accuracy=0.735156
INFO:root:Epoch[2] Batch [160]	Speed: 2.67 samples/sec	Train-accuracy=0.755078
INFO:root:Epoch[2] Batch [180]	Speed: 2.67 samples/sec	Train-accuracy=0.758984
INFO:root:Epoch[2] Train-accuracy=0.767578
INFO:root:Epoch[2] Time cost=8847.880
INFO:root:Saved checkpoint to "./out/model_dogs_vs_cats-0003.params"
INFO:root:Epoch[2] Validation-accuracy=0.592969
INFO:root:Epoch[3] Batch [20]	Speed: 2.67 samples/sec	Train-accuracy=0.766369
INFO:root:Epoch[3] Batch [40]	Speed: 2.67 samples/sec	Train-accuracy=0.780859
INFO:root:Epoch[3] Batch [60]	Speed: 2.67 samples/sec	Train-accuracy=0.756641
INFO:root:Epoch[3] Batch [80]	Speed: 2.67 samples/sec	Train-accuracy=0.772656
INFO:root:Epoch[3] Batch [100]	Speed: 2.67 samples/sec	Train-accuracy=0.774219
INFO:root:Epoch[3] Batch [120]	Speed: 2.67 samples/sec	Train-accuracy=0.757812
INFO:root:Epoch[3] Batch [140]	Speed: 2.67 samples/sec	Train-accuracy=0.776953
INFO:root:Epoch[3] Batch [160]	Speed: 2.67 samples/sec	Train-accuracy=0.780078
INFO:root:Epoch[3] Batch [180]	Speed: 2.66 samples/sec	Train-accuracy=0.796875
INFO:root:Epoch[3] Train-accuracy=0.782813
INFO:root:Epoch[3] Time cost=8901.720
INFO:root:Saved checkpoint to "./out/model_dogs_vs_cats-0004.params"
INFO:root:Epoch[3] Validation-accuracy=0.823785
```

As it can be seen validation accuracy increase, but prediction model should converge in order to gurantee that our CNN is learning something. Unfortunatelly due to hardware limitations I was unable to test over all 300 epochs in order to check if model converge, but interested parties may try this with better hardware setup. With my hardware setup (CPU only, 2.5 GHz Intel Core i3, 8Gb RAM, SSD) training over 300 epochs would take about 740 hours (or one month). I believe that within CUDA-enabled setup it will take rather hours. 

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
python src/dogs_vs_cats_predict.py --image-shape 3,32,32 --rec-prefix imgdata_test --model out/model_dogs_vs_cats --model-checkpoint 4 out/results data/data_set_test
```
The predictions output will be stored as coma separated file into `out/results` directory.


[1]: http://mxnet.io
[2]: http://deeplearning.net/tutorial/lenet.html
