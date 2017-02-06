import os
import sys
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import data, fit
import mxnet as mx

def read_list(lst_path):
    """
    Read list of images
        Args:
            lst_path: the path to list of image files
        Returns:
            list of images as in provided file
    """
    with open(lst_path) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            yield item

def data_iterator(rec_file, lst_file, batch_size, image_shape, args):
    """
    Creates data iterator from provided files
    Args:
        rec_file: the path RecordIO file with image data
        lst_path: the path to list of image files
        batch_size: the batch size [1]
        image_shape: the image shape feed into the network
        args: the command line arguments
    Returns:
        the ImageRecordIter to be feed into model
    """
    (rank, nworker) = (0, 1)
    rgb_mean = [float(i) for i in args.rgb_mean.split(',')]
    dataiter = mx.io.ImageRecordIter(
        path_imgrec         = rec_file,
        path_imglist        = lst_file,
        label_width         = 1,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = image_shape,
        preprocess_threads  = args.data_nthreads,
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = nworker,
        part_index          = rank
    )

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Do prediction from image list and record database')
    parser.add_argument('prefix', help='prefix of prediction results files.')
    parser.add_argument('root', help='path to folder containing lst and rec files.')
    parser.add_argument('--rec-prefix', type=str, default="imgdata_test",
                        help='the prefix for list and rec file names.')
    parser.add_argument('--model', type=str, nargs=1,
                        help='the path to the saved model file.')
    parser.add_argument('--image-shape', type=str,
                        help='the image shape feed into the network, e.g. (3,224,224)')
    parser.add_argument('--rgb-mean', type=str, default='123.68,116.779,103.939',
                        help='a tuple of size 3 for the mean rgb')
    parser.add_argument('--data-nthreads', type=int, default=4,
                        help='number of threads for data decoding')

    parser.set_defaults(
        # data
        image_shape    = '3,32,32'
    )
    args = parser.parse_args()
    args.prefix = os.path.abspath(args.prefix)
    args.root = os.path.abspath(args.root)
    return args

if __name__ == '__main__':
    args = parse_args()
    # list and records files
    rec_path = os.path.join(args.root, args.rec_prefix + ".rec")
    lst_path = os.path.join(args.root, args.rec_prefix + ".lst")

    # get test data iterator
    batch_size = 1 # we want per sample predictions
    image_shape = tuple([int(l) for l in args.image_shape.split(',')])
    test_iter = data_iterator(rec_path, lst_path, batch_size, image_shape, args)
    file_list = read_list(lst_path)

    # load model
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.model, 0)
    mod = mx.mod.Module(
        symbol=sym,
        context=mx.cpu()
    )
    # The ResNet is trained with RGB images of size 32 x 32. The training data is feed by
    # the variable data. We bind the module with the input shape and specify that it is
    # only for predicting. The number 1 added before the image shape (3x32x32)
    data_shape = (batch_size,) + image_shape
    mod.bind(for_training = False, data_shapes=[('data', data_shape)])
    mod.set_params(arg_params, aux_params)

    # Run predictions
    y = mod.predict(test_iter) # will collect and return all the prediction results.
    print 'shape of predict: %s' % (y.shape,)





# data_shape = (args.batch_size,) + image_shape
# item[1]
