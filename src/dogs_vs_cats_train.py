import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import data, fit
import mxnet as mx
#from score import score

def validate_model(model, val_iter):
    """
    The model validation against validation data records
        Args:
            m: the trained model
            val_iter: the validation records iterator
    """
    acc = mx.metric.create('acc')
    g = 0.72
    #(speed,) = score(model=m,
    #                 data_val='data/val-5k-256.rec',
    #                 rgb_mean='0, 0, 0', metrics=acc)
    r = acc.get()[1]
    print('Tested %s acc = %f, speed = %f img/sec' % (m, r, speed))
    assert r > g and r < g + .1

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train dogs vs cats",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 2)
    parser.set_defaults(
        # network
        network        = 'resnet',
        num_layers     = 101,
        # data
        data_train     = 'data/data_set/imgdata_train.rec',
        data_val       = 'data/data_set/imgdata_val.rec',
        num_classes    = 2,
        num_examples   = 25000,
        image_shape    = '3,128,128',
        pad_size       = 4,
        # train
        batch_size     = 128,
        num_epochs     = 300,
        lr             = .05,
        lr_step_epochs = '200,250',
        model_prefix = './out/model_dogs_vs_cats'
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.' + args.network)
    sym = net.get_symbol(**vars(args))

    # train
    model = fit.fit(args, sym, data.get_rec_iter)
