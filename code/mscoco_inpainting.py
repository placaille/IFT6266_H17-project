#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import cPickle as pkl
import numpy as np
import numpy.random as rng
import PIL.Image as Image
import argparse
import shutil
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as lyr
import matplotlib.pyplot as plt
from fuel.schemes import ShuffledScheme


def get_args():
    """
    Returns the arguments passed by command-line
    """
    parser = argparse.ArgumentParser()
    load_src = parser.add_mutually_exclusive_group(required=True)
    load_src.add_argument('-m', '--mila', help='If running from MILA servers',
                          action='store_true')
    load_src.add_argument('-l', '--laptop', help='If running from laptop',
                          action='store_true')

    return parser.parse_args()


def get_batch_data(batch_idx,
                   # PATH need to be fixed
                   mscoco="/Tmp/inpainting/", split="train2014",
                   caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"):
    '''
    Show an example of how to read the dataset
    @return inputs, targets, captions, color_count
    '''

    data_path = os.path.join(mscoco, split)
    caption_path = os.path.join(mscoco, caption_path)
    with open(caption_path) as fd:
        caption_dict = pkl.load(fd)

    print data_path + "/*.jpg"
    imgs = np.asarray(glob.glob(data_path + "/*.jpg"))
    batch_imgs = imgs[batch_idx]

    color_count = 0
    inputs = np.array([])
    targets = np.array([])
    captions = {}
    for i, img_path in enumerate(batch_imgs):
        img = Image.open(img_path)
        img_array = np.array(img)

        cap_id = os.path.basename(img_path)[:-4]

        # Get input/target from the images
        center = (
            int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:  # if colored images (RGB)
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16,
                  center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] +
                               16, center[1] - 16:center[1] + 16, :]
            color_count += 1
        else:  # if black and white (do nothing)
            continue

        inputs = np.append(inputs, input)
        targets = np.append(targets, target)
        captions[color_count - 1] = caption_dict[cap_id]

    return inputs, targets, captions, color_count


def init_dataset(args, dataset_name):
    """
    If running from MILA, copy on /Tmp/lacaillp/datasets/
    ---
    returns path of local dataset
    """
    if args.mila:
        src_dir = '/data/lisatmp3/lacaillp/datasets/'
        dst_dir = '/Tmp/lacaillp/datasets/'
    elif args.laptop:
        src_dir = '/Users/phil/datasets/'
        dst_dir = src_dir
    else:
        raise 'Location entered not valid (MILA/laptop)'

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    if not os.path.exists(dst_dir + dataset_name):
        print 'Dataset not stored locally, copying %s to %s...' \
            % (dataset_name, dst_dir)
        shutil.copytree(src_dir + dataset_name, dst_dir + dataset_name)
        print 'Copy completed.'

    return dst_dir + dataset_name


def build_network(input_var=None):
    """
    Build the network using Lasagne library
    """

    ##################
    # Network config #
    ##################

    input_channels = 3
    weight_init = lasagne.init.Normal()

    # encoder
    conv1_nb_filt = 64
    conv1_sz_filt = (9, 9)
    conv1_sz_padd = 2
    # conv1 output size = (60, 60)

    pool1_sz = (2, 2)
    # pool1 output size = (30, 30)

    conv2_nb_filt = 128
    conv2_sz_filt = (7, 7)
    conv2_sz_padd = 0
    # conv2 output size = (24, 24)

    pool2_sz = (4, 4)
    # pool2 size = (6, 6)

    conv3_nb_filt = 256
    conv3_sz_filt = (5, 5)
    conv3_sz_padd = 0
    # conv3 output size = (2, 2)

    pool3_sz = (2, 2)
    # pool3 output size = (1, 1)

    dens1_nb_unit = 512
    # dense1 output to the decoder (vector 512)

    # decoder
    tconv1_nb_filt = 256
    tconv1_sz_filt = (5, 5)
    tconv1_sz_strd = (1, 1)
    # conv1 output size = (5, 5)

    upsamp1_sz = (2, 2)
    # upsamp1 output size = (10, 10)

    tconv2_nb_filt = 128
    tconv2_sz_filt = (4, 4)
    tconv2_sz_strd = (1, 1)
    # tconv2 output size = (13, 13)

    upsamp2_sz = (2, 2)
    # upsamp2 output size = (26, 26)

    tconv3_nb_filt = 64
    tconv3_sz_filt = (5, 5)
    tconv3_sz_strd = (1, 1)
    # tconv3 output size = (30, 30)

    tconv4_nb_filt = 3
    tconv4_sz_filt = (3, 3)
    tconv4_sz_strd = (1, 1)
    # tconv4 output size = (32, 32)

    # final output = (3 channels, 32 x 32)

    #####################
    # Build the network #
    #####################

    # Add input layer
    network = lyr.InputLayer(
        shape=(None, input_channels, 64, 64), input_var=input_var)

    # Add convolution layer
    network = lyr.Conv2DLayer(incoming=network, num_filters=conv1_nb_filt,
                              filter_size=conv1_sz_filt, pad=conv1_sz_padd,
                              W=weight_init)
    # Add pooling layer
    network = lyr.MaxPool2DLayer(incoming=network, pool_size=pool1_sz)

    # Add convolution layer
    network = lyr.Conv2DLayer(incoming=network, num_filters=conv2_nb_filt,
                              filter_size=conv2_sz_filt, pad=conv2_sz_padd,
                              W=weight_init)
    # Add pooling layer
    network = lyr.MaxPool2DLayer(incoming=network, pool_size=pool2_sz)

    # Add convolution layer
    network = lyr.Conv2DLayer(incoming=network, num_filters=conv3_nb_filt,
                              filter_size=conv3_sz_filt, pad=conv3_sz_padd,
                              W=weight_init)
    # Add pooling layer
    network = lyr.MaxPool2DLayer(incoming=network, pool_size=pool3_sz)

    network = lyr.FlattenLayer(network)

    # Add dense layer
    network = lyr.DenseLayer(network, dens1_nb_unit, W=weight_init)
    network = lyr.ReshapeLayer(
        network, (input_var.shape[0], dens1_nb_unit, 1, 1))

    # Add transposed convolution layer
    network = lyr.TransposedConv2DLayer(incoming=network,
                                        num_filters=tconv1_nb_filt,
                                        filter_size=tconv1_sz_filt,
                                        stride=tconv1_sz_strd,
                                        W=weight_init)
    # Add upsampling layer
    network = lyr.Upscale2DLayer(incoming=network, scale_factor=upsamp1_sz)

    # Add transposed convolution layer
    network = lyr.TransposedConv2DLayer(incoming=network,
                                        num_filters=tconv2_nb_filt,
                                        filter_size=tconv2_sz_filt,
                                        stride=tconv2_sz_strd,
                                        W=weight_init)
    # Add upsampling layer
    network = lyr.Upscale2DLayer(incoming=network, scale_factor=upsamp2_sz)

    # Add transposed convolution layer
    network = lyr.TransposedConv2DLayer(incoming=network,
                                        num_filters=tconv3_nb_filt,
                                        filter_size=tconv3_sz_filt,
                                        stride=tconv3_sz_strd,
                                        W=weight_init)

    # Add transposed convolution layer
    network = lyr.TransposedConv2DLayer(incoming=network,
                                        num_filters=tconv4_nb_filt,
                                        filter_size=tconv4_sz_filt,
                                        stride=tconv4_sz_strd,
                                        W=weight_init,
                                        nonlinearity=lasagne.nonlinearities.sigmoid)

    return network

if __name__ == '__main__':

    args = get_args()
    dataset_path = init_dataset(args, 'mscoco_inpainting')

    # initiate tensors
    input_data = T.tensor4()  # used for givens database
    targt_data = T.tensor4()

    # Setup network, params and updates
    network = build_network(input_var=input_data.dimshuffle(0, 3, 1, 2))
    output = lyr.get_output(network).dimshuffle(0, 2, 3, 1)
    loss = T.mean(lasagne.objectives.squared_error(
        output, targt_data))
    preds = output

    params = network.get_params(trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=0.01)

    # Compile Theano functions
    print 'compiling...'
    train = theano.function(inputs=[input_data, targt_data], outputs=[
                            loss, output], updates=updates)
    print '- train compiled.'
    # valid = theano.function(inputs=[input_idx], outputs=[loss, preds])
    print '- valid compiled.'
    print 'compiled.'

    batch_size = 256
    nb_epochs = 200
    early_stop_limit = 10
    NB_TRAIN = 82782
    NB_VALID = 40504

    print 'Starting training...'

    valid_loss = []
    train_loss = []
    best_valid_loss = float('inf')

    for i in xrange(nb_epochs):

        # iterate over minibatches for training
        schemes = ShuffledScheme(examples=NB_TRAIN,
                                 batch_size=batch_size)

        epoch_acc = 0
        epoch_loss = 0
        num_batch = 0

        for batch_idx in schemes.get_request_iterator():

            # get training data for this batch
            inputs, targts, capts, color_count = get_batch_data(
                batch_idx, mscoco=dataset_path, split="train2014")

            inputs_rsp = np.reshape(inputs, (color_count, 64, 64, 3)) / 255
            targts_rsp = np.reshape(targts, (color_count, 32, 32, 3)) / 255

            print 'inputs shape:\t', inputs_rsp.shape
            print 'targts shape:\t', targts_rsp.shape

            inputs_rsp = inputs_rsp.astype(theano.config.floatX)
            targts_rsp = targts_rsp.astype(theano.config.floatX)

            batch_loss, output = train(inputs_rsp, targts_rsp)

            print 'outpts shape:\t', output.shape
            epoch_loss += batch_loss
            num_batch += 1

        # train_acc.append(np.round(epoch_acc / num_batch, 4))
        train_loss.append(np.round(epoch_loss / num_batch, 4))

        epoch_loss, preds = valid(valid_idx)
        # valid_acc.append(np.round(batch_acc, 4))
        valid_loss.append(np.round(epoch_loss, 4))

        print 'Epoch #%s of %s' % ((i + 1), nb_epochs)
        print '- Train (loss %s)' % (train_loss[i])
        print '- Valid (loss %s)' % (valid_loss[i])

        if valid_loss[i] < best_valid_loss:
            best_valid_loss = valid_loss[i]
            best_epoch_idx = i
            early_stp_counter = 0
        else:
            early_stp_counter += 1
            if early_stp_counter >= early_stop_limit:
                print '**Early stopping activated, %s epochs without improvement.' % early_stop_limit
                break

        plt.imsave(fname='img_epoch_%s_.jpg' %
                   (i + 1), arr=preds[0][0], cmap='gray')

    print 'Training completed.'

    print 'Best performance -- Epoch #%s' % (best_epoch_idx + 1)
    print '- Train %s %%' % (train_loss[best_epoch_idx] * 100)
    print '- Valid %s %%' % (valid_loss[best_epoch_idx] * 100)

    plt.imshow(preds[0])
    plt.show()
