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


def show_examples(batch_idx, batch_size,
                  # PATH need to be fixed
                  mscoco="/Tmp/inpainting/", split="train2014",
                  caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"):
    '''
    Show an example of how to read the dataset
    '''

    data_path = os.path.join(mscoco, split)
    caption_path = os.path.join(mscoco, caption_path)
    with open(caption_path) as fd:
        caption_dict = pkl.load(fd)

    print data_path + "/*.jpg"
    imgs = glob.glob(data_path + "/*.jpg")
    batch_imgs = imgs[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    for i, img_path in enumerate(batch_imgs):
        img = Image.open(img_path)
        img_array = np.array(img)

        cap_id = os.path.basename(img_path)[:-4]

        # Get input/target from the images
        center = (
            int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16,
                  center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] +
                               16, center[1] - 16:center[1] + 16, :]
        else:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16,
                  center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] +
                               16, center[1] - 16:center[1] + 16]

        # Image.fromarray(img_array).show()
        Image.fromarray(input).show()
        Image.fromarray(target).show()
        print i, caption_dict[cap_id]


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


def cross_entropy(softmax_out, true_y):
    """
    Returns the mean loss for a minibatch
    """
    return -T.mean(T.log(softmax_out)[T.arange(true_y.shape[0]), true_y])


def accuracy(preds, true_y):
    """
    Returns accuracy score for a minibatch (mean)
    """
    return T.mean(T.eq(preds, true_y))


def init_small_params():
    """
    Initialize the parameters
    """
    # First convolution layer (32 kernels, with 1 channel, of size 5x5)
    w1 = theano.shared(rng.normal(
        size=(64, 1, 5, 5)).astype(theano.config.floatX))
    b1 = theano.shared(np.zeros(shape=64, dtype=theano.config.floatX))

    # Second convolution layer (64 kernels, with 32 channel, of size 5x5)
    w2 = theano.shared(rng.normal(
        size=(128, 64, 5, 5)).astype(theano.config.floatX))
    b2 = theano.shared(np.zeros(shape=128, dtype=theano.config.floatX))

    # Full layer (2048 inputs,512 outputs)
    w3 = theano.shared(rng.normal(
        size=(2048, 512)).astype(theano.config.floatX))
    b3 = theano.shared(np.zeros(shape=512, dtype=theano.config.floatX))

    # Full layer (256 inputs, 10 outputs)
    w4 = theano.shared(rng.normal(size=(512, 10)).astype(theano.config.floatX))
    b4 = theano.shared(np.zeros(shape=10, dtype=theano.config.floatX))

    return [w1, b1, w2, b2, w3, b3, w4, b4]


def init_small_network(params, input_idx, input_data, targt_data, alpha_relu=1.0):

    w1, b1, w2, b2, w3, b3, w4, b4 = params

    # First convolution layer (32 kernels, with 1 channel, of size 3x3)
    conv1 = T.nnet.conv2d(input_data[input_idx], w1, subsample=(2, 2))
    h1 = T.nnet.relu(conv1 + b1.dimshuffle('x', 0, 'x', 'x'), alpha=alpha_relu)

    # Second convolution layer (64 kernels, with 32 channel, of size 5x5)
    conv2 = T.nnet.conv2d(h1, w2, subsample=(2, 2))
    h2 = T.nnet.relu(conv2 + b2.dimshuffle('x', 0, 'x', 'x'), alpha=alpha_relu)
    h2 = h2.flatten(2)

    # Fully connected (256 inputs, 10 outputs)
    h3 = T.nnet.relu(T.dot(h2, w3) + b3, alpha=alpha_relu)
    out = T.nnet.softmax(T.dot(h3, w4) + b4)

    # Compute loss and accuracy
    loss = cross_entropy(out, targt_data[input_idx])
    preds = T.argmax(out, axis=1)
    acc = accuracy(preds, targt_data[input_idx])

    return acc, loss


def build_mnist_cnn(input_var=None):
    """
    Generate the cnn using the Lasagne library
    """
    network = lyr.InputLayer(shape=(None, 1, 28, 28),
                             input_var=input_var)
    network = lyr.Conv2DLayer(network, 64, (5, 5), W=lasagne.init.GlorotNormal())
    network = lyr.MaxPool2DLayer(network, (2, 2))
    network = lyr.Conv2DLayer(network, 128, (5, 5),
                              W=lasagne.init.GlorotNormal())
    network = lyr.MaxPool2DLayer(network, (2, 2))
    network = lyr.Conv2DLayer(network, 256, (4, 4), W=lasagne.init.Normal())

    network = lyr.DenseLayer(network, 512, W=lasagne.init.GlorotNormal())
    network = lyr.DenseLayer(network, 10,
                             nonlinearity=lasagne.nonlinearities.softmax,
                             W=lasagne.init.GlorotNormal())

    return network


def build_auto_encoder_mnist_cnn(input_var=None):
    """
    Generate an auto-encoder cnn using the Lasagne library
    """
    # Build encoder part
    network = lyr.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    network = lyr.Conv2DLayer(network, 64, (5, 5), W=lasagne.init.Normal())
    network = lyr.MaxPool2DLayer(network, (2, 2))
    network = lyr.Conv2DLayer(network, 128, (5, 5), W=lasagne.init.Normal())
    network = lyr.MaxPool2DLayer(network, (2, 2))
    network = lyr.FlattenLayer(network)

    network = lyr.DenseLayer(network, 2048, W=lasagne.init.Normal())
    network = lyr.ReshapeLayer(network, (input_var.shape[0], 2048, 1, 1))

    # Build decoder part
    network = lyr.TransposedConv2DLayer(network, 128, (5, 5), W=lasagne.init.Normal())
    network = lyr.Upscale2DLayer(network, (2, 2))
    network = lyr.TransposedConv2DLayer(network, 64, (4, 4), W=lasagne.init.Normal())
    network = lyr.Upscale2DLayer(network, (2, 2))
    network = lyr.TransposedConv2DLayer(network, 1, (3, 3), W=lasagne.init.Normal(),
                                        nonlinearity=None)

    return network


if __name__ == '__main__':

    args = get_args()
    dataset_path = init_dataset(args, 'mnist')

    with open(dataset_path + '/mnist.pkl') as f:
        train, valid, test = pkl.load(f)

    train_x, train_y = train
    valid_x, valid_y = valid
    test_x, test_y = test

    train_x = train_x.reshape((50000, 1, 28, 28)).astype(theano.config.floatX)
    valid_x = valid_x.reshape((10000, 1, 28, 28)).astype(theano.config.floatX)
    test_x = test_x.reshape((10000, 1, 28, 28)).astype(theano.config.floatX)

    train_y = train_y.astype('int32')
    valid_y = valid_y.astype('int32')
    test_y = test_y.astype('int32')

    # print 'train_x', train_x.shape
    # print 'train_y', train_y.shape
    # print 'valid_x', valid_x.shape
    # print 'valid_y', valid_y.shape
    # print 'test_x ', test_x.shape
    # print 'test_y ', test_y.shape

    # show_examples(5, 1, mscoco=dataset_path)

    batch_size = 128
    nb_epochs = 200
    early_stop_limit = 20

    # initiate tensors
    input_idx = T.vector(dtype='int64')  # used for givens idx
    input_data = T.tensor4()  # used for givens database
    targt_data = T.tensor4()

    # Setup network, params and updates
    network = build_auto_encoder_mnist_cnn(input_data[input_idx])
    sgmd_output = lyr.get_output(network)
    loss = T.mean(lasagne.objectives.squared_error(sgmd_output, targt_data[input_idx]))
    preds = sgmd_output
    # acc = accuracy(preds, targt_data[input_idx])

    params = network.get_params(trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=0.01)

    # Compile Theano functions
    print 'compiling...'
    # test_funct = theano.function(inputs=[input_idx],
    #                              outputs=[output],
    #                              givens={input_data: train_x})
    train = theano.function(inputs=[input_idx],
                            outputs=loss,
                            updates=updates,
                            givens={input_data: train_x,
                                    targt_data: train_x})
    print '- train compiled.'
    valid = theano.function(inputs=[input_idx],
                            outputs=[loss, preds],
                            givens={input_data: valid_x,
                                    targt_data: valid_x})
    print '- valid compiled.'
    print 'compiled.'

    print 'Starting training...'
    valid_idx = np.arange(len(valid_y))
    # valid_acc = []
    valid_loss = []
    # train_acc = []
    train_loss = []
    best_valid_loss = float('inf')

    for i in xrange(nb_epochs):

        # iterate over minibatches for training
        schemes = ShuffledScheme(examples=train_x.shape[0],
                                 batch_size=batch_size)

        epoch_acc = 0
        epoch_loss = 0
        num_batch = 0
        for batch_idx in schemes.get_request_iterator():

            batch_loss = train(batch_idx)
            # epoch_acc += batch_acc
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

        plt.imsave(fname='img_epoch_%s_.jpg' % (i+1), arr=preds[0][0], cmap='gray')

    print 'Training completed.'

    print 'Best performance -- Epoch #%s' % (best_epoch_idx + 1)
    print '- Train %s %%' % (train_loss[best_epoch_idx] * 100)
    print '- Valid %s %%' % (valid_loss[best_epoch_idx] * 100)

    plt.imshow(preds[0])
    plt.show()
