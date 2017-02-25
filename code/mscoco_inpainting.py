#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as lyr
from fuel.schemes import ShuffledScheme

import models
import utils


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

    parser.add_argument('-e', '--epochs', help='Max number of epochs for training',
                        type=int, default=100)
    parser.add_argument('-g', '--gen', help='Number of images to generate from valid',
                        type=int, default=5)

    return parser.parse_args()


def main():

    args = get_args()

    # if running on server (MILA), copy dataset locally
    dataset_path = utils.init_dataset(args, 'mscoco_inpainting')

    # initiate tensors
    input_data = T.tensor4()
    targt_data = T.tensor4()

    input_var = input_data.dimshuffle((0, 3, 1, 2))
    targt_var = targt_data.dimshuffle((0, 3, 1, 2))

    # Setup network, params and updates
    network = models.small_cnn_autoencoder(input_var=input_var)
    outputs = lyr.get_output(network)
    loss = T.mean(lasagne.objectives.squared_error(outputs, targt_var))
    preds = lyr.get_output(network, deterministic=True)

    params = lyr.get_all_params(network, trainable=True)

    updates = lasagne.updates.adam(loss, params, learning_rate=0.001)

    # Compile Theano functions
    print 'compiling...'
    train = theano.function(inputs=[input_data, targt_data],
                            outputs=loss, updates=updates)
    print '- train compiled.'
    valid = theano.function(inputs=[input_data, targt_data],
                            outputs=[loss, preds])
    print '- valid compiled.'
    print 'compiled.'

    BATCH_SIZE = 128
    NB_EPOCHS = args.epochs
    NB_GEN = args.gen
    EARLY_STOP_LIMIT = 10
    NB_TRAIN = 82782
    NB_VALID = 40504

    print 'Starting training...'

    valid_loss = []
    train_loss = []
    best_valid_loss = float('inf')
    ID_PRINT = np.random.choice(NB_VALID, NB_GEN, replace=False)

    for i in xrange(NB_EPOCHS):

        # iterate over minibatches for training
        schemes_train = ShuffledScheme(examples=NB_TRAIN,
                                       batch_size=BATCH_SIZE)

        epoch_acc = 0
        epoch_loss = 0
        num_batch = 0

        print 'Epoch #%s of %s' % ((i + 1), NB_EPOCHS)

        for batch_idx in schemes_train.get_request_iterator():

            # get training data for this batch
            inputs, targts, capts, color_count = utils.get_batch_data(
                batch_idx, mscoco=dataset_path, split="train2014")

            batch_loss = train(inputs, targts)

            if num_batch % 100 == 0:
                print '- train batch %s, loss %s' % (num_batch, np.round(batch_loss, 4))

            epoch_loss += batch_loss
            num_batch += 1

        train_loss.append(np.round(epoch_loss, 4))

        print '- Epoch train (loss %s)' % (train_loss[i])

        # Validation only done on couple of images for speed
        inputs_val, targts_val, capts_val, color_count_val = utils.get_batch_data(
            ID_PRINT, mscoco=dataset_path, split='val2014')

        loss_val, preds_val = valid(inputs_val, targts_val)

        valid_loss.append(np.round(loss_val, 6))

        # Generate images
        gen_pics(inputs_val, targts_val, preds_val.transpose(
            (0, 2, 3, 1)), i, save=True)

        print '- Epoch valid (loss %s)' % (valid_loss[i])

        if valid_loss[i] < best_valid_loss:
            best_valid_loss = valid_loss[i]
            best_epoch_idx = i
            early_stp_counter = 0
        else:
            early_stp_counter += 1
            if early_stp_counter >= EARLY_STOP_LIMIT:
                print '**Early stopping activated, %s epochs without improvement.' % EARLY_STOP_LIMIT
                break

    print 'Training completed.'

    print 'Best performance -- Epoch #%s' % (best_epoch_idx + 1)
    print '- Train %s' % (train_loss[best_epoch_idx])
    print '- Valid %s' % (valid_loss[best_epoch_idx])


if __name__ == '__main__':

    main()
