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
    parser.add_argument('-v', '--verbose', help='High verbose option used for debug or dev',
                        action='store_true')

    return parser.parse_args()


def main():

    args = get_args()

    # if running on server (MILA), copy dataset locally
    dataset_path = utils.init_dataset(args, 'mscoco_inpainting')

    # Setup input variables
    inpt_noise = T.matrix()
    inpt_image = T.tensor4()
    inpt_image = inpt_image.dimshuffle((0, 3, 1, 2))

    # Build generator and discriminator
    dc_gan = models.DCGAN(args.verbose)
    generator = dc_gan.init_generator(input_var=inpt_noise)
    discriminator = dc_gan.init_discriminator(input_var=inpt_image)

    # Get images from generator
    image_fake = lyr.get_output(generator)

    # Get probabilities from discriminator
    probs_real = lyr.get_output(discriminator) # for real images
    probs_fake = lyr.get_output(discriminator, inputs=image_fake) # for fake images

    # Calc loss for discriminator
    d_loss_real = - T.mean(T.log(probs_real)) # minimize prob of error on true images
    d_loss_fake = - T.mean(T.log(1 - probs_fake)) # minimize prob of error on fake images
    loss_discr = d_loss_real + d_loss_fake

    # Calc loss for generator
    loss_gener = - d_loss_fake # minimize the error of the discriminator on fake images

    # Create params dict for both discriminator and generator
    params_discr = lyr.get_all_params(discriminator, trainable=True)
    params_gener = lyr.get_all_params(generator, trainable=True)

    # Set update rules for params using adam
    updates_discr = lasagne.updates.adam(loss_discr, params_discr, learning_rate=0.001)
    updates_gener = lasagne.updates.adam(loss_gener, params_gener, learning_rate=0.001)

    # Compile Theano functions
    print 'compiling...'
    train_discr = theano.function([inpt_image, inpt_noise], loss_discr, updates=updates_discr)
    print '- 1 of 2 train compiled.'
    train_gener = theano.function([inpt_noise], loss_gener, updates=updates_gener)
    print '- 2 of 2 train compiled.'

    #######################################
    # Nothing was changed pass this point #
    #######################################

    valid = theano.function(inputs=[input_data, targt_data],
                            outputs=[loss, preds])
    print '- 1 of 1 valid compiled.'
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
