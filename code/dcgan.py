#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as lyr
from fuel.schemes import ShuffledScheme

import models
import utils

def gen_train_fn(args):
    """
    Generate the networks and returns the train functions
    """
    if args.verbose:
        print 'Creating networks...'

    # Setup input variables
    inpt_noise = T.matrix()
    inpt_image = T.tensor4()

    # Build generator and discriminator
    dc_gan = models.DCGAN(args.verbose)
    generator = dc_gan.init_generator(input_var=inpt_noise)
    discriminator = dc_gan.init_discriminator(input_var=inpt_image)

    # Get images from generator
    image_fake = lyr.get_output(generator)

    # Get probabilities from discriminator
    probs_real = lyr.get_output(discriminator)  # for real images
    probs_fake = lyr.get_output(
        discriminator, inputs=image_fake)  # for fake images

    # Calc loss for discriminator
    # minimize prob of error on true images
    d_loss_real = - T.mean(T.log(probs_real))
    # minimize prob of error on fake images
    d_loss_fake = - T.mean(T.log(1 - probs_fake))
    loss_discr = d_loss_real + d_loss_fake

    # Calc loss for generator
    loss_gener = - d_loss_fake  # minimize the error of the discriminator on fake images

    # Create params dict for both discriminator and generator
    params_discr = lyr.get_all_params(discriminator, trainable=True)
    params_gener = lyr.get_all_params(generator, trainable=True)

    # Set update rules for params using adam
    updates_discr = lasagne.updates.adam(
        loss_discr, params_discr, learning_rate=0.002, beta1=0.5)
    updates_gener = lasagne.updates.adam(
        loss_gener, params_gener, learning_rate=0.002, beta1=0.5)

    if args.verbose:
        print 'done.'

    # Compile Theano functions
    print 'compiling...'
    train_d = theano.function(
        [inpt_image, inpt_noise], loss_discr, updates=updates_discr)
    print '- 1 of 2 train compiled.'
    train_g = theano.function(
        [inpt_noise], loss_gener, updates=updates_gener)
    print '- 2 of 2 train compiled.'
    print 'compiled.'

    return train_d, train_g


def main():

    args = utils.get_args()

    # if running on server (MILA), copy dataset locally
    dataset_path = utils.init_dataset(args, 'mscoco_inpainting')

    # build network and get theano functions for training
    train_fn = gen_train_fn(args)
    train_discr, train_gen = train_fn

    #######################################
    # Nothing was changed pass this point #
    #######################################

    BATCH_SIZE = 128
    NB_EPOCHS = args.epochs # default 25
    NB_GEN = args.gen # default 5
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
            img_batch, targts, capts, color_count = utils.get_batch_data(
                batch_idx, mscoco=dataset_path, split="train2014", crop=False)

            # generate batch uniform sample
            rdm_batch = np.random.uniform(-1., 1., size=(color_count, 100))
            rdm_batch = rdm_batch.astype(theano.config.floatX)

            d_batch_loss = train_discr(img_batch, rdm_batch)

            if num_batch % 100 == 0:
                print '- train batch %s, loss %s' % (num_batch, np.round(d_batch_loss, 4))

            epoch_loss += d_batch_loss
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
