#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import os
import glob
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as lyr
from fuel.schemes import ShuffledScheme

import models
import utils


def gen_theano_fn(args):
    """
    Generate the networks and returns the train functions
    """
    if args.verbose:
        print 'Creating networks...'

    # Setup input variables
    inpt_noise = T.matrix()
    inpt_image = T.tensor4()

    # Build generator and discriminator
    dc_gan = models.DCGAN(args)
    generator = dc_gan.init_generator(first_layer=64, input_var=None)
    discriminator = dc_gan.init_discriminator(first_layer=128, input_var=None)

    # Get images from generator (for training and outputing images)
    image_fake = lyr.get_output(generator, inputs=inpt_noise)
    image_fake_det = lyr.get_output(generator, inputs=inpt_noise, deterministic=True)

    # Get probabilities from discriminator
    probs_real = lyr.get_output(discriminator, inputs=inpt_image)
    probs_fake = lyr.get_output(discriminator, inputs=image_fake)
    probs_fake_det = lyr.get_output(
        discriminator, inputs=image_fake_det, deterministic=True)

    # Calc loss for discriminator
    # minimize prob of error on true images
    d_loss_real = - T.mean(T.log(probs_real))
    # minimize prob of error on fake images
    d_loss_fake = - T.mean(T.log(1 - probs_fake))
    loss_discr = d_loss_real + d_loss_fake

    # Calc loss for generator
    # minimize the error of the discriminator on fake images
    loss_gener = - T.mean(T.log(probs_fake))

    # Create params dict for both discriminator and generator
    params_discr = lyr.get_all_params(discriminator, trainable=True)
    params_gener = lyr.get_all_params(generator, trainable=True)

    # Set update rules for params using adam
    updates_discr = lasagne.updates.adam(
        loss_discr, params_discr, learning_rate=0.001, beta1=0.9)
    updates_gener = lasagne.updates.adam(
        loss_gener, params_gener, learning_rate=0.0005, beta1=0.6)

    if args.verbose:
        print 'Networks created.'

    # Compile Theano functions
    print 'compiling...'
    train_d = theano.function(
        [inpt_image, inpt_noise], loss_discr, updates=updates_discr)
    print '- 1 of 3 compiled.'
    train_g = theano.function(
        [inpt_noise], loss_gener, updates=updates_gener)
    print '- 2 of 3 compiled.'
    predict = theano.function([inpt_noise], [image_fake_det, probs_fake_det])
    print '- 3 of 3 compiled.'
    print 'compiled.'

    return train_d, train_g, predict, (discriminator, generator)


def main():

    args = utils.get_args()

    # Settings for training
    BATCH_SIZE = 128
    NB_EPOCHS = args.epochs  # default 25
    NB_GEN = args.gen  # default 5
    EARLY_STOP_LIMIT = 10
    GEN_TRAIN_DELAY = 30
    GEN_TRAIN_LOOPS = 10

    if args.verbose:
        BATCH_PRINT_DELAY = 1
    else:
        BATCH_PRINT_DELAY = 100

    # if running on server (MILA), copy dataset locally
    dataset_path = utils.init_dataset(args, 'mscoco_inpainting/preprocessed')
    train_path = os.path.join(dataset_path, 'train2014')
    valid_path = os.path.join(dataset_path, 'val2014')

    # build network and get theano functions for training
    theano_fn = gen_theano_fn(args)
    train_discr, train_gen, predict, model = theano_fn

    # get different file names for the split data set
    train_full_files = np.asarray(
        sorted(glob.glob(train_path + '/*_full.npy')))
    train_cter_files = np.asarray(
        sorted(glob.glob(train_path + '/*_cter.npy')))
    train_capt_files = np.asarray(
        sorted(glob.glob(train_path + '/*_capt.pkl')))

    valid_full_files = np.asarray(
        sorted(glob.glob(valid_path + '/*_full.npy')))
    valid_cter_files = np.asarray(
        sorted(glob.glob(valid_path + '/*_cter.npy')))
    valid_capt_files = np.asarray(
        sorted(glob.glob(valid_path + '/*_capt.pkl')))

    NB_TRAIN_FILES = len(train_full_files)
    NB_VALID_FILES = len(valid_full_files)

    print 'Starting training...'

    valid_loss = []
    train_loss = []
    best_valid_loss = float('inf')

    if not args.reload == None:
        discriminator, generator = model
        utils.reload_model(discriminator, model, args.reload)

    for i in xrange(NB_EPOCHS):

        print 'Epoch #%s of %s' % ((i + 1), NB_EPOCHS)

        epoch_acc = 0
        epoch_loss = 0
        num_batch = 0
        t_epoch = time.time()
        d_batch_loss = 0
        g_batch_loss = 0
        steps_loss_g = [] # will store every loss of generator
        steps_loss_d = [] # will store every loss of discriminator

        # iterate of split datasets
        for file_id in np.random.choice(NB_TRAIN_FILES, NB_TRAIN_FILES, replace=False):

            t_load = time.time()
            # load file
            train_full = np.load(open(train_full_files[file_id], 'r')).astype(
                theano.config.floatX)
            # train_cter = np.load(open(train_cter_files[file_id], 'r')).astype(theano.config.floatX)
            # train_capt = pkl.load(open(train_capt_files[file_id], 'rb')).astype(theano.config.floatX)

            if args.verbose:
                print 'file %s loaded in %s sec' % (train_full_files[file_id], round(time.time() - t_load, 0))

            # iterate over minibatches for training
            schemes_train = ShuffledScheme(examples=len(train_full),
                                           batch_size=BATCH_SIZE)

            for batch_idx in schemes_train.get_request_iterator():

                # generate batch of uniform samples
                rdm_d = np.random.uniform(-1., 1., size=(len(batch_idx), 100))
                rdm_d = rdm_d.astype(theano.config.floatX)

                # train with a minibatch on discriminator
                d_batch_loss = train_discr(train_full[batch_idx], rdm_d)

                steps_loss_d.append(d_batch_loss)
                steps_loss_g.append(g_batch_loss)

                if num_batch % BATCH_PRINT_DELAY == 0:
                    print '- train discr batch %s, loss %s' % (num_batch, np.round(d_batch_loss, 4))

                if num_batch % GEN_TRAIN_DELAY == 0:

                    for _ in xrange(GEN_TRAIN_LOOPS):

                        # generate batch of uniform samples
                        rdm_g = np.random.uniform(-1., 1., size=(len(batch_idx), 100))
                        rdm_g = rdm_g.astype(theano.config.floatX)

                        # train with a minibatch on generator
                        g_batch_loss = train_gen(rdm_g)

                        steps_loss_d.append(d_batch_loss)
                        steps_loss_g.append(g_batch_loss)

                        if num_batch % BATCH_PRINT_DELAY == 0:
                            print '- train gen step %s, loss %s' %(_+1, np.round(g_batch_loss, 4))

                epoch_loss += d_batch_loss + g_batch_loss
                num_batch += 1


        train_loss.append(np.round(epoch_loss, 4))

        if args.save:
            discriminator, generator = model
            utils.save_model(discriminator, 'discrminator_epoch_%s.pkl' % i)
            utils.save_model(generator, 'generator_epoch_%s.pkl' % i)


        print '- Epoch train (loss %s) in %s sec' % (train_loss[i], round(time.time() - t_epoch))

        # generate some random images
        gen_noise = np.random.uniform(-1., 1., size=(NB_GEN, 100))
        gen_noise = gen_noise.astype(theano.config.floatX)
        preds_gen, probs_discr = predict(gen_noise)

        if args.verbose:
            print 'Discriminator prob(real):', probs_discr

        # save the images
        utils.gen_pics_gan(preds_gen, i, show=False, save=True, tanh=False)
        utils.gen_pics_gan(train_full[:5], 888, show=False, save=True, tanh=False)

        # save losses at each step
        utils.dump_objects_output((steps_loss_d, steps_loss_g), 'steps_loss_epoch_%s.pkl' % i)

        # # Validation only done on couple of images for speed
        # inputs_val, targts_val, capts_val, color_count_val = utils.get_batch_data(
        #     ID_PRINT, mscoco=dataset_path, split='val2014')
        #
        # loss_val, preds_val = valid(inputs_val, targts_val)
        #
        # valid_loss.append(np.round(loss_val, 6))
        #
        # # Generate images
        # gen_pics(inputs_val, targts_val, preds_val.transpose(
        #     (0, 2, 3, 1)), i, save=True)
        #
        # print '- Epoch valid (loss %s)' % (valid_loss[i])
        #
        # if valid_loss[i] < best_valid_loss:
        #     best_valid_loss = valid_loss[i]
        #     best_epoch_idx = i
        #     early_stp_counter = 0
        # else:
        #     early_stp_counter += 1
        #     if early_stp_counter >= EARLY_STOP_LIMIT:
        #         print '**Early stopping activated, %s epochs without improvement.' % EARLY_STOP_LIMIT
        #         break

    print 'Training completed.'

    # print 'Best performance -- Epoch #%s' % (best_epoch_idx + 1)
    # print '- Train %s' % (train_loss[best_epoch_idx])
    # print '- Valid %s' % (valid_loss[best_epoch_idx])


if __name__ == '__main__':

    main()
