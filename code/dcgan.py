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
    corr_mask = T.matrix() # corruption mask
    corr_image = T.tensor4()

    # Shared variable for image reconstruction
    reconstr_noise = theano.shared(
        np.random.uniform(-1., 1., size=(1, 100)).astype(theano.config.floatX))

    # Build generator and discriminator
    dc_gan = models.DCGAN(args)
    generator = dc_gan.init_generator(first_layer=64, input_var=None)
    discriminator = dc_gan.init_discriminator(first_layer=128, input_var=None)

    # Get images from generator (for training and outputing images)
    image_fake = lyr.get_output(generator, inputs=inpt_noise)
    image_fake_det = lyr.get_output(generator, inputs=inpt_noise, deterministic=True)
    image_reconstr = lyr.get_output(generator, inputs=reconstr_noise, deterministic=True)

    # Get probabilities from discriminator
    probs_real = lyr.get_output(discriminator, inputs=inpt_image)
    probs_fake = lyr.get_output(discriminator, inputs=image_fake)
    probs_fake_det = lyr.get_output(
        discriminator, inputs=image_fake_det, deterministic=True)
    probs_reconstr = lyr.get_output(
        discriminator, inputs=image_reconstr, deterministic=True)

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

    # Contextual and perceptual loss for
    contx_loss = lasagne.objectives.squared_error(
        image_reconstr * corr_mask, corr_image * corr_mask)
    prcpt_loss = T.log(1.0 - probs_reconstr)

    # Total loss
    lbda = 0.0001
    reconstr_loss = T.mean(contx_loss + lbda * prcpt_loss)

    # Set update rule that will change the input noise
    # reconstr_updates = lasagne.updates.sgd(reconstr_loss, reconstr_noise, 0.0001)
    grad = T.grad(reconstr_loss, reconstr_noise)
    lr = 0.001
    update_rule = reconstr_noise - lr * grad

    if args.verbose:
        print 'Networks created.'

    # Compile Theano functions
    print 'compiling...'
    train_d = theano.function(
        [inpt_image, inpt_noise], loss_discr, updates=updates_discr)
    print '- 1 of 4 compiled.'
    train_g = theano.function(
        [inpt_noise], loss_gener, updates=updates_gener)
    print '- 2 of 4 compiled.'
    predict = theano.function([inpt_noise], [image_fake_det, probs_fake_det])
    print '- 3 of 4 compiled.'
    reconstr = theano.function(
        [corr_image, corr_mask], [reconstr_noise, image_reconstr, reconstr_loss], updates=[(reconstr_noise, update_rule)])
    print '- 4 of 4 compiled.'
    print 'compiled.'

    return train_d, train_g, predict, reconstr, (discriminator, generator)


def reconstruct_img(images_full, mask_corr, reconstr_fn):
    """
    Reconstructs the image
    ---
    mask_corr: matrix that is applied to make the image corrupted
    """

    preds = np.array([])
    images_corr = np.product((images_full, mask_corr))

    for image_corr in images_corr:
        print 'corr image shape', image_corr.shape
        print 'corr mask shape', mask_corr.shape
        reconstr_out = reconstr_fn(image_corr, mask_corr)
        reconstr_noise, prediction, reconstr_loss = reconstr_out
        print 'reconstr_loss', reconstr_loss
        preds = np.append(preds, prediction)

    reconstr_images = mask_corr * images_corr + (1.0 - mask_corr) * preds

    return reconstr_images


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
    train_discr, train_gen, predict, reconstr_fn, model = theano_fn

    # get different file names for the split data set
    train_files = utils.get_preprocessed_files(train_path)
    train_full_files, train_cter_files, train_capt_files = train_files

    valid_files = utils.get_preprocessed_files(valid_path)
    valid_full_files, valid_cter_files, valid_capt_files = valid_files

    NB_TRAIN_FILES = len(train_full_files)
    NB_VALID_FILES = len(valid_full_files)

    corruption_mask = utils.get_corruption_mask()

    print 'Starting training...'

    valid_loss = []
    train_loss = []
    best_valid_loss = float('inf')

    if not args.reload == None:
        discriminator, generator = model
        file_discr = 'discrminator_epoch_%s.pkl' % args.reload
        file_gen = 'generator_epoch_%s.pkl' % args.reload
        discriminator = utils.reload_model(args, discriminator, file_discr)
        generator = utils.reload_model(args, generator, file_gen)

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
            with open(train_full_files[file_id], 'r') as f:
                train_full = np.load(f).astype(theano.config.floatX)

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

                ##############
                #to deleete
                file_id = np.random.choice(NB_VALID_FILES, 1)

                # load file
                with open(valid_full_files[file_id], 'r') as f:
                    valid_full = np.load(f).astype(theano.config.floatX)

                t_load = time.time()

                if args.verbose:
                    print 'file %s loaded in %s sec' % (valid_full_files[file_id], round(time.time() - t_load, 0))

                # pick a given number of images from that file
                batch_valid = np.random.choice(len(valid_full), NB_GEN, replace=False)
                print 'batch valid', batch_valid

                # reconstruct image
                img_uncorrpt = valid_full[batch_valid]
                img_reconstr = reconstruct_img(img_uncorrpt, corruption_mask, reconstr_fn)

                # save images
                utils.save_pics_gan(args, img_reconstr, 'pred_epoch_%s' %(i+1), show=False, save=True, tanh=False)
                utils.save_pics_gan(args, img_uncorrpt, 'true_epoch_%s' %(i+1), show=False, save=True, tanh=False)

                ###
                #################
        train_loss.append(np.round(epoch_loss, 4))

        if args.save > 0 and i % args.save == 0:
            discriminator, generator = model
            utils.save_model(args, discriminator, 'discrminator_epoch_%s.pkl' % i)
            utils.save_model(args, generator, 'generator_epoch_%s.pkl' % i)

        print '- Epoch train (loss %s) in %s sec' % (train_loss[i], round(time.time() - t_epoch))

        # # generate some random images
        # gen_noise = np.random.uniform(-1., 1., size=(NB_GEN, 100))
        # gen_noise = gen_noise.astype(theano.config.floatX)
        # preds_gen, probs_discr = predict(gen_noise)


        # Reconstruct images from valid set
        # choose random valid file


        # save losses at each step
        utils.dump_objects_output(args, (steps_loss_d, steps_loss_g), 'steps_loss_epoch_%s.pkl' % i)


    print 'Training completed.'

    if args.mila:
        utils.move_results_from_local()


if __name__ == '__main__':

    main()
