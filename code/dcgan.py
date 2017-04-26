#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import os
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as lyr
import cPickle as pkl
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
    inpt_noise = T.matrix('inpt_noise')
    inpt_image = T.tensor4('inpt_image')
    corr_mask = T.matrix('corr_mask')  # corruption mask
    corr_image = T.tensor4('corr_image')
    if args.captions:
        inpt_embd = T.matrix('inpt_embedding')

    # Shared variable for image reconstruction
    reconstr_noise_shrd = theano.shared(
        np.random.uniform(-1., 1., size=(1, 100)).astype(theano.config.floatX))

    # Build generator and discriminator
    if args.captions:
        cond_gen_dc_gan = models.CaptionGenOnlyDCGAN(args)
        generator, lyr_gen_noise, lyr_gen_embd = cond_gen_dc_gan.init_generator(
            first_layer=64, input_var=None, embedding_var=None)
        discriminator = cond_gen_dc_gan.init_discriminator(first_layer=128, input_var=None)
    else:
        dc_gan = models.DCGAN(args)
        generator = dc_gan.init_generator(first_layer=64, input_var=None)
        discriminator = dc_gan.init_discriminator(first_layer=128, input_var=None)

    # Get images from generator (for training and outputing images)
    if args.captions:
        image_fake = lyr.get_output(generator, inputs={lyr_gen_noise: inpt_noise, lyr_gen_embd: inpt_embd})
        image_fake_det = lyr.get_output(generator,
                                        inputs={lyr_gen_noise: inpt_noise, lyr_gen_embd: inpt_embd},
                                        deterministic=True)
        image_reconstr = lyr.get_output(generator,
                                        inputs={lyr_gen_noise: reconstr_noise_shrd, lyr_gen_embd: inpt_embd},
                                        deterministic=True)
    else:
        image_fake = lyr.get_output(generator, inputs=inpt_noise)
        image_fake_det = lyr.get_output(generator, inputs=inpt_noise, deterministic=True)
        image_reconstr = lyr.get_output(generator, inputs=reconstr_noise_shrd, deterministic=True)

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
    contx_loss = T.mean(lasagne.objectives.squared_error(
        image_reconstr * corr_mask, corr_image * corr_mask))
    prcpt_loss = T.mean(T.log(1 - probs_reconstr))

    # Total loss
    lbda = 10.0 ** -5
    reconstr_loss = contx_loss + lbda * prcpt_loss

    # Set update rule that will change the input noise
    grad = T.grad(reconstr_loss, reconstr_noise_shrd)
    lr = 0.9
    update_rule = reconstr_noise_shrd - lr * grad

    if args.verbose:
        print 'Networks created.'

    # Compile Theano functions
    print 'compiling...'

    if args.captions:
        train_d = theano.function(
            [inpt_image, inpt_noise, inpt_embd], loss_discr, updates=updates_discr)
        print '- 1 of 4 compiled.'
        train_g = theano.function(
            [inpt_noise, inpt_embd], loss_gener, updates=updates_gener)
        print '- 2 of 4 compiled.'
        predict = theano.function([inpt_noise, inpt_embd], [image_fake_det, probs_fake_det])
        print '- 3 of 4 compiled.'
        reconstr = theano.function(
            [corr_image, corr_mask, inpt_embd], [reconstr_noise_shrd, image_reconstr, reconstr_loss, grad],
            updates=[(reconstr_noise_shrd, update_rule)])
        print '- 4 of 4 compiled.'
    else:
        train_d = theano.function(
            [inpt_image, inpt_noise], loss_discr, updates=updates_discr)
        print '- 1 of 4 compiled.'
        train_g = theano.function(
            [inpt_noise], loss_gener, updates=updates_gener)
        print '- 2 of 4 compiled.'
        predict = theano.function([inpt_noise], [image_fake_det, probs_fake_det])
        print '- 3 of 4 compiled.'
        reconstr = theano.function(
            [corr_image, corr_mask], [reconstr_noise_shrd, image_reconstr, reconstr_loss, grad],
            updates=[(reconstr_noise_shrd, update_rule)])
        print '- 4 of 4 compiled.'

    print 'compiled.'

    return train_d, train_g, predict, reconstr, reconstr_noise_shrd, (discriminator, generator)


def main():
    args = utils.get_args()

    # Settings for training
    BATCH_SIZE = 128
    NB_EPOCHS = args.epochs  # default 25
    NB_GEN = args.gen  # default 5
    TRAIN_STEPS_DISCR = 15
    TRAIN_STEPS_GEN = 10
    if args.reload is not None:
        RELOAD_SRC = args.reload[0]
        RELOAD_ID = args.reload[1]

    if args.verbose:
        BATCH_PRINT_DELAY = 1
    else:
        BATCH_PRINT_DELAY = 100

    # if running on server (MILA), copy dataset locally
    dataset_path = utils.init_dataset(args, 'mscoco_inpainting/preprocessed')
    train_path = os.path.join(dataset_path, 'train2014')
    valid_path = os.path.join(dataset_path, 'val2014')

    if args.captions:
        t = time.time()
        embedding_model = utils.init_google_word2vec_model(args)
        print 'Embedding model was loaded in %s secs' % np.round(time.time() - t, 0)

    # build network and get theano functions for training
    theano_fn = gen_theano_fn(args)
    train_discr, train_gen, predict, reconstr_fn, reconstr_noise_shrd, model = theano_fn

    # get different file names for the split data set
    train_files = utils.get_preprocessed_files(train_path)
    train_full_files, train_cter_files, train_capt_files = train_files

    valid_files = utils.get_preprocessed_files(valid_path)
    valid_full_files, valid_cter_files, valid_capt_files = valid_files

    NB_TRAIN_FILES = len(train_full_files)
    NB_VALID_FILES = len(valid_full_files)

    print 'Starting training...'

    train_loss = []

    if args.reload is not None:
        discriminator, generator = model
        file_discr = 'discrminator_epoch_%s.pkl' % RELOAD_ID
        file_gen = 'generator_epoch_%s.pkl' % RELOAD_ID
        loaded_discr = utils.reload_model(args, discriminator, file_discr, RELOAD_SRC)
        loaded_gen = utils.reload_model(args, generator, file_gen, RELOAD_SRC)

    for i in xrange(NB_EPOCHS):

        print 'Epoch #%s of %s' % ((i + 1), NB_EPOCHS)

        epoch_acc = 0
        epoch_loss = 0
        num_batch = 0
        t_epoch = time.time()
        d_batch_loss = 0
        g_batch_loss = 0
        steps_loss_g = []  # will store every loss of generator
        steps_loss_d = []  # will store every loss of discriminator
        d_train_step = 0

        # iterate of split datasets
        for file_id in np.random.choice(NB_TRAIN_FILES, NB_TRAIN_FILES, replace=False):

            t_load = time.time()

            # load file with full image
            with open(train_full_files[file_id], 'r') as f:
                train_full = np.load(f).astype(theano.config.floatX)

            if args.captions:
                # load file with the captions
                with open(train_capt_files[file_id], 'rb') as f:
                    train_capt = pkl.load(f)

            if args.verbose:
                print 'file %s loaded in %s sec' % (train_full_files[file_id], round(time.time() - t_load, 0))

            # iterate over minibatches for training
            schemes_train = ShuffledScheme(examples=len(train_full),
                                           batch_size=BATCH_SIZE)

            for batch_idx in schemes_train.get_request_iterator():

                d_train_step += 1

                t_batch = time.time()
                # generate batch of uniform samples
                rdm_d = np.random.uniform(-1., 1., size=(len(batch_idx), 100))
                rdm_d = rdm_d.astype(theano.config.floatX)

                # train with a minibatch on discriminator
                if args.captions:
                    # generate embeddings for the batch
                    d_capts_batch = utils.captions_to_embedded_matrix(embedding_model, batch_idx, train_capt)

                    d_batch_loss = train_discr(train_full[batch_idx], rdm_d, d_capts_batch)

                else:
                    d_batch_loss = train_discr(train_full[batch_idx], rdm_d)

                steps_loss_d.append(d_batch_loss)
                steps_loss_g.append(g_batch_loss)

                if num_batch % BATCH_PRINT_DELAY == 0:
                    print '- train discr batch %s, loss %s in %s sec' % (num_batch, np.round(d_batch_loss, 4),
                                                                         np.round(time.time() - t_batch, 2))

                # check if it is time to train the generator
                if d_train_step >= TRAIN_STEPS_DISCR:

                    # reset discriminator step counter
                    d_train_step = 0

                    # train the generator for given number of steps
                    for _ in xrange(TRAIN_STEPS_GEN):

                        # generate batch of uniform samples
                        rdm_g = np.random.uniform(-1., 1., size=(BATCH_SIZE, 100))
                        rdm_g = rdm_g.astype(theano.config.floatX)

                        # train with a minibatch on generator
                        if args.captions:
                            # sample a random set of captions from current training file
                            g_batch_idx = np.random.choice(len(train_full), BATCH_SIZE, replace=False)
                            g_capts_batch = utils.captions_to_embedded_matrix(embedding_model, g_batch_idx, train_capt)

                            g_batch_loss = train_gen(rdm_g, g_capts_batch)
                        else:
                            g_batch_loss = train_gen(rdm_g)

                        steps_loss_d.append(d_batch_loss)
                        steps_loss_g.append(g_batch_loss)

                        if num_batch % BATCH_PRINT_DELAY == 0:
                            print '- train gen step %s, loss %s' % (_ + 1, np.round(g_batch_loss, 4))

                epoch_loss += d_batch_loss + g_batch_loss
                num_batch += 1

        train_loss.append(np.round(epoch_loss, 4))

        if args.save > 0 and i % args.save == 0:
            discriminator, generator = model
            utils.save_model(args, discriminator, 'discrminator_epoch_%s.pkl' % i)
            utils.save_model(args, generator, 'generator_epoch_%s.pkl' % i)

        print '- Epoch train (loss %s) in %s sec' % (train_loss[i], round(time.time() - t_epoch))

        # save losses at each step
        utils.dump_objects_output(args, (steps_loss_d, steps_loss_g), 'steps_loss_epoch_%s.pkl' % i)

    print 'Training completed.'

    # Generate images out of pure noise with random captions (if applicable from valid)
    if NB_GEN > 0:

        if args.reload is not None:
            assert loaded_gen and loaded_discr, 'An error occured during loading, cannot generate.'
            save_code = 'rload_%s_%s' % (RELOAD_SRC, RELOAD_ID)
        else:
            save_code = 'no_reload'

        rdm_noise = np.random.uniform(-1., 1., size=(NB_GEN, 100))
        rdm_noise = rdm_noise.astype(theano.config.floatX)

        # choose random valid file
        file_id = np.random.choice(NB_VALID_FILES, 1)

        # load file
        with open(valid_full_files[file_id], 'r') as f:
            valid_full = np.load(f).astype(theano.config.floatX)

        if args.captions:

            # load file with the captions
            with open(valid_capt_files[file_id], 'rb') as f:
                valid_capt = pkl.load(f)

            # pick a given number of images from that file
            batch_valid = np.random.choice(len(valid_capt), NB_GEN, replace=False)
            captions = utils.captions_to_embedded_matrix(embedding_model, batch_valid, valid_capt)
            # captions = np.empty((NB_GEN, 300), dtype=theano.config.floatX)  # used for debugging

            # make predictions
            imgs_noise, probs_noise = predict(rdm_noise, captions)
        else:
            # make predictions
            imgs_noise, probs_noise = predict(rdm_noise)

        if args.verbose:
            print probs_noise

        # save images
        true_imgs = valid_full[batch_valid]

        utils.save_pics_gan(args, imgs_noise, 'noise_caption_%s' % args.captions + save_code, show=False, save=True, tanh=False)
        utils.save_pics_gan(args, true_imgs, 'true_caption_%s' % args.captions + save_code, show=False, save=True, tanh=False)

        if args.captions:
            utils.save_captions(args, save_code, valid_capt, batch_valid)

    if args.mila:
        utils.move_results_from_local()


if __name__ == '__main__':
    main()
