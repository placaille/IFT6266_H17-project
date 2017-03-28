#!/usr/bin/env python
# -*- coding: utf-8 -*-

import utils
import dcgan
import theano
import os
import time
import numpy as np


def reconstruct_img(images_full, mask_corr, reconstr_fn, reconstr_noise_shrd):
    """
    Reconstructs the image
    ---
    mask_corr: matrix that is applied to make the image corrupted
    """

    preds = np.ones((images_full.shape[0], 3, 64, 64))
    images_corr = np.product((images_full, mask_corr))

    for i, image_corr in enumerate(images_corr):

        image_corr = np.expand_dims(image_corr, axis=0)

        # set value of noise for new image
        reconstr_noise_shrd.set_value(
            np.random.uniform(-1., 1., size=(1, 100)).astype(theano.config.floatX))

        # 100 epoch on image to find best matching latent variables
        it = 0
        nb_grad_0 = 0
        while True:
            reconstr_out = reconstr_fn(image_corr, mask_corr)
            reconstr_noise, prediction, reconstr_loss, grad = reconstr_out

            if it % 500 == 0:
                print 'image %s - loss iteration %s - %s' % (i+1, it, reconstr_loss)
                print 'image %s - grad iteration %s - %s' % (i+1, it, np.sum(grad))

            if np.abs(np.sum(grad)) < 0.00001:
                nb_grad_0 += 1
                if nb_grad_0 == 5:
                    print 'image %s - loss iteration %s - %s' % (i+1, it, reconstr_loss)
                    print 'image %s - grad iteration %s - %s' % (i+1, it, np.sum(grad))
                    break
            else:
                nb_grad_0 = 0

            it += 1

        preds[i] = prediction[0]

    reconstr_images = mask_corr * images_corr + (1.0 - mask_corr) * preds
    return reconstr_images


def main():
    args = utils.get_args()

    NB_GEN = args.gen  # default 5
    RELOAD_SRC = args.reload[0]
    RELOAD_ID = args.reload[1]

    # if running on server (MILA), copy dataset locally
    dataset_path = utils.init_dataset(args, 'mscoco_inpainting/preprocessed')
    valid_path = os.path.join(dataset_path, 'val2014')

    # build network and get theano functions for training
    theano_fn = dcgan.gen_theano_fn(args)
    train_discr, train_gen, predict, reconstr_fn, reconstr_noise_shrd, model = theano_fn

    # get different file names for the split data set
    valid_files = utils.get_preprocessed_files(valid_path)
    valid_full_files, valid_cter_files, valid_capt_files = valid_files

    NB_VALID_FILES = len(valid_full_files)

    corruption_mask = utils.get_corruption_mask()

    if args.reload is not None:

        # Reload previously saved model
        discriminator, generator = model
        file_discr = 'discrminator_epoch_%s.pkl' % RELOAD_ID
        file_gen = 'generator_epoch_%s.pkl' % RELOAD_ID
        t_load = time.time()
        loaded_discr = utils.reload_model(args, discriminator, file_discr, RELOAD_SRC)
        loaded_gen = utils.reload_model(args, generator, file_gen, RELOAD_SRC)

        if loaded_discr and loaded_gen:

            if args.verbose:
                print 'models loaded in %s sec' % (round(time.time() - t_load, 0))

            # choose random valid file
            file_id = np.random.choice(NB_VALID_FILES, 1)

            # load file
            with open(valid_full_files[file_id], 'r') as f:
                valid_full = np.load(f).astype(theano.config.floatX)

            t_load = time.time()

            if args.verbose:
                print 'file %s loaded in %s sec' % (valid_full_files[file_id], round(time.time() - t_load, 0))

            # pick a given number of images from that file
            batch_valid = np.random.choice(len(valid_full), NB_GEN, replace=False)

            # reconstruct image
            img_uncorrpt = valid_full[batch_valid]
            img_reconstr = reconstruct_img(img_uncorrpt, corruption_mask, reconstr_fn, reconstr_noise_shrd)

            # save images
            utils.save_pics_gan(args, img_reconstr, 'pred_rload_%s_%s' % (RELOAD_SRC, RELOAD_ID), show=False, save=True, tanh=False)
            utils.save_pics_gan(args, img_uncorrpt, 'true_rload_%s_%s' % (RELOAD_SRC, RELOAD_ID), show=False, save=True, tanh=False)

            if args.mila:
                utils.move_results_from_local()

if __name__ == '__main__':
    main()
