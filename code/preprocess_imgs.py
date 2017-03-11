import numpy as np
import fuel
import utils
import os
import cPickle as pkl
import time
from fuel.schemes import ShuffledScheme

if __name__ == '__main__':

    args = utils.get_args()

    # if running on server (MILA), copy dataset locally
    dataset_path = utils.init_dataset(args, 'mscoco_inpainting')
    train_path = os.path.join(dataset_path, 'preprocessed/train2014')
    valid_path = os.path.join(dataset_path, 'preprocessed/val2014')

    NB_TRAIN = 82782
    NB_VALID = 40504
    MAX_IMGS = 15000

    train_count = 0
    valid_count = 0

    # process for training dataset
    schemes_train = ShuffledScheme(examples=NB_TRAIN,
                                   batch_size=MAX_IMGS)

    for i, batch_idx in enumerate(schemes_train.get_request_iterator()):

        print 'train file %s in progress...' %i

        file_name_full_img = os.path.join(train_path, 'train_%s_full.npy' %i)
        file_name_cter_img = os.path.join(train_path, 'train_%s_cter.npy' %i)
        file_name_capt_dic = os.path.join(train_path, 'train_%s_capt.pkl' %i)

        t = time.time()
        # get large 'batch' of data for saving in file
        full_img, targts, capts, color_count = utils.get_batch_data(
            batch_idx, mscoco=dataset_path, split="train2014", crop=False)

        train_count += color_count
        print 'train file %s in completed.' %i
        print '%s imgs saved, training total is now %s' %(color_count, train_count)
        print 'loaded batch in %s sec' % (time.time() - t)

        # save full images
        with open(file_name_full_img, 'w') as f:
            np.save(f, full_img)

        # save center of images
        with open(file_name_cter_img, 'w') as f:
            np.save(f, targts)

        # save captions
        with open(file_name_capt_dic, 'wb') as f:
            pkl.dump(capts, f)

        t = time.time()
        imgs = np.load(open(file_name_full_img, 'r'))
        cter = np.load(open(file_name_cter_img, 'r'))
        capt = pkl.load(open(file_name_capt_dic, 'rb'))
        print 'loaded array in %s sec' % (time.time() - t)

    # process for valid dataset
    schemes_valid = ShuffledScheme(examples=NB_VALID,
                                   batch_size=MAX_IMGS)

    for i, batch_idx in enumerate(schemes_valid.get_request_iterator()):

        print 'valid file %s in progress...' %i

        file_name_full_img = os.path.join(valid_path, 'valid_%s_full.npy' %i)
        file_name_cter_img = os.path.join(valid_path, 'valid_%s_cter.npy' %i)
        file_name_capt_dic = os.path.join(valid_path, 'valid_%s_capt.pkl' %i)

        t = time.time()
        # get large 'batch' of data for saving in file
        full_img, targts, capts, color_count = utils.get_batch_data(
            batch_idx, mscoco=dataset_path, split="val2014", crop=False)

        valid_count += color_count
        print 'valid file %s in completed.' %i
        print '%s imgs saved, valid total is now %s' %(color_count, valid_count)
        print 'loaded batch in %s sec' % (time.time() - t)

        # save full images
        with open(file_name_full_img, 'w') as f:
            np.save(f, full_img)

        # save center of images
        with open(file_name_cter_img, 'w') as f:
            np.save(f, targts)

        # save captions
        with open(file_name_capt_dic, 'wb') as f:
            pkl.dump(capts, f)

        t = time.time()
        imgs = np.load(open(file_name_full_img, 'r'))
        cter = np.load(open(file_name_cter_img, 'r'))
        capt = pkl.load(open(file_name_capt_dic, 'rb'))
        print 'loaded array in %s sec' % (time.time() - t)
