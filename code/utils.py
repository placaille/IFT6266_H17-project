import os
import glob
import PIL.Image as Image
import cPickle as pkl
import shutil
import numpy as np
import theano
import argparse

def get_batch_data(batch_idx,
                   # PATH need to be fixed
                   mscoco="/Tmp/inpainting/", split="train2014",
                   caption_path="dict_key_imgID_value_caps_train_and_valid.pkl",
                   crop=True):
    '''
    Show an example of how to read the dataset
    @return inputs, targets, captions, color_count
    '''

    data_path = os.path.join(mscoco, split)
    caption_path = os.path.join(mscoco, caption_path)
    with open(caption_path) as fd:
        caption_dict = pkl.load(fd)

    # print data_path + "/*.jpg"
    imgs = np.asarray(glob.glob(data_path + "/*.jpg"))
    batch_imgs = imgs[batch_idx]

    color_count = 0
    inputs = np.ndarray((len(batch_idx), 3, 64, 64))
    targets = np.ndarray((len(batch_idx), 3, 32, 32))
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
            if crop:
                input[center[0] - 16:center[0] + 16,
                      center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] +
                               16, center[1] - 16:center[1] + 16, :]
            color_count += 1
        else:  # if black and white (do nothing)
            continue

        inputs[color_count - 1] = np.transpose(input, (2, 0, 1))
        targets[color_count - 1] = np.transpose(target, (2, 0, 1))
        captions[color_count - 1] = caption_dict[cap_id]

    returns = [inputs[:color_count] / 255.,
               targets[:color_count] / 255.,
               captions,
               color_count]

    return returns


def load_preprocessed_info(path):

    full_files = np.asarray(sorted(glob.glob(path + '/*_full.npy')))
    cter_files = np.asarray(sorted(glob.glob(path + '/*_cter.npy')))
    capt_files = np.asarray(sorted(glob.glob(path + '/*_capt.pkl')))

    full_nb = []
    cter_nb = []
    capt_nb = []

    for full, cter, capt in zip(full_files, cter_files, capt_files):
        full_nb = np.load(open(full, 'r')).shape[0]
        cter_nb = np.load(open(cter, 'r')).shape[0]
        capt_nb = len(pkl.load(open(capt, 'rb')))

        assert full_nb == cter_nb == capt_nb, 'nb of elements don\'t match (full%s, cter%s, capt%s)' %(full_nb, cter_nb, capt_nb)

    return (full_files, full_nb), (cter_files, cter_nb), (capt_files, capt_nb)


def get_preprocessed_batch_data(batch_idx,
                   # PATH need to be fixed
                   path="/Tmp/inpainting/", split="train2014"):
    '''
    Show an example of how to read the dataset
    @return inputs, targets, captions, color_count
    '''

    data_path = os.path.join(path, split)

    # print data_path + "/*.jpg"
    imgs = np.asarray(glob.glob(data_path + "/*.jpg"))
    batch_imgs = imgs[batch_idx]



    return returns



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

def gen_pics(inputs, targts, preds, epoch, show=False, save=False):
    """
    Generates and/or save image out of array using PIL
    """
    if save or show:

        i = 0
        for input, targt, pred in zip(inputs, targts, preds):

            i += 1
            # set contour so we can fill the middle part with true and prediction
            true_im = np.copy(input)
            pred_im = np.copy(input)

            center = (int(np.floor(true_im.shape[0] / 2.)), int(np.floor(true_im.shape[1] / 2.)))

            true_im[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = targt
            pred_im[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = pred

            true_im = Image.fromarray(np.uint8(true_im * 255))
            pred_im = Image.fromarray(np.uint8(pred_im * 255))

            if save:

                true_im.save('./output/img_epoch_%s_id_%s_true.jpg' % (epoch + 1, i))
                pred_im.save('./output/img_epoch_%s_id_%s_pred.jpg' % (epoch + 1, i))

            if show:

                true_im.show(title='img_epoch_%s_id_%s_true' % (epoch + 1, i))
                pred_im.show(title='img_epoch_%s_id_%s_pred' % (epoch + 1, i))


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
                        type=int, default=25)
    parser.add_argument('-g', '--gen', help='Number of images to generate from valid',
                        type=int, default=5)
    parser.add_argument('-v', '--verbose', help='High verbose option used for debug or dev',
                        action='store_true')

    return parser.parse_args()
