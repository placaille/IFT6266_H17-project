import os
import glob
import PIL.Image as Image
import cPickle as pkl
import shutil
import numpy as np
import theano
import lasagne.layers as lyr
import argparse
from distutils.dir_util import copy_tree

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


def get_preprocessed_files(path):
    """
    returns arrays of preprocessed data files
    """

    full_files = np.asarray(sorted(glob.glob(path + '/*_full.npy')))
    cter_files = np.asarray(sorted(glob.glob(path + '/*_cter.npy')))
    capt_files = np.asarray(sorted(glob.glob(path + '/*_capt.pkl')))

    return full_files, cter_files, capt_files


def get_corruption_mask():
    """
    returns the corruption mask (when multiplied by image, makes it corrupted)
    """
    corruption_mask = np.ones(shape=(64, 64)).astype(theano.config.floatX)
    center = (
        int(np.floor(corruption_mask.shape[0] / 2.)),
        int(np.floor(corruption_mask.shape[1] / 2.)))
    corruption_mask[center[0] - 16:center[0] + 16,
                    center[1] - 16:center[1] + 16] = 0
    return corruption_mask


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
        shutil.copytree(src_dir + dataset_name, dst_dir + dataset_name, )
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
                path = os.path.join(os.getcwd(), 'output')
                if not os.path.exists(path):
                    os.makedirs(path)
                true_im.save(os.path.join(path, 'img_epoch_%s_id_%s_true.jpg' % (epoch + 1, i)))
                pred_im.save(os.path.join(path, 'img_epoch_%s_id_%s_pred.jpg' % (epoch + 1, i)))

            if show:

                true_im.show(title='img_epoch_%s_id_%s_true' % (epoch + 1, i))
                pred_im.show(title='img_epoch_%s_id_%s_pred' % (epoch + 1, i))


def save_pics_gan(args, images, save_code, show=False, save=False, tanh=True):
    """
    Generates and/or save image out of array using PIL
    """
    if save or show:
        i = 0
        for img in images:
            i += 1
            if tanh:
                img = (np.transpose(img, axes=(1, 2, 0)) + 1) / 2.0
            else:
                img = np.transpose(img, axes=(1, 2, 0))
            image = Image.fromarray(np.uint8(img * 255))
            if save:

                if args.mila:
                    path = '/Tmp/lacaillp/output/images/'
                elif args.laptop:
                    path = '/Users/phil/output/images/'

                if not os.path.exists(path):
                    os.makedirs(path)
                image.save(os.path.join(path, 'img_%s_id_%s.jpg' % (save_code, i)))

            if show:
                image.show(title='img_%s_id_%s' % (save_code, i))

        if save:
            print 'images were saved to %s' % path


def dump_objects_output(args, object, filename):
    """
    Dumps any object using pickle into ./output/objects_dump/ directory
    """
    if args.mila:
        path = '/Tmp/lacaillp/output/objects_dump/'
    elif args.laptop:
        path = '/Users/phil/output/objects_dump/'

    if not os.path.exists(path):
        os.makedirs(path)

    full_path = os.path.join(path, filename)
    with open(full_path, 'wb') as f:
        pkl.dump(object, f)
    print 'Object saved to file %s' % filename


def save_model(args, network, filename):
    """
    Saves the parameters of a model to a pkl file
    Will try to get save it in './output/saved_models', otherwise will create it
    """
    if args.mila:
        path = '/Tmp/lacaillp/output/saved_models/'
    elif args.laptop:
        path = '/Users/phil/output/saved_models/'

    if not os.path.exists(path):
        os.makedirs(path)

    full_path = os.path.join(path, filename)
    with open(full_path, 'wb') as f:
        pkl.dump(lyr.get_all_param_values(network), f)
    print 'Model saved to file %s' % filename


def reload_model(args, network, filename):
    """
    Returns the network loaded of the parameters
    Will try to get filename in './output/saved_models'
    """
    if args.mila:
        path = '/Tmp/lacaillp/output/saved_models/'
    elif args.laptop:
        path = '/Users/phil/output/saved_models/'

    full_path = os.path.join(path, filename)
    try:
        with open(full_path, 'rb') as f:
            values = pkl.load(full_path)
    except:
        print 'An error occured, model wasn\'t loaded.'
    else:
        lyr.set_all_param_values(network, values)
        print 'Network was successfully loaded from %s' % full_path
        return network


def move_results_from_local():
    """
    Copy results stored on Tmp/lacaillp/output to Tmp/lacaillp/lisatmp3.
    Used at end of run. If successful, deletes the local results
    """

    src_dir = '/Tmp/lacaillp/output/'
    dst_dir = '/data/lisatmp3/lacaillp/output/'

    if not os.path.exists(src_dir):
        print '%s doesn\'t exist, nothing was copied.'
    else:

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        try:
            copy_tree(src_dir, dst_dir)
        except:
            print 'Copy of data wasn\'t successful, local data was not deleted.'
        else:
            print 'Copy of data to %s was successful, local copy was deleted.' % dst_dir
            shutil.rmtree(src_dir)
            print 'Local data was deleted. dir = %s' % src_dir


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
    parser.add_argument('-g', '--gen', help='Number of images to generate',
                        type=int, default=5)
    parser.add_argument('-v', '--verbose', help='High verbose option used for debug or dev',
                        action='store_true')
    parser.add_argument('-s', '--save', help='Nb of epochs between saving model',
                        type=int, default=0)
    parser.add_argument('-r', '--reload', help='Reload previously trained model',
                        type=str, default=None)

    return parser.parse_args()
