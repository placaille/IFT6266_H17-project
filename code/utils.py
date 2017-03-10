import os
import glob
import PIL.Image as Image
import cPickle as pkl
import shutil
import numpy as np
import theano

def get_batch_data(batch_idx,
                   # PATH need to be fixed
                   mscoco="/Tmp/inpainting/", split="train2014",
                   caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"):
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
    inputs = np.ndarray((len(batch_idx), 64, 64, 3)).astype(theano.config.floatX)
    targets = np.ndarray((len(batch_idx), 32, 32, 3)).astype(theano.config.floatX)
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
            input[center[0] - 16:center[0] + 16,
                  center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] +
                               16, center[1] - 16:center[1] + 16, :]
            color_count += 1
        else:  # if black and white (do nothing)
            continue

        inputs[color_count - 1] = input
        targets[color_count - 1] = target
        captions[color_count - 1] = caption_dict[cap_id]

    returns = [inputs[:color_count] / 255.,
               targets[:color_count] / 255.,
               captions,
               color_count]

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