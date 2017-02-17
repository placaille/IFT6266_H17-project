import os
import sys
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image
import argparse
import shutil
from skimage.transform import resize


def getArgs():
    """
    Returns the arguments passed by command-line
    """
    parser = argparse.ArgumentParser()
    load_src = parser.add_mutually_exclusive_group(required=True)
    load_src.add_argument('-m', '--mila', help='If running from MILA servers',
                          action='store_true')
    load_src.add_argument('-l', '--laptop', help='If running from laptop',
                          action='store_true')

    return parser.parse_args()


def resize_mscoco():
    '''
    function used to create the dataset,
    Resize original MS_COCO Image into 64x64 images
    '''

    # PATH need to be fixed
    data_path = "/datasets/coco/coco/images/train2014"
    save_dir = "/Tmp/64_64/train2014/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    preserve_ratio = True
    image_size = (64, 64)
    # crop_size = (32, 32)

    imgs = glob.glob(data_path + "/*.jpg")

    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        print i, len(imgs), img_path

        if img.size[0] != image_size[0] or img.size[1] != image_size[1]:
            if not preserve_ratio:
                img = img.resize((image_size), Image.ANTIALIAS)
            else:
                # Resize based on the smallest dimension
                scale = image_size[0] / float(np.min(img.size))
                new_size = (
                    int(np.floor(scale * img.size[0])) + 1, int(np.floor(scale * img.size[1]) + 1))
                img = img.resize((new_size), Image.ANTIALIAS)

                # Crop the 64/64 center
                tocrop = np.array(img)
                center = (
                    int(np.floor(tocrop.shape[0] / 2.)), int(np.floor(tocrop.shape[1] / 2.)))
                print tocrop.shape, center, (center[0] - 32, center[0] + 32), (center[1] - 32, center[1] + 32)
                if len(tocrop.shape) == 3:
                    tocrop = tocrop[center[0] - 32:center[0] +
                                    32, center[1] - 32:center[1] + 32, :]
                else:
                    tocrop = tocrop[center[0] - 32:center[0] +
                                    32, center[1] - 32:center[1] + 32]
                img = Image.fromarray(tocrop)

        img.save(save_dir + os.path.basename(img_path))


def show_examples(batch_idx, batch_size,
                  # PATH need to be fixed
                  mscoco="/Tmp/inpainting/", split="train2014",
                  caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"):
    '''
    Show an example of how to read the dataset
    '''

    data_path = os.path.join(mscoco, split)
    caption_path = os.path.join(mscoco, caption_path)
    with open(caption_path) as fd:
        caption_dict = pkl.load(fd)

    print data_path + "/*.jpg"
    imgs = glob.glob(data_path + "/*.jpg")
    batch_imgs = imgs[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    for i, img_path in enumerate(batch_imgs):
        img = Image.open(img_path)
        img_array = np.array(img)

        cap_id = os.path.basename(img_path)[:-4]

        # Get input/target from the images
        center = (
            int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16,
                  center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] +
                               16, center[1] - 16:center[1] + 16, :]
        else:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16,
                  center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] +
                               16, center[1] - 16:center[1] + 16]

        # Image.fromarray(img_array).show()
        Image.fromarray(input).show()
        Image.fromarray(target).show()
        print i, caption_dict[cap_id]


def initDataset(args, dataset_name):
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


if __name__ == '__main__':
    # resize_mscoco()
    args = getArgs()
    dataset_path = initDataset(args, 'mscoco_inpainting')
    show_examples(5, 10, mscoco=dataset_path)
