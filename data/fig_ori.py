from math import sqrt
import torch

# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD   = (57.38, 57.12, 58.40)
# ----------------------- CONFIG CLASS ----------------------- #

class Config(object):
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)


# Default config
default_cfg = Config({
    'name': 'Base',
    'train_info': '/home/ubuntu/Documents/jx/ref_images/unc/train_batch/unc_train.npy',
    'train_images': '/home/ubuntu/Documents/jx/ref_images/unc/train_batch/',
    'valid_info': '/home/ubuntu/Documents/jx/ref_images/unc/val_batch/unc_val.npy',
    'valid_images': '/home/ubuntu/Documents/jx/ref_images/unc/val_batch/',
    # 'valid_info': '/home/jx/Documents/3st/referring1/Image_referring_datasets/unc/testA_batch/unc_testA.npy',
    # 'valid_images': '/home/jx/Documents/3st/referring1/Image_referring_datasets/unc/testA_batch/',


    # 'valid_info': '/home/jx/Documents/3st/referring1/Image_referring_datasets/unc/testB_batch/unc_testB.npy',
    # 'valid_images': '/home/jx/Documents/3st/referring1/Image_referring_datasets/unc/testB_batch/',

    # 'train_info': '/home/jx/Documents/3st/referring/Image_referring_datasets/unc+/train_batch/unc+_train.npy',
    # 'train_images': '/home/jx/Documents/3st/referring/Image_referring_datasets/unc+/train_batch/',
    # 'valid_info': '/home/jx/Documents/3st/referring/Image_referring_datasets/unc+/val_batch/unc+_val.npy',
    # 'valid_images': '/home/jx/Documents/3st/referring/Image_referring_datasets/unc+/val_batch/',

    # 'train_info': '../datasets/generated_data/unc/train_batch/unc_train.npy',
    # 'train_images': '../datasets/generated_data/unc/train_batch/',
    # 'valid_info': '../datasets/generated_data/unc/val_batch/unc_val.npy',
    # 'valid_images': '../datasets/generated_data/unc/val_batch/',

    # 'valid_info': '../datasets/generated_data/unc/testA_batch/unc_testA.npy',
    # 'valid_images': '../datasets/generated_data/unc/testA_batch/',
    # 'valid_info': '../datasets/generated_data/unc/testB_batch/unc_testB.npy',
    # 'valid_images': '../datasets/generated_data/unc/testB_batch/',

    # 'train_info': '../datasets/generated_data/unc+/train_batch/unc+_train.npy',
    # 'train_images': '../datasets/generated_data/unc+/train_batch/',
    # 'valid_info': '../datasets/generated_data/unc+/val_batch/unc+_val.npy',
    # 'valid_images': '../datasets/generated_data/unc+/val_batch/',
    # 'valid_info': '/home/jx/Documents/3st/referring/Image_referring_datasets/unc+/testA_batch/unc+_testA.npy',
    # 'valid_images': '/home/jx/Documents/3st/referring/Image_referring_datasets/unc+/testA_batch/',
    # 'valid_info': '/home/jx/Documents/3st/referring/Image_referring_datasets/unc+/testB_batch/unc+_testB.npy',
    # 'valid_images': '/home/jx/Documents/3st/referring/Image_referring_datasets/unc+/testB_batch/',

    # 'train_info': '../datasets/generated_data/Gref/train_batch/Gref_train.npy',
    # 'train_images': '../datasets/generated_data/Gref/train_batch/',
    # 'valid_info': '../datasets/generated_data/Gref/val_batch/Gref_val.npy',
    # 'valid_images': '../datasets/generated_data/Gref/val_batch/',

    # 'train_info': '../datasets/generated_data/referit/trainval_batch/referit_trainval.npy',
    # 'train_images': '../datasets/generated_data/referit/trainval_batch/',
    # 'valid_info': '../datasets/generated_data/referit/test_batch/referit_test.npy',
    # 'valid_images': '../datasets/generated_data/referit/test_batch/',

    # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
    'preserve_aspect_ratio': False,
    'has_gt': True,

    # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
    # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
    # If not specified, this just assumes category ids start at 1 and increase sequentially.
    'label_map': None,

    'lstm': False, # False for bert
    'd_length': False,#True,   testing length about word

    'lr': 5e-4,
    'max_iter': 375000,
    'lr_steps': (100000, 200000, 375000),
    # 'max_iter': 600000,#600000
    # 'lr_steps': (100000, 200000, 300000, 400000, 500000, 600000),
    'momentum': 0.9,
    'decay': 5e-4,
    # For each lr step, what to multiply the lr with
    'gamma': 0.1,
    'max_size': 320,
    'vit_img_size_train': 384,
    'vit_img_size_test': 384, 
    'freeze_bn': False,
    'delayed_settings': [],
    # Initial learning rate to linearly warmup from (if until > 0)
    'lr_warmup_init': 1e-5,

    # If > 0 then increase the lr linearly from warmup_init to lr each iter for until iters
    'lr_warmup_until': 1500,

    'pretrained_model': 'classification'  # segmentation or classification
})
cfg = default_cfg
def set_cfg(config_name: str):
    """ Sets the active config. Works even if cfg is already imported! """
    global cfg

    # Note this is not just an eval because I'm lazy, but also because it can
    # be used like ssd300_config.copy({'max_size': 400}) for extreme fine-tuning
    cfg.replace(eval(config_name))

    if cfg.name is None:
        cfg.name = config_name.split('_config')[0]


def set_dataset(dataset_name: str):
    """ Sets the dataset of the current config. """
    cfg.dataset = eval(dataset_name)

