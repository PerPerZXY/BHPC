#!/usr/bin/env python
# witten by: Xinrong Hu

import os
import argparse
import torch
from os.path import exists

from trixi.util import Config

from configs.Config import get_config
import configs.Config_mmwhs as config_mmwhs
from datasets.prepare_dataset.preprocessing import preprocess_data
from datasets.prepare_dataset.create_splits import create_splits
from experiments.SegExperiment import SegExperiment
from experiments.MixExperiment import MixExperiment
from datasets.downsanpling_data import downsampling_image

import datetime
import time

import matplotlib
import matplotlib.pyplot as plt

from datasets.prepare_dataset.rearrange_dir import rearrange_dir


def parse_option():
    parser = argparse.ArgumentParser("argument for run segmentation pipeline")

    parser.add_argument("--dataset", type=str, default="hippo")
    parser.add_argument("--train_sample", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("-f", "--fold", type=int, default=1)
    parser.add_argument("--saved_model_path", type=str, default=None)
    parser.add_argument("--freeze_model", action='store_true',
                        help="whether load saved model from saved_model_path")
    parser.add_argument("--load_saved_model", action='store_true',
                        help='whether freeze encoder of the segmenter')

    args = parser.parse_args()
    return args

def training(config):

    if not os.path.exists(os.path.join(config.split_dir, "splits.pkl")):
        create_splits(output_dir=config.split_dir, image_dir=config.data_dir)

    if config.saved_model_path is not None:
        config.load_model = True

    # config.saved_model_path = os.path.abspath('save') + '/SupCon/Hippocampus_models/' \
    #                     + 'SupCon_Hippocampus_resnet50_lr_0.0001_decay_0.0001_bsz_1_temp_0.7_trial_0_cosine/' \
    #                     + 'last.pth'
    print(config)
    exp = MixExperiment(config=config, name=config.name, n_epochs=config.n_epochs,
                        seed=42, append_rnd_to_name=config.append_rnd_string)   # visdomlogger_kwargs={"auto_start": c.start_visdom}

    exp.run()
    exp.run_test(setup=False)


def testing(config):

    c.do_load_checkpoint = True
    c.checkpoint_dir = c.base_dir + '/20210202-064334_Unet_mmwhs' + '/checkpoint/checkpoint_current'

    exp = SegExperiment(config=config, name='unet_test', n_epochs=config.n_epochs,
                        seed=42, globs=globals())
    exp.run_test(setup=True)


if __name__ == "__main__":
    args = parse_option()
    if args.dataset == "mmwhs":
        c = config_mmwhs.get_config()
    elif args.dataset == "hippo" or args.dataset == "Hippocampus":
        c = get_config()
    else:
        exit('the dataset is not supoorted currently')
    c.fold = args.fold
    c.batch_size = args.batch_size
    c.train_sample = args.train_sample
    if args.load_saved_model:
        c.saved_model_path = os.path.abspath('save') + '/SupCon/mmwhs_models/' \
                             + 'SupCon_mmwhs_adam_fold_1_lr_0.0001_decay_0.0001_bsz_4_temp_0.1_train_0.4_block/' \
                             + 'ckpt.pth'

    c.saved_model_path = args.saved_model_path
    c.freeze = args.freeze_model
    training(config=c)

