# -*- coding: utf-8 -*-
import sys

sys.path.append("/home/ypy/projects/CoAlign")
import argparse
import os
import time
from typing import OrderedDict

import torch
import open3d as o3d
from torch.utils.data import DataLoader
import numpy as np

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis

torch.multiprocessing.set_sharing_strategy('file_system')


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    # parser.add_argument('--model_dir', type=str, required=True,
    #                     help='Continued training path')

    parser.add_argument('--qkv', default='',
                        help='mark this process')

    parser.add_argument('--model_dir', type=str,
                        default="/home/ypy/projects/CoAlign/opencood/logs/test_dataa_info",
                        help='Continued training path')

    parser.add_argument('--also_laplace', action='store_true',
                        help="whether to use laplace to simulate noise. Otherwise Gaussian")
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='save how many numbers of visualization result?')
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single']

    hypes = yaml_utils.load_yaml(None, opt)

    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    left_hand = True if ("OPV2V" in hypes['test_dir'] or 'V2XSET' in hypes['test_dir']) else False

    if 'dair_v2x' in hypes['test_dir']:
        opt.also_laplace = True

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    torch.cuda.set_device("cuda:1")
    device = torch.device('cuda:1')


    print('Loading Model from checkpoint')
    saved_path = opt.model_dir


    # opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # data_loader = DataLoader(opencood_dataset,
    #                          batch_size=1,
    #                          num_workers=0,
    #                          collate_fn=opencood_dataset.collate_batch_test,
    #                          shuffle=False,
    #                          # shuffle=True,
    #                          pin_memory=False,
    #                          drop_last=False)

    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    data_loader = DataLoader(opencood_train_dataset,
                              batch_size=1,
                              num_workers=0,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              # shuffle=True,
                              shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    for i, batch_data in enumerate(data_loader):
        print(f"data_{i}")


if __name__ == '__main__':
    main()
