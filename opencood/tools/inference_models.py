# -*- coding: utf-8 -*-
# Author: ypy

import sys
sys.path.append("/data/gkx/Code")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
import argparse
import os
import time
from typing import OrderedDict
import re
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

    # parser.add_argument('--model_dir', type=str,
    #                     default="/data/gkx/Code/checkpoints/fisrt_no_fuse_best/net_epoch80.pth",
    #                     help='Continued training path')
    
    parser.add_argument('--diff_model_dir', type=str,
                        default="/data/gkx/Code/opencood/logs/opv2v_point_pillar_lidar_early_2025_04_08_14_30_17/net_epoch79.pth",
                        help='Continued training path')

    parser.add_argument("--hypes_yaml", "-y", type=str, default="/data/gkx/Code/opencood/hypes_yaml/opv2v/lidar_only_with_noise/diffusion/pointpillar_early_diff_dec.yaml",
                        help='data generation yaml file needed ')
    
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

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary

   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device("cuda:1")
    # device = torch.device('cuda:1')

    if torch.cuda.is_available():
        model.to(device)

    print('Loading Model from checkpoint')
    # model_path = opt.model_dir
    diff_model_path = opt.diff_model_dir
    
    # 构建模型文件路径
    directory = os.path.dirname(diff_model_path)
    model_name = os.path.splitext(os.path.basename(diff_model_path))[0]
    yaml_save_path = directory + "/" + "".join(model_name) + "_AP030507_.yaml"
    
    print(f"评估结果将保存至: {yaml_save_path}")

    # model = train_utils.load_saved_model_new(model_path, model)
    model = train_utils.load_saved_model_new(diff_model_path, model)
    
    # _,model = train_utils.load_saved_model(model_path, model)

    model.eval()
    print(type(model))
            
    if (str(type(model))[8:-2].split(".")[-1] == "PointPillarSDCoper" or str(type(model))[8:-2].split(".")[
        -1] == "DiscoNetSDCoper" or str(type(model))[8:-2].split(".")[-1] == "SDCoper") and hypes['model']['args']['activate_stage2']:
        model.isTrain = False
    
    # add noise to pose.

    pos_std_list = [0]
    rot_std_list = [0]
    pos_mean_list = [0]
    rot_mean_list = [0]

    if opt.also_laplace:
        use_laplace_options = [False, True]
    else:
        use_laplace_options = [False]

    for use_laplace in use_laplace_options:
        AP30 = []
        AP50 = []
        AP70 = []
        for (pos_mean, pos_std, rot_mean, rot_std) in zip(pos_mean_list, pos_std_list, rot_mean_list, rot_std_list):
            # setting noise
            np.random.seed(303)
            noise_setting = OrderedDict()
            noise_args = {'pos_std': pos_std,
                            'rot_std': rot_std,
                            'pos_mean': pos_mean,
                            'rot_mean': rot_mean}

            noise_setting['add_noise'] = True
            noise_setting['args'] = noise_args

            suffix = ""
            if use_laplace:
                noise_setting['args']['laplace'] = True
                suffix = "_laplace"

            # build dataset for each noise setting
            print('Dataset Building')
            print(f"Noise Added: {pos_std}/{rot_std}/{pos_mean}/{rot_mean}.")
            hypes.update({"noise_setting": noise_setting})
            opencood_dataset = build_dataset(hypes, visualize=True, train=False)
            data_loader = DataLoader(opencood_dataset,
                                        batch_size=1,
                                        num_workers=0,
                                        collate_fn=opencood_dataset.collate_batch_test,
                                        shuffle=False,
                                        # shuffle=True,
                                        pin_memory=False,
                                        drop_last=False)

            # Create the dictionary for evaluation
            result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                            0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                            0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

            noise_level = f"{pos_std}_{rot_std}_{pos_mean}_{rot_mean}_" + opt.fusion_method + suffix + opt.note

            for i, batch_data in enumerate(data_loader):
                print(f"{noise_level}_{i}")
                if batch_data is None:
                    continue
                with torch.no_grad():
                    batch_data = train_utils.to_device(batch_data, device)

                    if opt.fusion_method == 'late':
                        infer_result = inference_utils.inference_late_fusion(batch_data,
                                                                                model,
                                                                                opencood_dataset)
                    elif opt.fusion_method == 'early':
                        print("early")
                        infer_result = inference_utils.inference_early_fusion(batch_data,
                                                                                model,
                                                                                opencood_dataset)
                    elif opt.fusion_method == 'intermediate':
                        # print("mid")
                        infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                                        model,
                                                                                        opencood_dataset)
                    elif opt.fusion_method == 'no':
                        infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                            model,
                                                                            opencood_dataset)
                    elif opt.fusion_method == 'no_w_uncertainty':
                        infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                                            model,
                                                                                            opencood_dataset)
                    elif opt.fusion_method == 'single':
                        infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                            model,
                                                                            opencood_dataset,
                                                                            single_gt=True)
                    else:
                        raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                                    'fusion is supported.')

                    if infer_result is None:
                        continue
                    pred_box_tensor = infer_result['pred_box_tensor']
                    gt_box_tensor = infer_result['gt_box_tensor']
                    pred_score = infer_result['pred_score']

                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                                pred_score,
                                                gt_box_tensor,
                                                result_stat,
                                                0.3)
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                                pred_score,
                                                gt_box_tensor,
                                                result_stat,
                                                0.5)
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                                pred_score,
                                                gt_box_tensor,
                                                result_stat,
                                                0.7)

                torch.cuda.empty_cache()

            ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                                                directory, noise_level)
            AP30.append(ap30)
            AP50.append(ap50)
            AP70.append(ap70)

            dump_dict = {'ap30': AP30, 'ap50': AP50, 'ap70': AP70}
            # 这个也要改,每个model一个
        
            # yaml_save_path =  directory + "/" + "".join(model_name) + '_AP030507_emm_pro.yaml'
            yaml_utils.save_yaml(dump_dict,  yaml_save_path)


if __name__ == '__main__':
    main()
