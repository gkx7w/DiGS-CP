# -*- coding: utf-8 -*-

import sys
sys.path.append("/home/ubuntu/Code2")

import argparse
import os
import statistics

import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
import glob
from opencood.utils.box_utils import boxes_to_corners_3d
import random
import numpy as np

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

# 设置随机种子以确保确定性
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    
seed_everything(42)  # 固定种子

# disconet不能用这个文件训练！
def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")

    # V2VNET
    parser.add_argument("--hypes_yaml", "-y", type=str, default="/home/ubuntu/Code2/opencood/hypes_yaml/opv2v/lidar_only_with_noise/diffusion/pointpillar_early_diff_dec.yaml",
                        help='data generation yaml file needed ')

    parser.add_argument('--qkv', default='',
                        help='mark this process')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')

    parser.add_argument('--fusion_method', '-f', default="early",
                        help='passed to inference.')

    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()

    #  resume
    # hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    #  start
    hypes = yaml_utils.load_resume_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=4,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                            #   shuffle=True,
                              shuffle=False,

                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2
                              )
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=4,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device("cuda:0")
    # device = torch.device('cuda:0')

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        saved_path = train_utils.setup_train(hypes)
        print(f"resume from {init_epoch} epoch.")

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0 or batch_data['ego']['processed_lidar'] == {}:
                continue
            # the model will be evaluation mode during validation
            model.train()
            # opencood.models.disconet_compare.DiscoNetCompare
            # opencood.models.point_pillar_baseline_compare.PointPillarBaselineCompare
            # print("model name: ",type(model))

            if (str(type(model))[8:-2].split(".")[-1] == "PointPillarSDCoper" or str(type(model))[8:-2].split(".")[
                -1] == "DiscoNetSDCoper" or str(type(model))[8:-2].split(".")[-1] == "SDCoper") and \
                    hypes['model']['args']['activate_stage2']:
                pass
                # fixed some param
                # model.stage1_fix()

            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            
            ouput_dict = model(batch_data['ego'])
            
            # 可视化
            # if i > 6760:
            #     gt_bev_feature = ouput_dict['gt_feature'][0]
            #     pre_bev_feature = ouput_dict['pred_feature'][0]
            #     # 可视化gt bev
            #     visualize_averaged_channels_individual(gt_bev_feature, f"/home/ubuntu/Code2/opencood/bev_visualizations/gt_bev_{i}")
            #     # 可视化预测bev
            #     visualize_averaged_channels_individual(pre_bev_feature, f"/home/ubuntu/Code2/opencood/bev_visualizations/pre_bev_{i}")
            
            
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            criterion.logging(epoch, i, len(train_loader), writer)

            if supervise_single_flag:
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single")
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            # back-propagation
            # without enough data, should'n pass gd_fn
            should_backward = True
            if (str(type(model))[8:-2].split(".")[-1] == "PointPillarBaselineCompare" or str(type(model))[8:-2].split(".")[-1] == "DiscoNetCompare" or str(type(model))[8:-2].split(".")[-1] == "FpvrcnnCoBevt2" or str(type(model))[8:-2].split(".")[-1] == "FpvrcnnCoBevt256" or str(type(model))[8:-2].split(".")[-1] == "FpvrcnnCoBevtCompare") and hypes['model']['args']['activate_stage2']:
                if ouput_dict["det_scores_fused"] is None or sum([t.shape[0] for t in ouput_dict["det_scores_fused"]]) == 0:
                    should_backward = False

                # 这个问题没有完全解决，今晚再看看
                # print("现有的框的数量：",sum([t.shape[0] for t in ouput_dict["det_scores_fused"]]))
            if should_backward:
                final_loss.backward()
                optimizer.step()


        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step(epoch)

        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    run_test = False
    # ddp training may leave multiple bestval
    bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))
    
    if len(bestval_model_list) > 1:
        import numpy as np
        bestval_model_epoch_list = [eval(x.split("/")[-1].lstrip("net_epoch_bestval_at").rstrip(".pth")) for x in bestval_model_list]
        ascending_idx = np.argsort(bestval_model_epoch_list)
        for idx in ascending_idx:
            if idx != (len(bestval_model_list) - 1):
                os.remove(bestval_model_list[idx])

    if run_test:
        fusion_method = opt.fusion_method
        if 'noise_setting' in hypes and hypes['noise_setting']['add_noise']:
            cmd = f"python opencood/tools/inference_w_noise.py --model_dir {saved_path} --fusion_method {fusion_method}"
        else:
            cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)


def visualize_bev_features(bev_feature, save_dir='./bev_visualizations', n_cols=5):
    """
    Visualize Bird's Eye View (BEV) features.
    
    Args:
        bev_feature: Tensor of shape [N, C, H, W]
        save_dir: Directory to save visualizations
        n_cols: Number of columns in the grid plot
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    N, C, H, W = bev_feature.shape
    
    # Create a figure for each of the N BEV features
    for n in range(N):
        # Get the current BEV feature
        feature = bev_feature[n]  # Shape: [C, H, W]
        
        # Calculate how many rows we need
        n_rows = (C + n_cols - 1) // n_cols
        
        # Create a figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        fig.suptitle(f'BEV Feature {n+1}/{N}', fontsize=16)
        
        # Flatten axes for easy indexing if multiple rows
        if n_rows > 1:
            axes = axes.flatten()
        
        # Loop through each channel
        for c in range(C):
            # Get the current channel
            channel_data = feature[c].detach().cpu().numpy()  # Shape: [H, W]
            
            # Normalize data for better visualization
            norm = Normalize(vmin=channel_data.min(), vmax=channel_data.max())
            
            # Get the corresponding axis
            if C <= n_cols and n_rows == 1:
                ax = axes[c] if n_cols > 1 else axes
            else:
                ax = axes[c]
            
            # Plot the feature as a heatmap
            im = ax.imshow(channel_data, cmap='viridis', norm=norm)
            ax.set_title(f'Channel {c}')
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide empty subplots
        for c in range(C, n_rows * n_cols):
            if C <= n_cols and n_rows == 1:
                if c < n_cols:  # Check if we're still within the valid range
                    if n_cols > 1:
                        axes[c].axis('off')
            else:
                axes[c].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{save_dir}/bev_feature_{n+1}.png', dpi=200)
        plt.close(fig)


def visualize_averaged_channels_individual(bev_feature, save_dir='./bev_avg_viz'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    N, C, H, W = bev_feature.shape
    
    for n in range(N):
        # Average across channels
        feature_avg = torch.mean(bev_feature[n], dim=0).detach().cpu().numpy()
        
        # Create a new figure for each BEV
        fig, ax = plt.subplots(figsize=(6, 6))
        
        norm = Normalize(vmin=feature_avg.min(), vmax=feature_avg.max())
        im = ax.imshow(feature_avg, cmap='viridis', norm=norm)
        ax.set_title(f'BEV {n+1} (avg)')
        ax.axis('off')
        fig.colorbar(im, ax=ax)
        
        # Save individual figure
        plt.tight_layout()
        plt.savefig(f'{save_dir}/bev_{n+1}_avg_channels.png', dpi=200)
        plt.close(fig)


# def visualize_averaged_channels(bev_feature, save_dir='./bev_avg_viz'):
#     """
#     Visualize the average of all channels for each BEV feature.
    
#     Args:
#         bev_feature: Tensor of shape [N, C, H, W]
#         save_dir: Directory to save visualizations
#     """
#     import os
#     os.makedirs(save_dir, exist_ok=True)
    
#     N, C, H, W = bev_feature.shape
    
#     # Create a grid to display all N BEV averages
#     n_cols = min(7, N)
#     n_rows = (N + n_cols - 1) // n_cols
    
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
#     if n_rows > 1 or n_cols > 1:
#         axes = axes.flatten()
    
#     for n in range(N):
#         # Average across channels
#         feature_avg = torch.mean(bev_feature[n], dim=0).detach().cpu().numpy()
        
#         # Get corresponding axis
#         if N == 1:
#             ax = axes
#         else:
#             ax = axes[n]
        
#         norm = Normalize(vmin=feature_avg.min(), vmax=feature_avg.max())
#         im = ax.imshow(feature_avg, cmap='viridis', norm=norm)
#         ax.set_title(f'BEV {n+1} (avg)')
#         ax.axis('off')
#         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
#     # Hide empty subplots
#     for n in range(N, n_rows * n_cols):
#         if N > 1:  # Only if axes is a collection
#             axes[n].axis('off')
    
#     plt.tight_layout()
#     plt.savefig(f'{save_dir}/averaged_channels_all_bevs.png', dpi=200)
#     plt.close(fig)



if __name__ == '__main__':
    main()
