# -*- coding: utf-8 -*-
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
import glob
from opencood.utils.box_utils import boxes_to_corners_3d
import random
import numpy as np
from numpy import cov, trace, iscomplexobj
from scipy.linalg import sqrtm
from torch.cuda.amp import GradScaler


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
seed_everything(42)  


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")

    parser.add_argument("--hypes_yaml", "-y", type=str, default="",
                        help='data generation yaml file needed ')

    parser.add_argument('--qkv', default='',
                        help='mark this process')
    
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    
    parser.add_argument('--diff_model_dir', default='',
                        help='Continued diffusion training path')

    parser.add_argument('--fusion_method', '-f', default="early",
                        help='passed to inference.')

    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()

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
                              prefetch_factor=2,
                              worker_init_fn=worker_init_fn
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

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    #  Set layers to unfreeze based on different conditions
    if opt.model_dir: 
        trainable_layers = [
            'mdd',
            'roi_head',
            'rmpa',
            'dete_convertor',
            'cls_layers',
            'iou_layers',
            'reg_layers',
            ]
        init_epoch = 10
        diff_epoch = 0 
        print("loading model from", opt.model_dir)
        model = train_utils.load_saved_model_new(opt.model_dir, model)
        
    else:
        trainable_layers = [] 
        init_epoch = 0
    
    if trainable_layers:
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze specified layers
        unfrozen_count = 0
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in trainable_layers):
                param.requires_grad = True
                unfrozen_count += 1
                print(f"Unfrozen layer: {name}")
        
        print(f"Total unfrozen parameter groups:{unfrozen_count}")
    
    # Setup learning rate scheduler
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=diff_epoch if diff_epoch == 0 else diff_epoch,data_loader=train_loader)
    saved_path = train_utils.setup_train(hypes)

    if init_epoch > 0:
        print(f"Resume training from epoch {init_epoch}")
    else:
        print("Training from scratch")

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    # record training
    writer = SummaryWriter(saved_path)
    
    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate
    print("batch size: ", hypes['train_params']['batch_size'])
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0 or ('processed_lidar' in batch_data['ego'] and batch_data['ego']['processed_lidar'] == {}):
                continue
            # the model will be evaluation mode during validation
            model.train()
            # fixed some param
            model.stage1_fix()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch

            ouput_dict = model(batch_data['ego'])
            
            if ouput_dict is None:
                continue
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            criterion.logging(epoch, i, len(train_loader), writer, optimizer = optimizer)

            if supervise_single_flag:
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single")
                criterion.logging(epoch, i, len(train_loader), writer,  optimizer = optimizer, suffix="_single")

            # back-propagation
            # without enough data, should'n pass gd_fn
            should_backward = True
            if (str(type(model))[8:-2].split(".")[-1] == "PointPillarBaselineCompare" or 
                str(type(model))[8:-2].split(".")[-1] == "DiscoNetCompare" or 
                str(type(model))[8:-2].split(".")[-1] == "FpvrcnnCoBevt2" or 
                str(type(model))[8:-2].split(".")[-1] == "FpvrcnnCoBevt256" or 
                str(type(model))[8:-2].split(".")[-1] == "FpvrcnnCoBevtCompare") and hypes['model']['args']['activate_stage2']:
                if ouput_dict["det_scores_fused"] is None or sum([t.shape[0] for t in ouput_dict["det_scores_fused"]]) == 0:
                    should_backward = False

            if should_backward:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
    
        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))

        opencood_train_dataset.reinitialize()
        torch.cuda.empty_cache()
        

    print('Training Finished, checkpoints saved to %s' % saved_path)

    run_test = False
    # ddp training may leave multiple bestval
    bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))
    
    if len(bestval_model_list) > 1:
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




if __name__ == '__main__':
    main()
