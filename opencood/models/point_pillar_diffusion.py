# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn> Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import numpy as np

from opencood.models.sub_modules.pillar_vfe_nonet import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.roi_pooling import ROIPooling
from opencood.models.mdd_modules.radar_cond_diff_denoise import Cond_Diff_Denoise
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from opencood.utils import common_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.augmentor.augment_utils import global_rotation

class PointPillarDiffusion(nn.Module):
    def __init__(self, args):
        super(PointPillarDiffusion, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.mdd = Cond_Diff_Denoise(args['mdd_block'], 32)
        self.max_hwl = args['max_hwl']
        self.voxel_preprocessor = build_preprocessor(args['preprocess'], train=True)
        self.out_channel = args['mdd_block']['model']['out_ch']
        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'], # 384
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'], # 384
                                  kernel_size=1)
        

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        voxel_gt_mask = data_dict['processed_lidar']['gt_masks']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'voxel_gt_mask': voxel_gt_mask}
        # 对每辆车进行处理
        # 得到低层BEV特征 [B,C,H,W] 每辆车的低层BEV特征维度一样吗？
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict) # torch.Size([B, 10, 24, 28])       
        # 将gt抠出来的bev特征输入到mdd中
        # batch_dict['spatial_features'] = torch.randn(1, 10, 50, 50).to(voxel_features.device)
        batch_dict = self.mdd(batch_dict)
        # ！！V2X-R没有单独训练diffusion，是整体一起训的
        output_dict = {'pred_feature' : batch_dict['pred_feature'], 
                       'gt_feature' : batch_dict['batch_gt_spatial_features']}
        
        # batch_dict = self.scatter(batch_dict)
        # batch_dict = self.backbone(batch_dict)

        # spatial_features_2d = batch_dict['spatial_features_2d']

        # if self.shrink_flag:
        #     spatial_features_2d = self.shrink_conv(spatial_features_2d)

        # psm = self.cls_head(spatial_features_2d)
        # rm = self.reg_head(spatial_features_2d)

        # output_dict = {'cls_preds': psm,
        #                'reg_preds': rm}
                       
        # if self.use_dir:
        #     dm = self.dir_head(spatial_features_2d)
        #     output_dict.update({'dir_preds': dm})

        return output_dict