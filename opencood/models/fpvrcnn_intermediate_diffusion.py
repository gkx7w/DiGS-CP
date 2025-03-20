# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.cia_ssd_utils import SSFA, Head
from opencood.models.sub_modules.vsa import VoxelSetAbstraction
from opencood.models.sub_modules.roi_head import RoIHead
from opencood.models.sub_modules.matcher import Matcher
from opencood.data_utils.post_processor.fpvrcnn_postprocessor import \
    FpvrcnnPostprocessor
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.mdd_modules.radar_cond_diff_denoise import Cond_Diff_Denoise
from opencood.models.sub_modules.factors_encoder import Encoder4

class FPVRCNNIntermediateDiffusion(nn.Module):
    def __init__(self, args):
        super(FPVRCNNIntermediateDiffusion, self).__init__()
        lidar_range = np.array(args['lidar_range'])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) /
                             np.array(args['voxel_size'])).astype(np.int64)
        self.vfe = MeanVFE(args['mean_vfe'],
                           args['mean_vfe']['num_point_features'])
        self.spconv_block = VoxelBackBone8x(args['spconv'],
                                            input_channels=args['spconv'][
                                                'num_features_in'],
                                            grid_size=grid_size)
        self.map_to_bev = HeightCompression(args['map2bev'])
        self.ssfa = SSFA(args['ssfa'])
        self.head = Head(**args['head'])
        self.post_processor = FpvrcnnPostprocessor(args['post_processer'],
                                                   train=True)
        self.vsa = VoxelSetAbstraction(args['vsa'], args['voxel_size'],
                                       args['lidar_range'],
                                       num_bev_features=128,
                                       num_rawpoint_features=3)
        # 解耦+diffusion
        self.facter_enc = Encoder4(d=128,context_dim=16,latent_unit=20)
        self.mdd = Cond_Diff_Denoise(args['mdd_block'], 32)
        
        self.matcher = Matcher(args['matcher'], args['lidar_range'])
        self.roi_head = RoIHead(args['roi_head'])
        self.train_stage2 = args['activate_stage2']
        self.discrete_ratio = args['voxel_size'][0]
        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
                                  kernel_size=1)


    def forward(self, batch_dict):
        voxel_features = batch_dict['processed_lidar']['voxel_features']
        voxel_coords = batch_dict['processed_lidar']['voxel_coords']
        voxel_num_points = batch_dict['processed_lidar']['voxel_num_points']

        # save memory
        batch_dict.pop('processed_lidar')
        batch_dict.update({'voxel_features': voxel_features,
                           'voxel_coords': voxel_coords,
                           'voxel_num_points': voxel_num_points,
                           'batch_size': int(batch_dict['record_len'].sum()),
                           'proj_first': batch_dict['proj_first'],
                           'lidar_pose': batch_dict['lidar_pose']})

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        # print("第一阶段backbone完成")
        out = self.ssfa(batch_dict['spatial_features'])

        batch_dict['stage1_out'] = self.head(out)

        # batch_dict['preds_dict_stage1'] = self.head(out)

        data_dict, output_dict = {}, {}
        data_dict['ego'], output_dict['ego'] = batch_dict, batch_dict
        
        if data_dict['train']:
            # choose gt in stage1
            gt_boxes = [b[m][:, [0, 1, 2, 3, 4, 5, 6]].float() for b, m in
                        zip(batch_dict['object_bbx_center'],
                            batch_dict['object_bbx_mask'].bool())]  # hwl -> lwh order 这里不转换，后面会转换
            
            for i in range(len(gt_boxes)):
                gt_boxes[i][:,3:6] -= 0.5
            gt_score = [torch.ones((t.shape[0]),device = t.device) for t in gt_boxes]
            print("gt shape: ",[t.shape for t in gt_boxes],[t.shape for t in gt_score])
            pred_box3d_list, scores_list = gt_boxes, gt_score
        else:
            # choose output in stage1
            pred_box3d_list, scores_list = \
                self.post_processor.post_process(data_dict, output_dict,
                                                stage1=True)
            
        
        # factor_encoder输入为gt抠出来的bev特征，得到解耦因子 
        batch_dict = self.facter_enc(batch_dict)
        # 解耦因子作为mdd引导条件，训练时以加噪的融合的gt抠出来的bev特征作为初始输入 这部分逻辑应该在mdd中实现
        batch_dict = self.mdd(batch_dict)
        
        batch_dict['fused_features'] = batch_dict['pred_feature'] * (batch_dict['spatial_features'] != 0)
        #vis_feature_denoise(lidar_batch_dict)
        output_dict = {'pred_feature' : batch_dict['spatial_features']}
        
        if data_dict['train']:
            output_dict.update({'gt_feature' : batch_dict['de_spatial_features']})

        batch_dict={
            'spatial_features' : \
            torch.cat([batch_dict['spatial_features'], batch_dict['spatial_features']],dim = 1),
            'record_len': record_len
        } 

        batch_dict = self.backbone(batch_dict)
        
        fused_features = batch_dict['fused_features']
        

        psm = self.cls_head(fused_features)
        rm = self.reg_head(fused_features)    
        output_dict.update({'psm': psm,
                       'rm': rm,})
        return output_dict