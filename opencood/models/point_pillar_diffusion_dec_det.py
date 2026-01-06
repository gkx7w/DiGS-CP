# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import time
from opencood.models.sub_modules.pillar_vfe_dec_diff import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter_diffusion import PointPillarScatter
from opencood.models.sub_modules.point_pillar_scatter_dec import PointPillarScatter as PointPillarScatter_dec
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.diff_modules.cond_diff_denoise import Cond_Diff_Denoise
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from opencood.utils import common_utils
from opencood.tools import train_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.augmentor.augment_utils import global_rotation
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.common_utils import merge_features_to_dict
from opencood.models.sub_modules.cia_ssd_utils import SSFA, Head
from opencood.models.sub_modules.rmpa import Resolutionaware_Multiscale_Progressive_Attention
from opencood.models.sub_modules.roi_head_with_diff import RoIHead
from opencood.models.sub_modules.matcher import Matcher
from opencood.data_utils.post_processor.fpvrcnn_postprocessor import \
    FpvrcnnPostprocessor
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, When2commFusion
from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor
from opencood.data_utils.datasets.early_fusion_dataset_diffusion import visualize_gt_boxes
from opencood.visualization.simple_vis import visualize_bev_features_custom

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x


def normalized_affine_bev(bev,normalized_affine_matrix,record_len):
    _, C, H, W = bev.shape
    B, L = normalized_affine_matrix.shape[:2]
    split_x = regroup(bev, record_len)
    batch_node_features = split_x
    out = []

    for b in range(B):
        # if i==0 and b == 0:
        #     plot_one_BEV(batch_node_features[b],"before   project  bs"+str(b))

        N = record_len[b]
        t_matrix = normalized_affine_matrix[b][:N, :N, :, :]
        # update each node i
        ego = 0  # ego
        neighbor_feature = warp_affine_simple(batch_node_features[b],
                                              t_matrix[ego, :, :, :],
                                              (H, W))

        # if i==0 and b == 0:
        #     plot_one_BEV(neighbor_feature,"after   project   bs"+str(b))

        out.append(neighbor_feature)
    out = torch.cat(out, dim=0)
    return out

class PointPillarDiffusionDecDet(nn.Module):
    def __init__(self, args):
        super(PointPillarDiffusionDecDet, self).__init__()

        lidar_range = np.array(args['lidar_range'])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) /
                             np.array(args['voxel_size'])).astype(np.int64)
        self.batch_len = args['N']
        self.max_hwl = args['max_hwl']
        self.voxel_size = args['voxel_size']
        self.train_stage2 = args['activate_stage2']
        self.is_inference = args.get('is_inference', False)
        self.use_diff = args.get('use_diffusion', False)
        
        self.pre_processor = SpVoxelPreprocessor(args["preprocess"], train = True)
        self.post_processor = FpvrcnnPostprocessor(args['post_processer'], train=True)
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.scatter_dec = PointPillarScatter_dec(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", True)  # default true
        if is_resnet:
            print("is_resnet")
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'],
                                              64)  # or you can use ResNetBEVBackbone, which is stronger
        else:
            print("not_resnet")
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'],
                                            64)  
        self.voxel_preprocessor = build_preprocessor(args['preprocess'], train=True)
        self.fusion_net = nn.ModuleList()
        for i in range(len(args['base_bev_backbone']['layer_nums'])):
            if args['fusion_method'] == "max":
                self.fusion_net.append(MaxFusion())
            if args['fusion_method'] == "att":
                self.fusion_net.append(AttFusion(args['att']['feat_dim'][i]))

        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(64, args['compression'])

        self.head = Head(**args['head'])
        self.rmpa = Resolutionaware_Multiscale_Progressive_Attention(args['vsa'], args['voxel_size'],
                                       args['lidar_range'],
                                       num_bev_features=128,
                                       num_rawpoint_features=3)

        self.matcher = Matcher(args['matcher'], args['lidar_range'])
        self.roi_head = RoIHead(args['roi_head'])
        self.roi_head.decoupling = True
        
        self.mdd = Cond_Diff_Denoise(args['mdd_block'])
    
        pre_channel = args['mdd_block']['model']['d_cond']
        fc_layers = [pre_channel, pre_channel//2]
        self.dp_ratio = args['roi_head']['dp_ratio']
        
        hidden_dim = args['dete_hidden']
        self.dete_convertor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        self.cls_layers, _ = self._make_fc_layers(hidden_dim, fc_layers,
                                                            output_channels=args['roi_head']['num_cls'])
        self.iou_layers, _ = self._make_fc_layers(hidden_dim, fc_layers,
                                                  output_channels=args['roi_head']['num_cls'])
        self.reg_layers, _ = self._make_fc_layers(hidden_dim, fc_layers,
                                                  output_channels=args['roi_head']['num_cls'] * 7)
        self._init_weights(weight_init='xavier')


    def _init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)
    
    def _make_fc_layers(self, input_channels, fc_list, output_channels=None):
        fc_layers = []
        pre_channel = input_channels
        for k in range(len(fc_list)):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                # nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.dp_ratio > 0:
                fc_layers.append(nn.Dropout(self.dp_ratio))
        if output_channels is not None:
            fc_layers.append(
                nn.Conv1d(pre_channel, output_channels, kernel_size=1,
                          bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers, pre_channel    
    
    def stage1_fix(self):
        # 固定参数
        modules_to_fix = [
            self.pillar_vfe, self.scatter_dec, self.backbone, self.head,
        ] #self.rmpa, self.roi_head, self.scatter, self.mdd
        
        if self.compression:
            modules_to_fix.append(self.naive_compressor)
        if self.shrink_flag:
            modules_to_fix.append(self.shrink_conv)
        
        for module in modules_to_fix:
            # 固定BN层
            for submodule in module.modules():
                if isinstance(submodule, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    submodule.eval()
    
    def get_processed_lidar(self, batch_dict):
        processed_lidar_batch = []
        ori_lidar = batch_dict['origin_lidar_for_diff'][:, 1:]
        batch_indices = batch_dict['origin_lidar_for_diff'][:, 0].long()
        for batch_i in range(len(batch_dict['boxes_fused'])):
            bs_mask = (batch_indices == batch_i)
            batch_ori_lidar = ori_lidar[bs_mask].cpu().numpy()  # (N, 4)
            pre_fused_boxes = batch_dict['boxes_fused'][batch_i].cpu().numpy() # (n, 7)
            pre_fused_boxes[:, 3:6] = np.array(self.max_hwl)
            point_indices = points_in_boxes_cpu(batch_ori_lidar[:, :3], pre_fused_boxes[:,[0, 1, 2, 5, 4, 3, 6]]) 
            box_voxel_stack = []
            box_coords_stack = []
            box_num_points_stack = []
            box_masks = []
            rotation_angles = -pre_fused_boxes[:, 6].astype(float)
            for car_idx in range(len(pre_fused_boxes)):
                box_point = batch_ori_lidar[point_indices[car_idx] > 0]
                if len(box_point) < 100:
                    continue 
                box_point[:, :3] -= pre_fused_boxes[car_idx][0:3]
                # 旋转点云 
                box_point = common_utils.rotate_points_along_z(box_point[np.newaxis, :, :], np.array([rotation_angles[car_idx]]))[0]
                # 体素化 
                processed_lidar_car = self.pre_processor.preprocess(box_point, is_car=True) #True
                box_voxel_stack.append(processed_lidar_car['voxel_features'])
                box_coords_stack.append(processed_lidar_car['voxel_coords'])
                box_num_points_stack.append(processed_lidar_car['voxel_num_points'])
                box_masks.append(np.full(processed_lidar_car['voxel_features'].shape[0], car_idx, dtype=np.int32))
            
            if len(box_coords_stack) == 0:
                print(batch_dict['path'])
                processed_lidar = None 
            else:
                processed_lidar = {
                    'voxel_features': np.concatenate(box_voxel_stack, axis=0),
                    'voxel_coords': np.concatenate(box_coords_stack, axis=0),
                    'voxel_num_points': np.concatenate(box_num_points_stack, axis=0),
                    'gt_masks': np.concatenate(box_masks, axis=0),
                    }
            processed_lidar_batch.append(processed_lidar)
        if None in processed_lidar_batch:
            batch_dict.update({'processed_lidar': None})
            return batch_dict
        else:
            processed_lidar_dict = merge_features_to_dict(processed_lidar_batch)
            processed_lidar_torch_dict = \
                            self.pre_processor.collate_batch(processed_lidar_dict)
            processed_lidar_torch_dict = train_utils.to_device(processed_lidar_torch_dict, device = batch_dict['dec_voxel_features'].device)
            batch_dict.update({'processed_lidar': processed_lidar_torch_dict})
            return batch_dict
    
    
    def forward(self, batch_dict):
        dec_voxel_features = batch_dict['dec_processed_lidar']['voxel_features']
        dec_voxel_coords = batch_dict['dec_processed_lidar']['voxel_coords']
        dec_voxel_num_points = batch_dict['dec_processed_lidar']['voxel_num_points']
        record_len = batch_dict['record_len']
    
        batch_dict.pop('dec_processed_lidar')
        batch_dict.update({
                           'dec_voxel_features': dec_voxel_features,
                           'dec_voxel_coords': dec_voxel_coords,
                           'dec_voxel_num_points': dec_voxel_num_points,
                           'batch_size': int(batch_dict['record_len'].sum()),
                           'batch_len': self.batch_len,
                           'record_len': record_len,
                           })
        with torch.no_grad(): 
            # dec n, 4 -> n, c
            batch_dict = self.pillar_vfe(batch_dict, stage='dec')  
            # dec n, c -> N, C, H, W
            batch_dict = self.scatter_dec(batch_dict)
            batch_dict["rmpa_project_lidar"] = False
            # calculate pairwise affine transformation matrix
            _, _, H0, W0 = batch_dict['dec_spatial_features'].shape  # original feature map shape H0, W0
            normalized_affine_matrix = normalize_pairwise_tfm(batch_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

            batch_dict["normalized_affine_matrix"] = normalized_affine_matrix

            spatial_features = batch_dict['dec_spatial_features']

            if self.compression:
                spatial_features = self.naive_compressor(spatial_features)

            # multiscale backbone
            feature_list = self.backbone.get_multiscale_feature(spatial_features)

            mv_feature = self.backbone.decode_multiscale_feature(feature_list)

            for i, t in enumerate(feature_list):
                batch_dict["dec_spatial_features_%dx" % 2 ** (i + 1)] = t
            
            if self.shrink_flag:
                mv_feature = self.shrink_conv(mv_feature)

            batch_dict['stage1_out'] = self.head(mv_feature)

            data_dict, output_dict = {}, {}
            data_dict['ego'], output_dict['ego'] = batch_dict, batch_dict

            # 返回nms后每个batch融合后的框
            pred_box3d_list, scores_list = \
                self.post_processor.post_process(data_dict, output_dict,
                                                stage1=True)
            batch_dict['det_boxes'] = pred_box3d_list
            batch_dict['det_scores'] = scores_list
            if batch_dict['det_boxes'] is None:
                output_dict = None
                return output_dict
        if pred_box3d_list is not None and self.train_stage2:
            batch_dict = self.rmpa(batch_dict)
            batch_dict = self.matcher(batch_dict)
            batch_dict = self.roi_head(batch_dict)
            
        # diff       
        self.get_processed_lidar(batch_dict)
        if batch_dict['processed_lidar'] is None:
            output_dict = None
            return output_dict
        
        if not self.is_inference:
            voxel_features = batch_dict['processed_lidar']['voxel_features']
            voxel_coords = batch_dict['processed_lidar']['voxel_coords']
            voxel_num_points = batch_dict['processed_lidar']['voxel_num_points']
            voxel_gt_mask = batch_dict['processed_lidar']['gt_masks']
            batch_dict.pop('processed_lidar')
            batch_dict.update({
                            'voxel_features': voxel_features,
                            'voxel_coords': voxel_coords,
                            'voxel_num_points': voxel_num_points,
                            'voxel_gt_mask': voxel_gt_mask
                            })
            # 得到低层BEV特征 [B,C,H,W] 
            batch_dict = self.pillar_vfe(batch_dict, stage='diff')
            batch_dict = self.scatter(batch_dict) 
            batch_dict = self.mdd(batch_dict)
            output_dict = {'pred_out': batch_dict['pred_out'],
                           'gt_noise' : batch_dict['gt_noise'],
                           'gt_x0':batch_dict['gt_x0'],
                           'target':batch_dict['target'],
                           }
            
        # 第二阶段预测输出 [42,256]  --> [42,256,1]    
        transformed_factors = [self.dete_convertor(factor) for factor in batch_dict['fused_object_factors']]
        rcnn_reg = [self.reg_layers(factor.unsqueeze(dim = 2)).transpose(1,2).contiguous().squeeze(dim=1) for factor in transformed_factors] # [42, 7]
        rcnn_iou = [self.iou_layers(factor.unsqueeze(dim = 2)).transpose(1,2).contiguous().squeeze(dim=1) for factor in transformed_factors] # [42, 1]
        rcnn_cls = [self.cls_layers(factor.unsqueeze(dim = 2)).transpose(1,2).contiguous().squeeze(dim=1) for factor in transformed_factors] # [42, 1]
        rcnn_cls, rcnn_reg, rcnn_iou = [torch.cat(x, dim=0) for x in [rcnn_cls, rcnn_reg, rcnn_iou]]
        output_dict['stage2_out'] = {
                                'rcnn_cls': rcnn_cls,
                                'rcnn_iou': rcnn_iou,
                                'rcnn_reg': rcnn_reg,
                                }
        batch_dict['stage2_out'] = output_dict['stage2_out']
        output_dict['rcnn_label_dict'] = batch_dict['rcnn_label_dict']

        return output_dict