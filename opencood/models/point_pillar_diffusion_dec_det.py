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
# from opencood.models.mdd_modules.radar_cond_diff_denoise import Cond_Diff_Denoise
from opencood.models.mdd_modules.my_radar_cond_diff_denoise import Cond_Diff_Denoise
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from opencood.utils import common_utils
from opencood.tools import train_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.augmentor.augment_utils import global_rotation
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.common_utils import merge_features_to_dict
from opencood.models.sub_modules.cia_ssd_utils import SSFA, Head
from opencood.models.sub_modules.rmpa import Resolutionaware_Multiscale_Progressive_Attention
from opencood.models.sub_modules.roi_head_with_jo import RoIHead
from opencood.models.sub_modules.matcher_new import Matcher
from opencood.data_utils.post_processor.fpvrcnn_postprocessor import \
    FpvrcnnPostprocessor
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, When2commFusion
from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor
from opencood.data_utils.datasets.early_fusion_dataset_diffusion import visualize_gt_boxes
from opencood.visualization.simple_vis import visualize_averaged_channels_individual,visualize_channels_individually,visualize_all_channels_grid

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
        
        # self.voxel_preprocessor = build_preprocessor(args['preprocess'], train=True)
        # self.fusion_net = nn.ModuleList()
        # for i in range(len(args['base_bev_backbone']['layer_nums'])):
        #     if args['fusion_method'] == "max":
        #         self.fusion_net.append(MaxFusion())
        #     if args['fusion_method'] == "att":
        #         self.fusion_net.append(AttFusion(args['att']['feat_dim'][i]))
        # self.out_channel = args['mdd_block']['model']['out_ch']
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
        
        self.mdd = Cond_Diff_Denoise(args['mdd_block'], 32)
        
        # fc_layers = [args['roi_head']['n_fc_neurons']] * 2
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
        # self.cls_layers, pre_channel = self._make_fc_layers(pre_channel, fc_layers,
        #                                                     output_channels=args['roi_head']['num_cls'])
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
    
    def get_processed_lidar(self, batch_dict):
        processed_lidar_batch = []
        ori_lidar = batch_dict['origin_lidar_for_diff'][:, 1:]
        batch_indices = batch_dict['origin_lidar_for_diff'][:, 0].long()
        # 推理的时候没有早期融合的点云，获取每个视角已经投影到ego坐标系的点云，与已投影的boxes_fused对应，但我推理的时候也不用diff呀
        # ori_lidar = batch_dict['origin_lidar_for_vsa_project'][:, 1:]
        # batch_indices = batch_dict['origin_lidar_for_vsa_project'][:, 0].long()
        for batch_i in range(len(batch_dict['boxes_fused'])):
            bs_mask = (batch_indices == batch_i)
            batch_ori_lidar = ori_lidar[bs_mask].cpu().numpy()  # (N, 4)
            pre_fused_boxes = batch_dict['boxes_fused'][batch_i].cpu().numpy() # (n, 7)
            # pc_range = [-140.8, -40, -3, 140.8, 40, 1]
            # visualize_gt_boxes(pre_fused_boxes, batch_ori_lidar, pc_range, "/data/gkx/Code/opencood/vis_output/origin_pre_boxes.png")
            # 将box扩展到相同大小 3:6 hwl
            pre_fused_boxes[:, 3:6] = np.array(self.max_hwl)
            # visualize_gt_boxes(pre_fused_boxes, batch_ori_lidar, pc_range, "/data/gkx/Code/opencood/vis_output/expend_pre_boxes.png")        
            # 获取框中的点云  看一下boxhwl对不对应
            point_indices = points_in_boxes_cpu(batch_ori_lidar[:, :3], pre_fused_boxes[:,[0, 1, 2, 5, 4, 3, 6]]) 
            box_voxel_stack = []
            box_coords_stack = []
            box_num_points_stack = []
            box_masks = []
            rotation_angles = -pre_fused_boxes[:, 6].astype(float)
            for car_idx in range(len(pre_fused_boxes)):
                # 获取当前box中的点并平移到以box中心为原点的坐标系 box里没有点怎么办？？特征全为0吗？
                box_point = batch_ori_lidar[point_indices[car_idx] > 0]
                if len(box_point) < 100:
                    # print("此bxo中没有点云！！")
                    # pc_range = [-15, -15, -1, 15, 15, 1]
                    # visualize_gt_boxes(pre_fused_boxes[car_idx][np.newaxis, :], box_point, pc_range, f"/home/ubuntu/Code2/opencood/vis_output/pred_expand_{car_idx}.png",scale_bev=10)
                    continue
                
                box_point[:, :3] -= pre_fused_boxes[car_idx][0:3]
                
                # pre_fused_boxes[car_idx][0:3] = [0, 0, 0]
                # pc_range = [-15, -15, -1, 15, 15, 1]
                # visualize_gt_boxes(pre_fused_boxes[car_idx][np.newaxis, :], box_point, pc_range, f"/data/gkx/Code/opencood/vis_output/pred_expand_{car_idx+1}.png",scale_bev=10)
                # 旋转点云 
                box_point = common_utils.rotate_points_along_z(box_point[np.newaxis, :, :], np.array([rotation_angles[car_idx]]))[0]
                
                # pre_fused_boxes[car_idx][0:3] = common_utils.rotate_points_along_z(pre_fused_boxes[car_idx][np.newaxis, np.newaxis, 0:3], np.array([float(pre_fused_boxes[car_idx][6])]))[0,0]
                # pre_fused_boxes[car_idx][6] += rotation_angles[car_idx]
                # visualize_gt_boxes(pre_fused_boxes[car_idx][np.newaxis, :], box_point, pc_range, f"/data/gkx/Code/opencood/vis_output/pred_rotate_{car_idx+1}.png",scale_bev=10)
                # 体素化 不能并行！！
                processed_lidar_car = self.pre_processor.preprocess(box_point, is_car=True) #True
                box_voxel_stack.append(processed_lidar_car['voxel_features'])
                box_coords_stack.append(processed_lidar_car['voxel_coords'])
                box_num_points_stack.append(processed_lidar_car['voxel_num_points'])
                # 用unique找到所有值，没有的那个id再对应的boxes_fused里面pop掉
                box_masks.append(np.full(processed_lidar_car['voxel_features'].shape[0], car_idx, dtype=np.int32))
            
            if len(box_coords_stack) == 0:
                print(batch_dict['path'])
                print("所有box都没有点云？？")
                pc_range = [-140.8, -40, -3, 140.8, 40, 1]
                visualize_gt_boxes(pre_fused_boxes, batch_ori_lidar, pc_range, "./opencood/vis_output/origin_pre_boxes.png")
                processed_lidar = None # 此处只能设置batch_size为1
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
        # decouple
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
        with torch.no_grad(): #完全不计算梯度
            # dec n, 4 -> n, c
            batch_dict = self.pillar_vfe(batch_dict, stage='dec')  
            # dec n, c -> N, C, H, W
            batch_dict = self.scatter_dec(batch_dict)
            # 标注一下以后都不要投影
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
            
            # 不涉及任何投影
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
            # 先投影
            batch_dict = self.rmpa(batch_dict)
            # 用iou来寻找公共物体
            batch_dict = self.matcher(batch_dict)
            # 获取解耦因子
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
            batch_dict = self.scatter(batch_dict) # torch.Size([B, 10, 24, 28])
                
            # 将box抠出来的bev特征输入到mdd中
            # batch_dict['spatial_features'] = torch.randn(1, 10, 50, 50).to(voxel_features.device)
            batch_dict = self.mdd(batch_dict)
            output_dict = {'pred_out': batch_dict['pred_out'],
                           'gt_noise' : batch_dict['gt_noise'],
                           'gt_x0':batch_dict['gt_x0'],
                           'target':batch_dict['target'],
                           't': batch_dict['t'],
                           }
            # # 可视化特征
            # viz_config = [
            #     # ('batch_gt_spatial_features', 'gt_bev'),
            #     ('gt_x0', 'gt_x0'),
            #     # ('norm_gt_x0', 'norm_gt_x0'),
            #     # ('gt_noise', 'gt_noise'),
            #     # ('pred_out', 'pre_bev'),
            #     ('pred_out_inf_with_cond', 'pre_inf_with_cond_bev'),
            #     ('pred_out_inf_no_cond', 'pre_inf_no_cond_bev'),
            #     # ('pred_out_inf_no_cond_nonorm', 'pre_inf_with_no_bev_nonorm'),
            #     # ('noise', 'noise'),
            #     # ('x', 'x')
            # ]
            # base_path = f"/data/gkx/Code/opencood/bev_visualizations"
            # # 计算全局最小值和最大值
            # features = [batch_dict[key][0] for key, _ in viz_config]
            # global_vmin, global_vmax = float('inf'), float('-inf')
            # for feature in features:
            #     # channels = torch.mean(feature, dim=1).detach().cpu().numpy()
            #     channels = feature.detach().cpu().numpy()
            #     global_vmin = min(global_vmin, np.min(channels))
            #     global_vmax = max(global_vmax, np.max(channels))
            # # 可视化
            # for key, name in viz_config:
            #     visualize_channels_individually(
            #         batch_dict[key][0], 
            #         f"{base_path}/{name}_{i}", 
            #         # global_vmin, 
            #         # global_vmax
            #     )
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