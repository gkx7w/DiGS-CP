import random, os

import torch
from torch import nn
import numpy as np

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.cia_ssd_utils import SSFA, Head


from opencood.models.sub_modules.rmpa import Resolutionaware_Multiscale_Progressive_Attention

from opencood.models.sub_modules.roi_head_with_sd import RoIHead


from opencood.models.sub_modules.matcher_compare import Matcher


from opencood.data_utils.post_processor.fpvrcnn_postprocessor import \
    FpvrcnnPostprocessor


from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

# stage1 backbone

from icecream import ic
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, When2commFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm


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

class DiscoNetSDCoper(nn.Module):
    def __init__(self, args):
        super(DiscoNetSDCoper, self).__init__()

        lidar_range = np.array(args['lidar_range'])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) /
                             np.array(args['voxel_size'])).astype(np.int64)

        self.train_stage2 = args['activate_stage2']
        self.discrete_ratio = args['voxel_size'][0]

        self.max_cav = args['max_cav']

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])

        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])
        self.voxel_size = args['voxel_size']

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(64, args['compression'])

        self.fusion_net = DiscoFusion(self.out_channel)

        print("in head chanel: ",self.out_channel)
        self.head = Head(**args['head'])
        self.post_processor = FpvrcnnPostprocessor(args['post_processer'],
                                                   train=True)

        self.rmpa = Resolutionaware_Multiscale_Progressive_Attention(args['vsa'], args['voxel_size'],
                                       args['lidar_range'],
                                       num_bev_features=128,
                                       num_rawpoint_features=3)

        self.matcher = Matcher(args['matcher'], args['lidar_range'])
        self.roi_head = RoIHead(args['roi_head'])


    def stage1_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.fusion_net.parameters():
            p.requires_grad = False

        for p in self.head.parameters():
            p.requires_grad = False

        # 固定BN层所有参数，这次理论上是真的全部固定了
        for p in self.pillar_vfe.modules():
            if isinstance(p, nn.BatchNorm2d) or isinstance(p, nn.BatchNorm1d) or isinstance(p, nn.BatchNorm3d):
                p.eval()

        for p in self.scatter.modules():
            if isinstance(p, nn.BatchNorm2d) or isinstance(p, nn.BatchNorm1d) or isinstance(p, nn.BatchNorm3d):
                p.eval()

        for p in self.backbone.modules():
            if isinstance(p, nn.BatchNorm2d) or isinstance(p, nn.BatchNorm1d) or isinstance(p, nn.BatchNorm3d):
                p.eval()

        # 这个忘记固定了!!!!!! self.fusion_net，寄
        for p in self.fusion_net.modules():
            if isinstance(p, nn.BatchNorm2d) or isinstance(p, nn.BatchNorm1d) or isinstance(p, nn.BatchNorm3d):
                p.eval()

        if self.compression:
            for p in self.naive_compressor.modules():
                if isinstance(p, nn.BatchNorm2d) or isinstance(p, nn.BatchNorm1d) or isinstance(p, nn.BatchNorm3d):
                    p.eval()
        if self.shrink_flag:
            for p in self.shrink_conv.modules():
                if isinstance(p, nn.BatchNorm2d) or isinstance(p, nn.BatchNorm1d) or isinstance(p, nn.BatchNorm3d):
                    p.eval()

        # 这两个需要再优化一下
        for p in self.head.modules():
            if isinstance(p, nn.BatchNorm2d) or isinstance(p, nn.BatchNorm1d) or isinstance(p, nn.BatchNorm3d):
                p.eval()


    def forward(self, batch_dict):

        voxel_features = batch_dict['processed_lidar']['voxel_features']
        voxel_coords = batch_dict['processed_lidar']['voxel_coords']
        voxel_num_points = batch_dict['processed_lidar']['voxel_num_points']
        record_len = batch_dict['record_len']
        pairwise_t_matrix = batch_dict['pairwise_t_matrix']

        # save memory
        batch_dict.pop('processed_lidar')
        batch_dict.update({'voxel_features': voxel_features,
                           'voxel_coords': voxel_coords,
                           'voxel_num_points': voxel_num_points,
                           'batch_size': int(batch_dict['record_len'].sum()),
                           'record_len': record_len,
                           # 这两个是coalign版本特供的，咱也不知道是啥
                           'proj_first': batch_dict['proj_first'],
                           'lidar_pose': batch_dict['lidar_pose'],
                           'pairwise_t_matrix': pairwise_t_matrix
                           })
        # print("record_len:",record_len)

        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_dict['spatial_features'].shape  # original feature map shape H0, W0
        normalized_affine_matrix = normalize_pairwise_tfm(batch_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

        batch_dict["normalized_affine_matrix"] = normalized_affine_matrix

        spatial_features = batch_dict['spatial_features']

        batch_dict = self.backbone(batch_dict)

        # multiscale features
        feature_list = self.backbone.get_multiscale_feature(spatial_features)

        for i, t in enumerate(feature_list):
            # print("spatial_features_%dx"% 2**(i+1)," 修正前  ",t.shape)

            out = normalized_affine_bev(t, normalized_affine_matrix, record_len)
            batch_dict["spatial_features_%dx" % 2 ** (i + 1)] = out


        batch_dict['spatial_features'] = normalized_affine_bev(batch_dict['spatial_features'],
                                                                   normalized_affine_matrix, record_len)

        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        spatial_features_2d = self.fusion_net(spatial_features_2d, record_len, normalized_affine_matrix)

        batch_dict['stage1_out'] = self.head(spatial_features_2d)


        data_dict, output_dict = {}, {}
        data_dict['ego'], output_dict['ego'] = batch_dict, batch_dict

        pred_box3d_list, scores_list = \
            self.post_processor.post_process(data_dict, output_dict,
                                             stage1=True)
        pred_box3d_list_mv, scores_list_mv, fused_feature_list = [], [], []
        if pred_box3d_list != None:
            for i, v in enumerate(batch_dict['record_len']):
                for j in range(v):
                    pred_box3d_list_mv.append(pred_box3d_list[i])
                    scores_list_mv.append(scores_list[i])

        batch_dict['det_boxes_fused'] = pred_box3d_list
        batch_dict['det_scores_fused'] = scores_list
        batch_dict['det_boxes'] = pred_box3d_list_mv
        batch_dict['det_scores'] = scores_list_mv

        if pred_box3d_list is not None and self.train_stage2:

            batch_dict = self.rmpa(batch_dict)
            batch_dict = self.matcher(batch_dict)
            batch_dict = self.roi_head(batch_dict)


        return batch_dict