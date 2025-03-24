import random, os

import torch
from torch import nn
import numpy as np

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.cia_ssd_utils import SSFA, Head


from opencood.models.sub_modules.rmpa import Resolutionaware_Multiscale_Progressive_Attention

# from opencood.models.sub_modules.roi_head_with_sd import RoIHead

from opencood.models.sub_modules.roi_head_with_jo import RoIHead


# draw pic
# from opencood.models.sub_modules.roi_head_compare_plt import RoIHead

# from opencood.models.sub_modules.matcher_compare import Matcher

# new matcher
from opencood.models.sub_modules.matcher_new import Matcher


# 可以基于coalign做出第二版，稍后完善

from opencood.data_utils.post_processor.fpvrcnn_postprocessor import \
    FpvrcnnPostprocessor


from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


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

class SDCoper2(nn.Module):
    def __init__(self, args):
        super(SDCoper2, self).__init__()

        lidar_range = np.array(args['lidar_range'])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) /
                             np.array(args['voxel_size'])).astype(np.int64)

        self.train_stage2 = args['activate_stage2']
        self.discrete_ratio = args['voxel_size'][0]

        # 新增的进行cobevt的部分
        self.max_cav = args['max_cav']

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", True)  # default true
        if is_resnet:
            print("is_resnet")
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'],
                                              64)  # or you can use ResNetBEVBackbone, which is stronger
        else:
            print("not_resnet")
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'],
                                            64)  # or you can use ResNetBEVBackbone, which is stronger
        self.voxel_size = args['voxel_size']

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
        self.post_processor = FpvrcnnPostprocessor(args['post_processer'],
                                                   train=True)

        self.rmpa = Resolutionaware_Multiscale_Progressive_Attention(args['vsa'], args['voxel_size'],
                                       args['lidar_range'],
                                       num_bev_features=128,
                                       num_rawpoint_features=3)

        self.matcher = Matcher(args['matcher'], args['lidar_range'])
        self.roi_head = RoIHead(args['roi_head'])

        # 新增阈值 小于才用特征
        self.feature_threshold = 0
        # 不确定性权重
        self.uncertainty_weight = 0.1
        # self.graph_matcher = GraphMatcher()


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


        for p in self.pillar_vfe.modules():
            if isinstance(p, nn.BatchNorm2d) or isinstance(p, nn.BatchNorm1d) or isinstance(p, nn.BatchNorm3d):
                p.eval()

        for p in self.scatter.modules():
            if isinstance(p, nn.BatchNorm2d) or isinstance(p, nn.BatchNorm1d) or isinstance(p, nn.BatchNorm3d):
                p.eval()

        for p in self.backbone.modules():
            if isinstance(p, nn.BatchNorm2d) or isinstance(p, nn.BatchNorm1d) or isinstance(p, nn.BatchNorm3d):
                p.eval()

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

        for p in self.head.modules():
            if isinstance(p, nn.BatchNorm2d) or isinstance(p, nn.BatchNorm1d) or isinstance(p, nn.BatchNorm3d):
                p.eval()


    def forward(self, batch_dict):

        voxel_features = batch_dict['processed_lidar']['voxel_features']
        voxel_coords = batch_dict['processed_lidar']['voxel_coords']
        voxel_num_points = batch_dict['processed_lidar']['voxel_num_points']
        record_len = batch_dict['record_len']
        # spatial_correction_matrix = data_dict['spatial_correction_matrix']

        # save memory
        batch_dict.pop('processed_lidar')
        batch_dict.update({'voxel_features': voxel_features,
                           'voxel_coords': voxel_coords,
                           'voxel_num_points': voxel_num_points,
                           'batch_size': int(batch_dict['record_len'].sum()),
                           'record_len': record_len,
                           # 这两个是coalign版本特供的，咱也不知道是啥
                           'proj_first': batch_dict['proj_first'],
                           'lidar_pose': batch_dict['lidar_pose']
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

        if self.compression:
            spatial_features = self.naive_compressor(spatial_features)

        # multiscale fusion
        feature_list = self.backbone.get_multiscale_feature(spatial_features)

        # print("特征列表尺度：",[t.shape for t in feature_list])
        mv_feature = self.backbone.decode_multiscale_feature(feature_list)
        # print("合并后单车特征尺度：",mv_feature.shape)

        # fused_feature_list = [].
        # for i, fuse_module in enumerate(self.fusion_net):
        #     fused_feature_list.append(fuse_module(feature_list[i], record_len, normalized_affine_matrix))
        # fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)

        # print("融合后特征列表尺度：",[t.shape for t in fused_feature_list])
        # print("融合后特征尺度：",fused_feature.shape)

        for i, t in enumerate(feature_list):

            out = normalized_affine_bev(t, normalized_affine_matrix, record_len)
            batch_dict["spatial_features_%dx" % 2 ** (i + 1)] = out

        batch_dict['spatial_features'] = normalized_affine_bev(batch_dict['spatial_features'],
                                                                   normalized_affine_matrix, record_len)
        if self.shrink_flag:
            # fused_feature = self.shrink_conv(fused_feature)
            mv_feature = self.shrink_conv(mv_feature)

        # print("送入分类头特征尺度: ",fused_feature.shape)
        # print("送入分类头单车特征尺度: ",mv_feature.shape)

        batch_dict['stage1_out'] = self.head(mv_feature)

        data_dict, output_dict = {}, {}
        data_dict['ego'], output_dict['ego'] = batch_dict, batch_dict


        # choose output in sd
        # 返回nms后每个batch融合后的框
        pred_box3d_list, scores_list = \
            self.post_processor.post_process(data_dict, output_dict,
                                             stage1=True)

        # print([t.shape for t in pred_box3d_list],[t.shape for t in scores_list])
        # # 这就是我们第一阶段的结果，我们要认真使用，A，获取特征（在后续实现，先不动这个），B打包 cls和box等信息 
        # Vehicles = []
        # Graphs = []
        # for boxes,scores in zip(pred_box3d_list,scores_list):
        #     Vehicle_tmp = []
        #     for b,s in zip(boxes,scores):
        #         Vehicle_tmp.append(Vehicle(box=b.cpu(),score=s.cpu(),threshold=self.feature_threshold))
        #     Vehicles.append(Vehicle_tmp)
        #     # for t in 
        #     Graphs.append(VehicleGraph(Vehicle_tmp))

        # # 先不考虑特征，把点对拿出来看看，同时把他们的gt拿出来看看，gt是roihead算出来的，我之前设计下，应该能把每个单车box找到对应gt
        # # 这里没有project_first,每个车都在ego下面，这种对其效果，好像不太能GT得找的上才行，这里我们就不考虑了，直接算，直接投影

        # # 这东西是乱序的，跟GT对应不上  
        # # 场景下多辆车才匹配
        # if len(Graphs)>1:
        #     for i in range(1,len(Graphs)):
        #         matches, score = self.graph_matcher.match(Graphs[0], Graphs[i], 0.1)
        #         print(f"ego vs car_{i} 匹配结果: {matches}")
        #         print(f"ego vs car{i} 匹配分数: {score}")

        # 真正的对应关系呢，让我找一找 


        batch_dict['det_boxes'] = pred_box3d_list
        batch_dict['det_scores'] = scores_list



        # gt_boxes = [b[m][:, [0, 1, 2, 3, 4, 5, 6]].float() for b, m in
        #             zip(batch_dict['object_bbx_center'],
        #                 batch_dict['object_bbx_mask'].bool())]  # hwl -> lwh order 这里不转换，后面会转换
        
        # for i in range(len(gt_boxes)):
        #     gt_boxes[i][:,3:6] -= 0.5
        # gt_score = [torch.ones((t.shape[0]),device = t.device) for t in gt_boxes]
        # print("gt shape: ",[t.shape for t in gt_boxes],[t.shape for t in gt_score])
        # pred_box3d_list, scores_list = gt_boxes, gt_score


        if pred_box3d_list is not None and self.train_stage2:

            batch_dict = self.rmpa(batch_dict)
            batch_dict = self.matcher(batch_dict)
            batch_dict = self.roi_head(batch_dict)

        return batch_dict