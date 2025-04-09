import copy
import random

import torch
import torch.nn as nn

from opencood.pcdet_utils.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from opencood.pcdet_utils.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from opencood.utils import common_utils

from opencood.utils import box_utils

from opencood.models.sub_modules.my_msa import Muti_Stacked_Attention

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


class Resolutionaware_Multiscale_Progressive_Attention(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        SA_cfg = self.model_cfg['sa_layer']

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        c_in_list = []

        # print("当前的key",self.model_cfg['features_source'].keys())

        for k,v in self.model_cfg['features_source'].items():
            if k != "raw_points":
                c_in += v
                c_in_list.append(v)


        self.msa_points_fusion = True

        # print("msa points fusion:",c_in,c_in_list,self.model_cfg['num_out_features'])
        if not self.msa_points_fusion:
            self.vsa_point_feature_fusion = nn.Sequential(
                nn.Linear(c_in, self.model_cfg['num_out_features'], bias=False),
                nn.BatchNorm1d(self.model_cfg['num_out_features']),
                nn.ReLU(),
            )
        else:
            self.msa_point_feature_fusion = Muti_Stacked_Attention(c_in_list,self.model_cfg['num_out_features'])


        self.num_point_features = self.model_cfg['num_out_features']
        self.num_point_features_before_fusion = c_in

    #     测试
        self.c_in = c_in

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    def get_sampled_points(self, batch_dict):

        print("4096")
        batch_size = batch_dict['batch_size']

        # print("不同points shape: ",batch_dict['origin_lidar_for_vsa_project'].shape,batch_dict['origin_lidar_for_vsa_project'].shape,batch_dict['origin_lidar_for_vsa_noproject'].shape)
        if self.model_cfg['point_source'] == 'raw_points':
            src_points = batch_dict['origin_lidar_for_vsa_project'][:, 1:]
            batch_indices = batch_dict['origin_lidar_for_vsa_project'][:, 0].long()
        elif self.model_cfg['point_source'] == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError

        keypoints_batch = torch.randn((batch_size, self.model_cfg['num_keypoints'], 4), device=src_points.device)
        keypoints_batch[..., 0] = keypoints_batch[..., 0] * 140
        keypoints_batch[..., 1] = keypoints_batch[..., 0] * 40
        # points with height flag 10 are padding/invalid, for later filtering
        keypoints_batch[..., 2] = 10.0
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)

            # print("每辆车的点运数量",sampled_points.shape)
            # sample points with FPS
            # some cropped pcd may have very few points, select various number
            # of points to ensure similar sample density
            # 50000 is approximately the number of points in one full pcd
            num_kpts = int(self.model_cfg['num_keypoints'] * sampled_points.shape[1] / 50000) + 1
            num_kpts = min(num_kpts, self.model_cfg['num_keypoints'])
            cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                sampled_points[:, :, 0:3].contiguous(), num_kpts
            ).long()

            if sampled_points.shape[1] < num_kpts:
                empty_num = num_kpts - sampled_points.shape[1]
                cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

            keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            keypoints_batch[bs_idx, :len(keypoints[0]), :] = keypoints

        # keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints_batch


    def get_all_points(self, batch_dict):
        batch_size = batch_dict['batch_size']
        # print("all,  ",batch_size,batch_dict['record_len'])

        # print("不同points shape: ",batch_dict['origin_lidar_for_vsa_project'].shape,batch_dict['origin_lidar_for_vsa_project'].shape,batch_dict['origin_lidar_for_vsa_noproject'].shape)
        if self.model_cfg['point_source'] == 'raw_points':
            # 根据投影要求选择是否投影
            if "rmpa_project_lidar" in batch_dict and not batch_dict['rmpa_project_lidar']:
                src_points = batch_dict['origin_lidar_for_vsa_noproject'][:, 1:]
                batch_indices = batch_dict['origin_lidar_for_vsa_noproject'][:, 0].long()
            else:
                # 默认是投影的
                src_points = batch_dict['origin_lidar_for_vsa_project'][:, 1:]
                batch_indices = batch_dict['origin_lidar_for_vsa_project'][:, 0].long()
        elif self.model_cfg['point_source'] == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        tmp_bs_idx_points = []
        max_pointnum_batch = 0
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            # print("每辆车的点云数量",sampled_points.shape)
            tmp_bs_idx_points.append(sampled_points)
            if sampled_points.shape[1]>max_pointnum_batch:
                max_pointnum_batch = sampled_points.shape[1]

        keypoints_batch = torch.zeros((batch_size, max_pointnum_batch, 4), device=src_points.device)
        # points with height flag 10 are padding/invalid, for later filtering
        keypoints_batch[..., 2] = 10.0

        for bs_idx in range(batch_size):

            sampled_points = tmp_bs_idx_points[bs_idx]
            keypoints_batch[bs_idx, :sampled_points.shape[1], :] = sampled_points


        return keypoints_batch

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """

        # keypoints = self.get_sampled_points(batch_dict) # BxNx4

        keypoints = self.get_all_points(batch_dict) # BxNx4
        # 有没有说法，不要fps了，直接拿全部来计算
        # print("key points shape",keypoints.shape,batch_dict['origin_lidar_for_vsa_project'].shape,batch_dict['batch_size'],keypoints.shape)


        # kpt_mask1 = torch.logical_and(keypoints[..., 2] > -2.8, keypoints[..., 2] < 1.0)
        kpt_mask1 = torch.logical_and(keypoints[..., 2] > -3, keypoints[..., 2] < 1.0)

        kpt_mask2 = None
        # Only select the points that are in the predicted bounding boxes
        if 'det_boxes' in batch_dict:
            dets_list = batch_dict['det_boxes']
            max_len = max([len(dets) for dets in dets_list])
            boxes = torch.zeros((len(dets_list), max_len, 7), dtype=dets_list[0].dtype,
                                device=dets_list[0].device)
            for i, dets in enumerate(dets_list):
                # 这个是新增的部分
                dets = dets[:, [0, 1, 2, 5, 4, 3, 6]]  # hwl -> lwh
                if len(dets)==0:
                    continue
                cur_dets = dets.clone()

                if self.model_cfg['enlarge_selection_boxes']:
                    # print(self.model_cfg['enlarge_selection_boxes'])
                    cur_dets[:, 3:6] += 0.5
                boxes[i, :len(dets)] = cur_dets
            # mask out some keypoints to spare the GPU storage
            kpt_mask2 = points_in_boxes_gpu(keypoints[..., :3], boxes) >= 0
            # print("point_in_box:",boxes.shape,kpt_mask2.shape,kpt_mask1.shape,keypoints.shape)

        kpt_mask = torch.logical_and(kpt_mask1, kpt_mask2) if kpt_mask2 is not None else kpt_mask1
        # Ensure there are more than 2 points are selected to satisfy the
        # condition of batch norm in the FC layers of feature fusion module

        kpt_mask_flag = False
        if (kpt_mask).sum() < 2:
            # 此处可能在原本没有框的视角中选中两个点，后续我会对没有框的视角进行mask，导致最终点数不一致
            kpt_mask[0, torch.randint(0, 1024, (2,))] = True
            # 加个flag吧，后面如果flag成立，默认加入第一个视角中的点？？
            kpt_mask_flag = True
        batch_dict.update({'kpt_mask_flag': kpt_mask_flag})
        
        point_features_list = []

        for k in self.model_cfg['features_source'].keys():
            if k != "raw_points":
                point_bev_features = self.interpolate_from_bev_features(
                    keypoints[..., :3], batch_dict[k], batch_dict['batch_size'],
                    bev_stride=self.model_cfg['features_stride'][k]
                )
                point_features_list.append(point_bev_features[kpt_mask])


        batch_size, num_keypoints, _ = keypoints.shape

        new_xyz = keypoints[kpt_mask]
        # 这里会没有点？
        new_xyz_batch_cnt = torch.tensor([(mask).sum() for mask in kpt_mask], device=new_xyz.device).int()

        if self.msa_points_fusion:
            point_features = self.msa_point_feature_fusion(point_features_list)
        else:
            point_features = torch.cat(point_features_list, dim=1)
            batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
            point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))


        cur_idx = 0
        batch_dict['point_features'] = []
        batch_dict['point_coords'] = []
        for num in new_xyz_batch_cnt:
            batch_dict['point_features'].append(point_features[cur_idx:cur_idx + num])
            batch_dict['point_coords'].append(new_xyz[cur_idx:cur_idx + num])
            cur_idx += num

        return batch_dict
