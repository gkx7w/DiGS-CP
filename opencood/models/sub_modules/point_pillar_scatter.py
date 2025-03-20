import torch
import numpy as np
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features']
        self.max_hwl = self.model_cfg['max_hwl']
        self.nx, self.ny, self.nz = model_cfg['grid_size']  # [56, 24, 1] or [704, 200, 1] 
        self.voxel_size = model_cfg['voxel_size'] 
        self.lidar_range = model_cfg['lidar_range']  

        assert self.nz == 1

    def forward(self, batch_dict):
        """ 将生成的pillar按照坐标索引还原到原空间中
        Args:
            pillar_features:(M, 64)
            coords:(M, 4) 第一维是batch_index

        Returns:
            batch_spatial_features:(4, 64, H, W)
            
            |-------|
            |       |             |-------------|
            |       |     ->      |  *          |
            |       |             |             |
            | *     |             |-------------|
            |-------|

            Lidar Point Cloud        Feature Map
            x-axis up                Along with W 
            y-axis right             Along with H

            Something like clockwise rotation of 90 degree.

        """
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        gt_mask = batch_dict['voxel_gt_mask']
        
        batch_size = coords[:, 0].max().int().item() + 1
        
        # 为每个batch创建一个字典，用于存储该batch中每个gt box的spatial features
        batch_gt_spatial_features = []
        
        for batch_idx in range(batch_size):
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]  # (batch_idx_voxel, 4)
            # 获取该batch中的gt_mask
            this_gt_mask = gt_mask[batch_mask]  # (batch_idx_voxel, )
            # 为该batch创建一个字典，用于存储每个gt box的spatial features
            gt_spatial_features_dict = {}           
            assert len(this_coords) > 0, f"batch {batch_idx} 的坐标不能为空"
            # 对每个GT box单独处理
            unique_gt_ids = torch.unique(this_gt_mask[this_gt_mask >= 0])
            for gt_id in unique_gt_ids:
                # 为每个GT box创建一个独立的spatial feature
                gt_spatial_feature = torch.zeros(
                    self.num_bev_features,
                    self.nz * self.nx * self.ny,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device)
                # 选择属于当前gt_box的pillar !!每个gt的voxel数量不一致
                gt_pillar_mask = this_gt_mask == gt_id
                gt_pillars = pillar_features[batch_mask][gt_pillar_mask]  # (gt_box_voxel, 64)
                gt_coords = this_coords[gt_pillar_mask]  # (gt_box_voxel, 4)
                assert len(gt_coords) > 0, f"batch {batch_idx} 中 gt_id {gt_id.item()} 的坐标不能为空"
                # 计算indices
                indices = gt_coords[:, 1] + gt_coords[:, 2] * self.nx + gt_coords[:, 3]
                indices = indices.type(torch.long)
                # 转置特征
                gt_pillars = gt_pillars.t()  # (64, gt_box_voxel)
                # 将特征填充到该gt box的空间特征图中
                gt_spatial_feature[:, indices] = gt_pillars
                # 将特征图还原为标准BEV特征图格式
                gt_spatial_feature = gt_spatial_feature.view(
                    self.num_bev_features * self.nz, self.ny, self.nx)
                # 将该gt box的spatial feature存储到字典中
                gt_spatial_features_dict[gt_id.item()] = gt_spatial_feature
            # 将该batch的gt空间特征字典添加到batch列表中 ！！此时每个gtbev大小一致，可堆叠为tensor
            batch_gt_spatial_features.append(torch.stack(list(gt_spatial_features_dict.values())))
        
        batch_dict['batch_gt_spatial_features'] = batch_gt_spatial_features
        
        return batch_dict

