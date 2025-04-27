"""
Pillar VFE, credits to OpenPCDet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(
                inputs[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2,
                                                  1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size,
                 point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg

        self.use_norm = self.model_cfg['use_norm']
        self.with_distance = self.model_cfg['with_distance']

        self.use_absolute_xyz = self.model_cfg['use_absolute_xyz']
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg['num_filters']
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                         last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num,
                               dtype=torch.int,
                               device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def analyze_features(self, statistical_features):
        """分析特征的统计分布
        Args:
            statistical_features: [M, 7] tensor, 包含相对偏移[0:3]，方差[3:6]，最大距离[6]
        """
        # 转换为numpy进行分析
        features = statistical_features.detach().cpu().numpy()
        
        # 创建子图
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        
        # 1. 分析相对偏移量
        offsets = features[:, :3]
        axes[0].boxplot([offsets[:, i] for i in range(3)], labels=['x', 'y', 'z'])
        axes[0].set_title('Relative Offsets Distribution')
        axes[0].set_ylabel('Offset Value')
        
        # 2. 分析方差
        vars = features[:, 3:6]
        axes[1].boxplot([vars[:, i] for i in range(3)], labels=['x', 'y', 'z'])
        axes[1].set_title('Variance Distribution')
        axes[1].set_ylabel('Variance Value')
        
        # 3. 分析最大距离
        max_dist = features[:, 6]
        axes[2].hist(max_dist, bins=50)
        axes[2].set_title('Max Distance Distribution')
        axes[2].set_xlabel('Distance')
        axes[2].set_ylabel('Count')
        
        # 打印基本统计信息
        print("\nFeature Statistics:")
        print("\nRelative Offsets (x, y, z):")
        print(f"Mean: {np.mean(offsets, axis=0)}")
        print(f"Std: {np.std(offsets, axis=0)}")
        print(f"Min: {np.min(offsets, axis=0)}")
        print(f"Max: {np.max(offsets, axis=0)}")
        
        print("\nVariance (x, y, z):")
        print(f"Mean: {np.mean(vars, axis=0)}")
        print(f"Std: {np.std(vars, axis=0)}")
        print(f"Min: {np.min(vars, axis=0)}")
        print(f"Max: {np.max(vars, axis=0)}")
        
        print("\nMax Distance:")
        print(f"Mean: {np.mean(max_dist)}")
        print(f"Std: {np.std(max_dist)}")
        print(f"Min: {np.min(max_dist)}")
        print(f"Max: {np.max(max_dist)}")
        
        plt.tight_layout()
        plt.savefig('feature_analysis.png')
        plt.close()
    
    def forward(self, batch_dict, stage):
        """encoding voxel feature using point-pillar method
        Args:
            voxel_features: [M, 32, 4]
            voxel_num_points: [M,]
            voxel_coords: [M, 4]
        Returns:
            features: [M,64], after PFN
        """
        if stage == 'dec':  
            voxel_features, voxel_num_points, coords = \
                batch_dict['dec_voxel_features'], batch_dict['dec_voxel_num_points'], \
                batch_dict['dec_voxel_coords']
            points_mean = \
            voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
            voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
            # 点相对于体素内质心的偏移量
            f_cluster = voxel_features[:, :, :3] - points_mean
            # 点相对于体素中心的偏移量
            f_center = torch.zeros_like(voxel_features[:, :, :3])
            f_center[:, :, 0] = voxel_features[:, :, 0] - (
                    coords[:, 3].to(voxel_features.dtype).unsqueeze(
                        1) * self.voxel_x + self.x_offset)
            f_center[:, :, 1] = voxel_features[:, :, 1] - (
                    coords[:, 2].to(voxel_features.dtype).unsqueeze(
                        1) * self.voxel_y + self.y_offset)
            f_center[:, :, 2] = voxel_features[:, :, 2] - (
                    coords[:, 1].to(voxel_features.dtype).unsqueeze(
                        1) * self.voxel_z + self.z_offset)

            if self.use_absolute_xyz:
                features = [voxel_features, f_cluster, f_center]
            else:
                features = [voxel_features[..., 3:], f_cluster, f_center]

            if self.with_distance:
                points_dist = torch.norm(voxel_features[:, :, :3], 2, 2,
                                        keepdim=True)
                features.append(points_dist)
            features = torch.cat(features, dim=-1)

            voxel_count = features.shape[1]
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count,
                                            axis=0)
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
            features *= mask # torch.Size([N, 16, 10])
            for pfn in self.pfn_layers:
                features = pfn(features)
            features = features.squeeze()
            batch_dict['dec_pillar_features'] = features
        else: # stage == 'diff'
            voxel_features, voxel_num_points, coords = \
                batch_dict['voxel_features'], batch_dict['voxel_num_points'], \
                batch_dict['voxel_coords']
            # 创建mask来标识有效点 [M, 16, 1]
            points_mask = torch.arange(voxel_features.shape[1], device=voxel_features.device) \
                .unsqueeze(0).expand(voxel_features.shape[0], -1) \
                < voxel_num_points.unsqueeze(-1)
            points_mask = points_mask.unsqueeze(-1)
            
            # 使用mask获取有效点的xyz坐标 [M, 16, 3]
            valid_points = voxel_features[:, :, :3] * points_mask.float()
            
            # 1. 计算相对于质心的平均偏移量 [M, 3]
            # 首先计算质心
            points_mean = valid_points.sum(dim=1) / voxel_num_points.unsqueeze(-1).float()  # [M, 3]
            
            # 计算每个点到质心的相对偏移
            relative_offsets = valid_points - points_mean.unsqueeze(1)  # [M, 16, 3]
            
            # 计算有效点的平均相对偏移（使用mask确保只考虑有效点）
            mean_relative_offsets = (relative_offsets * points_mask.float()).sum(dim=1) / \
                                voxel_num_points.unsqueeze(-1).float()  # [M, 3]
            
            # 2. 计算点云方差 Variance [M, 3]
            # 广播减法计算每个点到均值的差
            centered_points = valid_points - points_mean.unsqueeze(1)  # [M, 16, 3]
            # 计算平方
            squared_diff = (centered_points * points_mask.float()) ** 2  # [M, 16, 3]
            # 使用sum和voxel_num_points计算方差
            variance = squared_diff.sum(dim=1) / voxel_num_points.unsqueeze(-1).float()  # [M, 3]
            
            # 3. 计算最大点间距离 Diameter [M, 1]
            # 使用广播计算所有点对之间的距离
            # reshape以便使用cdist
            # 检查是否有有效点
            if points_mask.any():
                # reshape以便使用cdist
                points_expanded = valid_points.view(-1, voxel_features.shape[1], 3)  # [M, 16, 3]
                # 计算每个柱体内所有点对的距离
                distances = torch.cdist(points_expanded, points_expanded) # [M, 16, 16]
                # 创建mask来处理无效点对
                mask_matrix = points_mask.squeeze(-1).unsqueeze(-1) * points_mask.squeeze(-1).unsqueeze(1) # [M, 16, 16]
                # 将无效点对的距离设为0
                distances = distances * mask_matrix
                
                # 检查每个柱体是否有点
                valid_voxels = mask_matrix.sum(dim=(1,2)) > 0
                if valid_voxels.any():
                    # 只对有效柱体计算最大距离
                    valid_distances = distances[valid_voxels]
                    max_dists = torch.max(valid_distances.view(valid_distances.shape[0], -1), dim=1)[0]
                    # 初始化全零张量
                    max_distances = torch.zeros((distances.shape[0], 1), device=distances.device)
                    max_distances[valid_voxels] = max_dists.unsqueeze(-1) # [M, 1]
                else:
                    max_distances = torch.zeros((distances.shape[0], 1), device=distances.device) # [M, 1]
            else:
                # 如果没有有效点，返回零张量
                max_distances = torch.zeros((valid_points.shape[0], 1), device=valid_points.device) # [M, 1]
            
            # 将voxel_num_points转换为浮点数并增加维度 [M, 1]
            points_count = voxel_num_points.float().unsqueeze(-1)
            
            # 拼接所有特征 [M, 8] (原来是[M, 7])
            statistical_features = torch.cat([mean_relative_offsets, variance, max_distances, points_count], dim=1)
            
            # 处理只有一个点的情况（方差和最大距离应为0）
            single_point_mask = (voxel_num_points == 1).unsqueeze(-1)  # [M, 1]
            statistical_features[:, 1:4] = statistical_features[:, 1:4] * (~single_point_mask)  # 注意这里改为1:4，因为points_count不需要置零
            
            # 将统计特征添加到batch_dict中
            statistical_features = self.normalize_statistical_features(statistical_features)
            batch_dict['pillar_features'] = statistical_features
            
            # 在返回之前添加分析
            if False:  # 可以选择只在训练时分析
                self.analyze_features(statistical_features)
        
            # features = features.view(features.size(0), 8, 20) # 将 [N, 16, 10] 重塑为 [N, 8, 20]
            # features = features.mean(dim=1) # 在第二维(dim=1)上取平均，结果为 [N, 20]
            # features = features.squeeze()
            # batch_dict['pillar_features'] = features

        return batch_dict
    
    def normalize_statistical_features(self, statistical_features):
        """
        对统计特征进行通道级别的归一化
        输入: statistical_features [M, 8] - 包含mean_relative_offsets(3), variance(3), max_distances(1), points_count(1)
        输出: 归一化后的特征 [M, 8]
        """
        # 复制一份特征，避免修改原始数据
        normalized_features = statistical_features.clone()
        
        # 分别对不同类型的特征进行归一化
        # 1. 对mean_relative_offsets进行归一化 (前3个通道)
        for i in range(3):
            channel_data = normalized_features[:, i]
            
            # 方式1: 如果值很小，使用标准化而非最小-最大缩放
            std_val = channel_data.std()
            if std_val > 1e-10:  # 使用更小的阈值
                # 使用标准差归一化，保留符号信息
                normalized_features[:, i] = channel_data / (3 * std_val)  # 3倍标准差通常覆盖大部分数据
            else:
                # 如果标准差极小，使用绝对值最大值归一化
                max_abs = torch.max(torch.abs(channel_data))
                if max_abs > 1e-10:
                    normalized_features[:, i] = channel_data / max_abs
                else:
                    normalized_features[:, i] = torch.zeros_like(channel_data)
        
        # 2. 对variance进行归一化 (3-6通道)
        for i in range(3, 6):
            channel_data = normalized_features[:, i]
            # variance通常是非负的，可以使用最大值归一化
            max_val = channel_data.max()
            if max_val > 1e-6:
                normalized_features[:, i] = channel_data / max_val
            else:
                normalized_features[:, i] = torch.zeros_like(channel_data)
        
        # 3. 对max_distances进行归一化 (第6个通道)
        max_dist = normalized_features[:, 6]
        max_val = max_dist.max()
        if max_val > 1e-6:
            normalized_features[:, 6] = max_dist / max_val
        
        # 4. 对points_count进行归一化 (第7个通道)
        # 通常点数是整数，可以考虑除以最大点数或者用log变换
        points_count = normalized_features[:, 7]
        max_points = points_count.max()
        if max_points > 0:
            # 方案1: 直接除以最大值
            # normalized_features[:, 7] = points_count / max_points
            
            # 方案2: 对点数使用log变换再归一化，能更好地处理点数分布
            log_points = torch.log1p(points_count)  # log(1+x)避免log(0)
            max_log = log_points.max()
            if max_log > 0:
                normalized_features[:, 7] = log_points / max_log
        
        return normalized_features
